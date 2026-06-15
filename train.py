"""
MAPPO training loop with:
  • vectorised environments (multiprocessing)
  • non-blocking IPC (multiprocessing.connection.wait)
  • curriculum learning (stepwise grid scaling)
  • GAE advantage estimation
  • clipped PPO surrogate + entropy bonus
  • annealed behavioural-cloning (BC) auxiliary loss
  • batched forward passes across all environments
  • separate actor/critic gradient clipping + return normalisation
  • sparse cross-entropy BC loss
  • JSON-lines metric logging + live curses terminal dashboard
"""

import glob
import math
import sys
import sys
import time
import json
import os
import multiprocessing as mp
from multiprocessing.connection import wait as mp_wait

os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from net    import GhostActor, GhostCritic
from worker import Env
from obs    import MAX_H, MAX_W, MAX_GHOSTS, SPATIAL_CH
from curriculum import CurriculumScheduler, STAGES

# ── hyperparameters ──────────────────────────────────────────────────
NUM_ENVS        = 8     # 8 is the perfect sweet spot for a 6-core/12-thread CPU
ROLLOUT_STEPS   = 256
MINI_BATCH      = 1024  # 512 prevents the 6GB VRAM spike and OOM
PPO_EPOCHS      = 12    # Kept at 12 to maximize learning per rollout
GAMMA           = 0.99
GAE_LAMBDA      = 0.95
CLIP_EPS        = 0.2
ENT_COEF        = 0.002
VF_COEF         = 0.5
MAX_GRAD_NORM   = 0.5
LR              = 3e-4
BC_ANNEAL_EP    = 2000
BC_INIT         = 0.5     # user requested reduction to prevent flattening actor loss
BC_FLOOR        = 0.02
K_NOMINATIONS   = 3
LOG_DIR         = os.path.join(os.path.dirname(__file__), "logs")
CKPT_DIR        = os.path.join(os.path.dirname(__file__), "checkpoints")
BC_ANNEAL_UPDATES = 2000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── padding helper ───────────────────────────────────────────────────

def _pad_spatial(arr, target_h=MAX_H, target_w=MAX_W):
    """Pad (*, H, W) numpy array to (*, target_h, target_w)."""
    h, w = arr.shape[-2], arr.shape[-1]
    if h == target_h and w == target_w:
        return arr
    pad_h = target_h - h
    pad_w = target_w - w
    if arr.ndim == 3:
        return np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), constant_values=0)
    elif arr.ndim == 2:
        return np.pad(arr, ((0, pad_h), (0, pad_w)), constant_values=0)
    elif arr.ndim == 4:
        out = np.pad(arr, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), constant_values=0)
        if out.shape[1] == SPATIAL_CH:
            out[:, 0, h:, :] = 1.0
            out[:, 0, :, w:] = 1.0
        return out
    return arr


# ── vectorised environment ───────────────────────────────────────────

def _worker(env_id, conn, rows, cols, n_ghosts, n_power, static_pacman=False):
    """Child process: owns one Env, responds to step/reset/close/set_curriculum."""
    try:
        print(f"Worker {env_id} started")
        env = Env(env_id, num_ghosts=n_ghosts, grid_rows=rows, grid_cols=cols, n_power=n_power)
        env.static_pacman = static_pacman
        obs = env.reset()
        conn.send(obs)           # send initial observation
    except Exception as e:
        import traceback
        traceback.print_exc()
        conn.send(e)
        return
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            result = env.step(data)
            obs, rew, done, info = result
            if done:
                obs = env.reset()
                info["was_done"] = True
            conn.send((obs, rew, done, info))
        elif cmd == "reset":
            obs = env.reset()
            conn.send(obs)
        elif cmd == "set_curriculum":
            rows, cols, n_ghosts, n_power, static_pacman = data
            env.grid_rows = rows
            env.grid_cols = cols
            env.num_ghosts = n_ghosts
            env.n_power = n_power
            env.static_pacman = static_pacman
            obs = env.reset()
            conn.send(obs)
        elif cmd == "close":
            break


def _recv_unordered(conns, n):
    """Receive from n connections using wait() to avoid blocking in order."""
    results = [None] * n
    conn_to_idx = {id(c): i for i, c in enumerate(conns)}
    pending = set(conns)
    while pending:
        ready = mp_wait(pending)
        for conn in ready:
            i = conn_to_idx[id(conn)]
            results[i] = conn.recv()
            pending.remove(conn)
    return results


class VecEnv:
    def __init__(self, n, rows=33, cols=41, n_ghosts=7, n_power=28, static_pacman=False):
        self.n = n
        ctx = mp.get_context("spawn")
        self.parent, self.child = zip(*[ctx.Pipe() for _ in range(n)])
        self.procs = []
        for i, c in enumerate(self.child):
            p = ctx.Process(target=_worker,
                            args=(i, c, rows, cols, n_ghosts, n_power, static_pacman),
                            daemon=True)
            p.start()
            self.procs.append(p)
        # receive initial observations (non-blocking collect)
        self.current_obs = _recv_unordered(self.parent, n)

    def reset(self):
        for p in self.parent:
            p.send(("reset", None))
        self.current_obs = _recv_unordered(self.parent, self.n)
        return self.current_obs

    def set_curriculum(self, rows, cols, n_ghosts, n_power, static_pacman=False):
        """Reconfigure all workers for a new curriculum stage."""
        for p in self.parent:
            p.send(("set_curriculum", (rows, cols, n_ghosts, n_power, static_pacman)))
        self.current_obs = _recv_unordered(self.parent, self.n)
        return self.current_obs

    def step(self, actions: list[dict]):
        if not actions:
            print("vec_env.step() with empty actions")
            for p in self.parent:
                p.send(("step", {}))
            results = _recv_unordered(self.parent, self.n)
            print("vec_env.step() returned")
        else:
            for p, a in zip(self.parent, actions):
                p.send(("step", a))
            results = _recv_unordered(self.parent, self.n)
        
        obs_list, rew_list, done_list, info_list = [], [], [], []
        for i, (obs, rew, done, info) in enumerate(results):
            self.current_obs[i] = obs
            obs_list.append(obs)
            rew_list.append(rew)
            done_list.append(done)
            info_list.append(info)
        return obs_list, rew_list, done_list, info_list

    def close(self):
        for p in self.parent:
            p.send(("close", None))
        for p in self.procs:
            p.join(timeout=5)


# ── GAE ──────────────────────────────────────────────────────────────

def compute_gae(buf_rewards_e, buf_values_e, buf_dones_e, last_val_dict_e, gamma, lam):
    T = len(buf_rewards_e)
    adv_dict_list = [{} for _ in range(T)]
    ret_dict_list = [{} for _ in range(T)]
    
    gae = {}
    
    for t in reversed(range(T)):
        nt = 1.0 - buf_dones_e[t]
        for gid, r in buf_rewards_e[t].items():
            v_t = buf_values_e[t][gid]
            if t == T - 1:
                nv = last_val_dict_e.get(gid, 0.0)
            else:
                nv = buf_values_e[t+1].get(gid, 0.0)
            
            delta = r + gamma * nv * nt - v_t
            gae[gid] = delta + gamma * lam * nt * gae.get(gid, 0.0)
            adv_dict_list[t][gid] = gae[gid]
            ret_dict_list[t][gid] = gae[gid] + v_t
            
        for gid in list(gae.keys()):
            if gid not in buf_rewards_e[t]:
                gae[gid] = 0.0
                
    return adv_dict_list, ret_dict_list


# ── critic value helper ──────────────────────────────────────────────

def _critic_value(critic, spatial, vector, n_alive):
    """Run IPPO critic independently, return mean scalar."""
    if n_alive == 0:
        return torch.zeros(0, device=DEVICE)
    return critic(spatial, vector).squeeze(-1)                  # (n_alive,)


# ── main training loop ───────────────────────────────────────────────

def train():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, "metrics.jsonl")


    # ── curriculum setup ─────────────────────────────────────────────
    curriculum = CurriculumScheduler(start_stage=0)
    stage = curriculum.stage
    print(f"Curriculum: starting at Stage {curriculum.stage_idx} "
          f"({stage.rows}×{stage.cols}, {stage.n_ghosts} ghosts)")

    print("Initializing VecEnv (spawn before CUDA to prevent hang)...")
    vec_env = VecEnv(NUM_ENVS, rows=stage.rows, cols=stage.cols,
                     n_ghosts=stage.n_ghosts, n_power=stage.n_power,
                     static_pacman=True)

    print("Initializing networks...")
    actor  = GhostActor().to(DEVICE)
    critic = GhostCritic().to(DEVICE)
    opt_actor  = torch.optim.Adam(actor.parameters(), lr=LR)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=LR*2)

    start_update = 1
    episodes     = 0
    total_steps  = 0
    if "--resume" in sys.argv:
        # find the latest checkpoint automatically
        ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "ckpt_*.pt")),
                       key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[1]))
        if ckpts:
            ckpt_path = ckpts[-1]
            print(f"Resuming from {ckpt_path} ...")
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            actor.load_state_dict(ckpt["actor"])
            critic.load_state_dict(ckpt["critic"])
            opt_actor.load_state_dict(ckpt["opt_actor"])
            opt_critic.load_state_dict(ckpt["opt_critic"])
            curriculum.load_state_dict(ckpt["curriculum"])
            start_update = ckpt["update"] + 1
            episodes     = ckpt.get("episodes", 0)
            total_steps  = ckpt.get("total_steps", ckpt["update"] * ROLLOUT_STEPS * NUM_ENVS)
            # restore curriculum stage in VecEnv
            stage = curriculum.stage
            vec_env.set_curriculum(stage.rows, stage.cols,
                                   stage.n_ghosts, stage.n_power,
                                   static_pacman=(start_update <= 50))
            print(f"Resumed at update {start_update}, "
                  f"stage {curriculum.stage_idx} "
                  f"({stage.rows}×{stage.cols}, {stage.n_ghosts}g)")
        else:
            print("No checkpoints found — starting from scratch.")


    print("VecEnv initialized. Starting training...")
    t0 = time.time()
    current_returns = [0.0] * NUM_ENVS
    
    for update in range(start_update, 50_001):
        if update == 51:
            print("Transitioning to moving Pacman (static_pacman = False)...")
            vec_env.set_curriculum(stage.rows, stage.cols,
                                   stage.n_ghosts, stage.n_power,
                                   static_pacman=False)
            # update vec_env's current_obs from the curriculum reset
        # ── collect rollout ──────────────────────────────────────────
        # Per-env, per-step storage  (lists of length ROLLOUT_STEPS)
        buf_spatial   = [[] for _ in range(NUM_ENVS)]
        buf_gsp       = [[] for _ in range(NUM_ENVS)]
        buf_vector    = [[] for _ in range(NUM_ENVS)]
        buf_mask      = [[] for _ in range(NUM_ENVS)]
        buf_htarget   = [[] for _ in range(NUM_ENVS)]
        buf_actions   = [[] for _ in range(NUM_ENVS)]
        buf_logprobs  = [[] for _ in range(NUM_ENVS)]
        buf_values    = [[] for _ in range(NUM_ENVS)]
        buf_rewards   = [[] for _ in range(NUM_ENVS)]
        buf_dones     = [[] for _ in range(NUM_ENVS)]
        buf_gids      = [[] for _ in range(NUM_ENVS)]
        ep_returns       = []
        ep_pacman_scores = []

        for _ in range(ROLLOUT_STEPS):
            step_actions = [{}] * NUM_ENVS

            # ── BUG-5 fix: batch forward pass across all envs ────────
            # Collect all alive ghosts across all environments
            batch_sp, batch_ve, batch_vm, batch_gsp = [], [], [], []
            batch_env_idx = []   # which env each ghost belongs to
            batch_gids = []      # ghost id within its env
            env_n_ghosts = []    # how many ghosts per env (for splitting)

            for e in range(NUM_ENVS):
                obs = vec_env.current_obs[e]
                gids, sp, ve, vm, ht, global_sp, grid_shape = obs
                n_g = len(gids)
                env_n_ghosts.append(n_g)

                if n_g == 0:
                    buf_values[e].append({})
                    buf_gids[e].append([])
                    buf_spatial[e].append(np.empty((0, SPATIAL_CH, MAX_H, MAX_W), dtype=np.float32))
                    buf_gsp[e].append(np.empty((0, 5, MAX_H, MAX_W), dtype=np.float32))
                    buf_vector[e].append(np.empty((0, 101), dtype=np.float32))
                    buf_mask[e].append(np.empty((0, MAX_H, MAX_W), dtype=bool))
                    buf_htarget[e].append(np.empty((0, MAX_H, MAX_W), dtype=np.float32))
                    buf_actions[e].append(np.empty((0, K_NOMINATIONS), dtype=np.int64))
                    buf_logprobs[e].append(np.empty((0,), dtype=np.float32))
                    continue

                # Pad trimmed observations to current stage size for CNN
                sp_padded = _pad_spatial(sp, target_h=stage.rows, target_w=stage.cols)
                vm_padded = _pad_spatial(vm, target_h=stage.rows, target_w=stage.cols)
                ht_padded = _pad_spatial(ht, target_h=stage.rows, target_w=stage.cols)
                gsp_padded = _pad_spatial(global_sp, target_h=stage.rows, target_w=stage.cols)
                gsp_padded_batch = np.repeat(gsp_padded[np.newaxis, ...], n_g, axis=0)

                batch_sp.append(sp_padded)
                batch_gsp.append(gsp_padded_batch)
                batch_ve.append(ve)
                batch_vm.append(vm_padded.astype(bool))
                batch_env_idx.extend([e] * n_g)
                batch_gids.extend(gids)

                # Store padded obs for PPO buffer
                buf_spatial[e].append(sp_padded)
                buf_gsp[e].append(gsp_padded_batch)
                buf_vector[e].append(ve)
                buf_mask[e].append(vm_padded.astype(bool))
                buf_htarget[e].append(ht_padded)
                buf_gids[e].append(gids)

            # Run a single batched forward pass for ALL ghosts across ALL envs
            if batch_sp:
                all_sp = np.concatenate(batch_sp, axis=0)
                all_gsp = np.concatenate(batch_gsp, axis=0)
                all_ve = np.concatenate(batch_ve, axis=0)
                all_vm = np.concatenate(batch_vm, axis=0)

                t_sp = torch.tensor(all_sp, device=DEVICE, dtype=torch.float32)
                t_gsp = torch.tensor(all_gsp, device=DEVICE, dtype=torch.float32)
                t_ve = torch.tensor(all_ve, device=DEVICE, dtype=torch.float32)
                t_vm = torch.tensor(all_vm, device=DEVICE, dtype=torch.bool)

                with torch.no_grad():
                    idx, lp, scores, pool, vec = actor(
                        t_sp, t_ve, t_vm, K=K_NOMINATIONS)
                    val = _critic_value(critic, t_gsp, t_ve, len(batch_gids))

                idx_np = idx.cpu().numpy()
                sc_np  = scores.cpu().numpy()
                lp_np  = lp.cpu().numpy()
                val_np = val.cpu().numpy()

                # Split results back per-environment
                offset = 0
                for e in range(NUM_ENVS):
                    n_g = env_n_ghosts[e]
                    if n_g == 0:
                        continue

                    obs = vec_env.current_obs[e]
                    gids = obs[0]

                    e_idx = idx_np[offset:offset + n_g]
                    e_sc  = sc_np[offset:offset + n_g]
                    e_lp  = lp_np[offset:offset + n_g]
                    e_val = val_np[offset:offset + n_g]

                    env_act = {}
                    for i, gid in enumerate(gids):
                        pairs = [(int(x // stage.cols), int(x % stage.cols))
                                 for x in e_idx[i]]
                        env_act[gid] = (pairs, e_sc[i])

                    step_actions[e] = env_act

                    buf_actions[e].append(e_idx)
                    buf_logprobs[e].append(e_lp.sum(axis=1))
                    v_dict = {gids[i]: float(e_val[i]) for i in range(n_g)}
                    buf_values[e].append(v_dict)

                    offset += n_g

            obs_list, rew_list, done_list, info_list = vec_env.step(
                step_actions)

            for e in range(NUM_ENVS):
                r = rew_list[e]
                mean_r = sum(r.values()) / max(1, len(r)) if r else 0.0
                current_returns[e] += mean_r
                
                buf_rewards[e].append(r)
                buf_dones[e].append(1.0 if done_list[e] else 0.0)
                if done_list[e]:
                    episodes += 1
                    ep_returns.append(current_returns[e])
                    ep_pacman_scores.append(info_list[e].get("pacman_score", 0))
                    current_returns[e] = 0.0

        total_steps += ROLLOUT_STEPS * NUM_ENVS

        # ── batched bootstrap values across ALL envs ──────────────────
        boot_sp, boot_gsp, boot_ve, boot_vm = [], [], [], []
        boot_env_idx, boot_gids_list, boot_n_ghosts = [], [], []
        for e in range(NUM_ENVS):
            if len(buf_rewards[e]) == 0:
                boot_n_ghosts.append(0)
                continue
            obs = vec_env.current_obs[e]
            gids, sp, ve, vm, ht, global_sp, grid_shape = obs
            boot_n_ghosts.append(len(gids))
            if len(gids) > 0:
                sp_padded = _pad_spatial(sp, target_h=stage.rows, target_w=stage.cols)
                vm_padded = _pad_spatial(vm, target_h=stage.rows, target_w=stage.cols).astype(bool)
                gsp_padded = _pad_spatial(global_sp, target_h=stage.rows, target_w=stage.cols)
                gsp_padded_batch = np.repeat(gsp_padded[np.newaxis, ...], len(gids), axis=0)
                boot_sp.append(sp_padded)
                boot_gsp.append(gsp_padded_batch)
                boot_ve.append(ve)
                boot_vm.append(vm_padded)
                boot_gids_list.append(gids)

        # Single batched forward pass for all bootstrap values
        all_last_v = [{}] * NUM_ENVS
        if boot_sp:
            cat_sp = np.concatenate(boot_sp, axis=0)
            cat_gsp = np.concatenate(boot_gsp, axis=0)
            cat_ve = np.concatenate(boot_ve, axis=0)
            cat_vm = np.concatenate(boot_vm, axis=0)
            t_sp = torch.tensor(cat_sp, device=DEVICE, dtype=torch.float32)
            t_gsp = torch.tensor(cat_gsp, device=DEVICE, dtype=torch.float32)
            t_ve = torch.tensor(cat_ve, device=DEVICE, dtype=torch.float32)
            t_vm = torch.tensor(cat_vm, device=DEVICE, dtype=torch.bool)
            with torch.no_grad():
                _, _, _, pool, vec = actor(t_sp, t_ve, t_vm, K=K_NOMINATIONS)
                val = _critic_value(critic, t_gsp, t_ve, len(cat_sp))
            v_np = val.cpu().numpy()
            offset = 0
            gids_iter = iter(boot_gids_list)
            for e in range(NUM_ENVS):
                n_g = boot_n_ghosts[e]
                if n_g == 0:
                    continue
                gids = next(gids_iter)
                all_last_v[e] = {gids[i]: float(v_np[offset + i]) for i in range(n_g)}
                offset += n_g

        # ── flatten per-env rollouts into one big batch ──────────────
        all_sp, all_gsp, all_ve, all_vm, all_ht = [], [], [], [], []
        all_act, all_lp, all_adv, all_ret = [], [], [], []

        for e in range(NUM_ENVS):
            T = len(buf_rewards[e])
            if T == 0:
                continue

            last_v = all_last_v[e]
            adv_dict_list, ret_dict_list = compute_gae(buf_rewards[e], buf_values[e], buf_dones[e], last_v, GAMMA, GAE_LAMBDA)

            # expand per-step per-ghost
            for t in range(T):
                if len(buf_spatial[e]) <= t:
                    continue
                sp_t  = buf_spatial[e][t]     # (N, C, H, W)
                gsp_t = buf_gsp[e][t]
                gids_t = buf_gids[e][t]
                n_g   = len(gids_t)
                for i in range(n_g):
                    gid = gids_t[i]
                    all_sp.append(sp_t[i])
                    all_gsp.append(gsp_t[i])
                    all_ve.append(buf_vector[e][t][i])
                    all_vm.append(buf_mask[e][t][i])
                    all_ht.append(buf_htarget[e][t][i])
                    all_act.append(buf_actions[e][t][i])
                    all_lp.append(buf_logprobs[e][t][i])
                    all_adv.append(adv_dict_list[t].get(gid, 0.0))
                    all_ret.append(ret_dict_list[t].get(gid, 0.0))

        if not all_sp:
            continue

        # build dataset
        ds_sp  = torch.tensor(np.array(all_sp),  device=DEVICE, dtype=torch.float32)
        ds_gsp = torch.tensor(np.array(all_gsp), device=DEVICE, dtype=torch.float32)
        ds_ve  = torch.tensor(np.array(all_ve),  device=DEVICE, dtype=torch.float32)
        ds_vm  = torch.tensor(np.array(all_vm),  device=DEVICE, dtype=torch.bool)
        ds_ht  = torch.tensor(np.array(all_ht),  device=DEVICE, dtype=torch.float32)
        ds_act = torch.tensor(np.array(all_act), device=DEVICE, dtype=torch.long)
        ds_olp = torch.tensor(np.array(all_lp),  device=DEVICE, dtype=torch.float32)
        ds_adv = torch.tensor(np.array(all_adv), device=DEVICE, dtype=torch.float32)
        ds_ret = torch.tensor(np.array(all_ret), device=DEVICE, dtype=torch.float32)

        N_total = ds_sp.shape[0]
        indices = np.arange(N_total)

        # Normalize advantages GLOBALLY across the entire batch, not per-minibatch
        ds_adv = (ds_adv - ds_adv.mean()) / (ds_adv.std() + 1e-8)

        # Normalize returns for critic stability (prevents value loss explosion)
        ret_mean = ds_ret.mean()
        ret_std  = ds_ret.std() + 1e-8
        ds_ret   = (ds_ret - ret_mean) / ret_std

        # BC coefficient: starts at BC_INIT, decays sharply after update 50
        decay_step = max(0, update - 50)
        lam_bc = max(BC_FLOOR, BC_INIT * math.exp(-decay_step / 200.0))

        # ── PPO epochs ───────────────────────────────────────────────
        metrics = {"actor_loss": 0, "value_loss": 0, "bc_loss": 0,
                   "entropy": 0, "n_batches": 0}

        for epoch_i in range(PPO_EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, N_total, MINI_BATCH):
                end = min(start + MINI_BATCH, N_total)
                idx = indices[start:end]
                b_sp  = ds_sp[idx]
                b_gsp = ds_gsp[idx]
                b_ve  = ds_ve[idx]
                b_vm  = ds_vm[idx]
                b_ht  = ds_ht[idx]
                b_act = ds_act[idx]
                b_olp = ds_olp[idx]
                b_adv = ds_adv[idx]
                b_ret = ds_ret[idx]

                # single forward pass — returns logprobs, entropy, tokens for critic, and logits for BC
                new_lp, ent, pool, vec, flat_logits = actor.evaluate_actions(
                    b_sp, b_ve, b_vm, b_act)

                # value (per-ghost, IPPO CNN critic)
                v_pred = critic(b_gsp, b_ve).squeeze(-1)

                # PPO clipped surrogate
                ratio = torch.exp(new_lp - b_olp)
                s1 = ratio * b_adv
                s2 = ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * b_adv
                a_loss = -torch.min(s1, s2).mean()

                # value loss
                v_loss = F.mse_loss(v_pred, b_ret)

                # BC loss: sparse cross-entropy at nominated indices only
                # Equivalent to KL but avoids 1350 zero-contribution terms
                b_ht_masked = b_ht * b_vm.float()
                ht_flat     = b_ht_masked.view(b_ht_masked.shape[0], -1)
                ht_row_sums = ht_flat.sum(dim=1)
                valid_bc    = ht_row_sums > 1e-6
                if valid_bc.any():
                    ht_valid  = ht_flat[valid_bc]
                    ht_prob   = (ht_valid / ht_valid.sum(dim=1, keepdim=True)).detach()
                    fl_bc     = flat_logits[valid_bc].clamp(min=-1e4)
                    log_pi    = F.log_softmax(fl_bc, dim=-1)
                    # Sparse CE: -sum(p * log_pi) only where p > 0
                    bc        = -(ht_prob * log_pi).sum(dim=-1).mean()
                else:
                    bc = torch.tensor(0.0, device=DEVICE)

                loss = a_loss + VF_COEF * v_loss - ENT_COEF * ent.mean() + lam_bc * bc

                opt_actor.zero_grad()
                opt_critic.zero_grad()
                loss.backward()
                # Separate gradient clipping: prevents critic from eating actor's budget
                grad_norm_a = nn.utils.clip_grad_norm_(actor.parameters(), MAX_GRAD_NORM)
                grad_norm_c = nn.utils.clip_grad_norm_(critic.parameters(), MAX_GRAD_NORM)
                if torch.isfinite(grad_norm_a) and torch.isfinite(grad_norm_c):
                    opt_actor.step()
                    opt_critic.step()
                else:
                    opt_actor.zero_grad()
                    opt_critic.zero_grad()

                metrics["actor_loss"] += a_loss.item()
                metrics["value_loss"] += v_loss.item()
                metrics["bc_loss"]    += bc.item()
                metrics["entropy"]    += ent.mean().item()
                metrics["n_batches"]  += 1

            # Note: torch.cuda.empty_cache() removed — counterproductive
            # with expandable_segments:True and costs ~1% per-update

        nb = max(1, metrics["n_batches"])
        mean_ret = round(float(np.mean(ep_returns)), 3) if ep_returns else None
        row = {
            "update":     update,
            "episodes":   episodes,
            "steps":      total_steps,
            "wall_s":     round(time.time() - t0, 1),
            "actor_loss": round(metrics["actor_loss"] / nb, 5),
            "value_loss": round(metrics["value_loss"] / nb, 5),
            "bc_loss":    round(metrics["bc_loss"] / nb, 5),
            "entropy":    round(metrics["entropy"] / nb, 5),
            "bc_coef":    round(lam_bc, 4),
            "mean_return": mean_ret,
            "pacman_score": round(float(np.mean(ep_pacman_scores)), 1) if ep_pacman_scores else None,
            "curriculum_stage": curriculum.stage_idx,
            "grid_size": f"{curriculum.stage.rows}x{curriculum.stage.cols}",
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")

        # ── curriculum advancement ───────────────────────────────────
        if lam_bc <= 0.05:
            curriculum.record_return(mean_ret)
        else:
            curriculum.record_return(None)
            curriculum._return_history.clear()

        if curriculum.should_advance():
            curriculum.advance()
            stage = curriculum.stage
            print(f"\n{'='*60}")
            print(f"CURRICULUM ADVANCE → Stage {curriculum.stage_idx} "
                  f"({stage.rows}×{stage.cols}, {stage.n_ghosts} ghosts)")
            print(f"{'='*60}\n")
            is_static_pacman = (update <= 50)
            vec_env.set_curriculum(stage.rows, stage.cols,
                                  stage.n_ghosts, stage.n_power,
                                  static_pacman=is_static_pacman)
            current_returns = [0.0] * NUM_ENVS
            # Halve LR on stage transition to prevent catastrophic forgetting
            for pg in opt_actor.param_groups:
                pg['lr'] *= 0.5
            for pg in opt_critic.param_groups:
                pg['lr'] *= 0.5
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "curriculum_advance": curriculum.stage_idx,
                    "update": update,
                    "new_grid": f"{stage.rows}x{stage.cols}",
                    "new_lr": opt_actor.param_groups[0]['lr'],
                }) + "\n")

        # ── terminal status ──────────────────────────────────────────
        if update % 10 == 0:
            elapsed = time.time() - t0
            mins, secs = divmod(int(elapsed), 60)
            hrs, mins = divmod(mins, 60)
            runtime = f"{hrs}h {mins:02d}m {secs:02d}s" if hrs else f"{mins}m {secs:02d}s"

            if lam_bc > 0.25:
                phase = "\033[95mHybrid RL + IL\033[0m"         # magenta
            elif lam_bc > 0.05:
                phase = "\033[93mIL → RL Transition\033[0m"    # yellow
            else:
                phase = "\033[92mReinforcement Learning\033[0m" # green

            stg = curriculum.stage
            cur_lr = opt_actor.param_groups[0]['lr']

            print(f"\n┌─── Update {update:>5} / 50k ── {runtime} ─────────────────────────────────")
            print(f"│  Phase: {phase}   Curriculum: Stage {curriculum.stage_idx} ({stg.rows}×{stg.cols}, {stg.n_ghosts}g)")
            print(f"│  Episodes: {episodes:<8}  Steps: {total_steps:<10}  LR: {cur_lr:.2e}")
            print(f"│  BC Coef:   {lam_bc:.4f}    Policy Loss: {row['actor_loss']:>+.5f}")
            print(f"│  Value Loss: {row['value_loss']:.5f}   BC Loss: {row['bc_loss']:.5f}   Entropy: {row['entropy']:.5f}")
            ret_str = f"{row['mean_return']:.3f}" if row['mean_return'] is not None else "—"
            pac_str = f"{row['pacman_score']:.1f}" if row['pacman_score'] is not None else "—"
            print(f"│  Ghost Return: {ret_str:<10}  Pacman Score: {pac_str}")
            print(f"└{'─'*64}")

        if update % 200 == 0:
            path = os.path.join(CKPT_DIR, f"ckpt_{update}.pt")
            torch.save({"actor": actor.state_dict(),
                         "critic": critic.state_dict(),
                         "opt_actor": opt_actor.state_dict(),
                         "opt_critic": opt_critic.state_dict(),
                         "update": update,
                         "episodes": episodes,
                         "total_steps": total_steps,
                         "curriculum": curriculum.state_dict()}, path)
            with open(log_path, "a") as f:
                f.write(json.dumps({"checkpoint": path, "update": update}) + "\n")
            print(f"  💾 Checkpoint saved: {path}")

    vec_env.close()


if __name__ == "__main__":
    train()
