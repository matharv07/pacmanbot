import glob
import math
import sys
import time
import json
import os
import multiprocessing as mp
from multiprocessing.connection import wait as mp_wait
if mp.current_process().name == 'MainProcess':
    try:
        import setup_dependencies
        setup_dependencies.main()
    except Exception as e:
        print(f"Failed to check dependencies: {e}")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from net    import GhostActor, GhostCritic
from worker import Env
from obs    import MAX_H, MAX_W, MAX_GHOSTS, SPATIAL_CH, VEC_DIM, CRITIC_VEC_DIM
from curriculum import CurriculumScheduler, STAGES

os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')

NUM_ENVS        = 10
ROLLOUT_STEPS   = 256
MINI_BATCH      = 2048  # Increased to saturate GPU throughput
PPO_EPOCHS      = 12    # Kept at 12 to maximize learning per rollout
GAMMA           = 0.99
GAE_LAMBDA      = 0.95
CLIP_EPS        = 0.2
ENT_COEF        = 0.002
VF_COEF         = 0.5
MAX_GRAD_NORM   = 0.5
LR              = 3e-4
BC_INIT         = 0.5     # user requested reduction to prevent flattening actor loss
BC_FLOOR        = 0.002
K_NOMINATIONS   = 3
LOG_DIR         = os.path.join(os.path.dirname(__file__), "logs")
CKPT_DIR        = os.path.join(os.path.dirname(__file__), "checkpoints")
BC_ANNEAL_UPDATES = 150
BC_ADVANCE_GATE = 0.10
CURRICULUM_START_STAGE = 0
critic_warmup_remaining = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BatchTransfer:
    def __init__(self, device):
        self.host_buf = None
        self.dev_buf = None
        self.device = device

    def transfer(self, *numpy_arrays):
        sizes = [arr.nbytes for arr in numpy_arrays]
        padded_sizes = [(s + 7) & ~7 for s in sizes]
        total_bytes = sum(padded_sizes)
        if self.host_buf is None or self.host_buf.numel() < total_bytes:
            new_size = max(total_bytes, self.host_buf.numel() * 2 if self.host_buf is not None else total_bytes)
            self.host_buf = torch.empty(new_size, dtype=torch.uint8, pin_memory=True)
            self.dev_buf = torch.empty(new_size, dtype=torch.uint8, device=self.device)
        offset = 0
        for arr, p_size in zip(numpy_arrays, padded_sizes):
            arr_uint8 = np.frombuffer(arr.data, dtype=np.uint8)
            t_src = torch.from_numpy(arr_uint8)
            self.host_buf[offset:offset + arr.nbytes].copy_(t_src)
            offset += p_size
        self.dev_buf[:total_bytes].copy_(self.host_buf[:total_bytes], non_blocking=True)
        res = []
        offset = 0
        for arr, p_size in zip(numpy_arrays, padded_sizes):
            dtype_mapping = {'float32': torch.float32, 'float64': torch.float64, 'int64': torch.int64, 'int32': torch.int32, 'bool': torch.bool, 'uint8': torch.uint8}
            pt_dtype = dtype_mapping[str(arr.dtype)]
            byte_slice = self.dev_buf[offset:offset + arr.nbytes]
            t_arr = byte_slice.view(pt_dtype).reshape(arr.shape)
            res.append(t_arr)
            offset += p_size
        return res

def _pad_spatial(arr, target_h=MAX_H, target_w=MAX_W):
    #Pad (*, H, W) numpy array to (*, target_h, target_w)
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
        if out.shape[1] in (SPATIAL_CH, 11):
            out[:, 0, h:, :] = 1.0
            out[:, 0, :, w:] = 1.0
        return out
    return arr

def _worker(env_id, conn, rows, cols, n_ghosts, n_power, static_pacman=False):
    try:
        env = Env(env_id, num_ghosts=n_ghosts, grid_rows=rows, grid_cols=cols, n_power=n_power)
        env.static_pacman = static_pacman
        obs = env.reset()
        conn.send(obs)           #send initial observation
    except Exception as e:
        import traceback
        traceback.print_exc()
        conn.send(e)
        return
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            if isinstance(data, tuple) and len(data) == 2:
                a, bc_prob = data
                result = env.step(a, bc_prob)
            else:
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

def _recv_unordered(conns, procs=None):
    n = len(conns)
    results = [None] * n
    conn_to_idx = {id(c): i for i, c in enumerate(conns)}
    pending = set(conns)
    while pending:
        ready = mp_wait(pending, timeout=1.0)
        if not ready:
            if procs:
                dead = [p.pid for p in procs if not p.is_alive()]
                if dead:
                    raise RuntimeError(f"Worker processes {dead} died unexpectedly (OOM/Segfault)")
            continue
        for conn in ready:
            i = conn_to_idx[id(conn)]
            try:
                results[i] = conn.recv()
            except EOFError:
                if procs:
                    dead = [p.pid for p in procs if not p.is_alive()]
                    if dead:
                        raise RuntimeError(f"Worker processes {dead} died unexpectedly (EOFError)")
                raise RuntimeError("EOFError on multiprocessing pipe")
            pending.remove(conn)
    return results

class VecEnv:
    def __init__(self, n, rows=33, cols=41, n_ghosts=7, n_power=28, static_pacman=False):
        self.n = n
        ctx = mp.get_context("spawn")
        self.parent, self.child = zip(*[ctx.Pipe() for _ in range(n)])
        self.procs = []
        for i, c in enumerate(self.child):
            p = ctx.Process(target=_worker, args=(i, c, rows, cols, n_ghosts, n_power, static_pacman), daemon=True)
            p.start()
            self.procs.append(p)
        self.current_obs = _recv_unordered(self.parent, procs=self.procs)

    def reset(self):
        for p in self.parent:
            p.send(("reset", None))
        self.current_obs = _recv_unordered(self.parent, procs=self.procs)
        return self.current_obs

    def set_curriculum(self, rows, cols, n_ghosts, n_power, static_pacman=False):
        for p in self.parent:
            p.send(("set_curriculum", (rows, cols, n_ghosts, n_power, static_pacman)))
        self.current_obs = _recv_unordered(self.parent, procs=self.procs)
        return self.current_obs

    def step(self, actions: list[dict], bc_prob: float = 0.0):
        if not actions:
            print("vec_env.step() with empty actions")
            for p in self.parent:
                p.send(("step", ({}, bc_prob)))
            results = _recv_unordered(self.parent, procs=self.procs)
            print("vec_env.step() returned")
        else:
            for p, a in zip(self.parent, actions):
                p.send(("step", (a, bc_prob)))
            results = _recv_unordered(self.parent, procs=self.procs)
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


def _critic_value(critic, spatial_unique, vector, env_n_ghosts):
    if sum(env_n_ghosts) == 0:
        return torch.zeros(0, device=DEVICE)
    pool = critic.encode_spatial(spatial_unique)
    repeats = torch.tensor(env_n_ghosts, device=DEVICE, dtype=torch.long)
    pool_expanded = torch.repeat_interleave(pool, repeats, dim=0)
    return critic.forward_from_pool(pool_expanded, vector).squeeze(-1)

def train():
    os.makedirs(LOG_DIR, exist_ok=True)
    import threading
    import queue
    train_thread = None
    result_queue = queue.Queue()
    os.makedirs(CKPT_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, "metrics.jsonl")
    curriculum = CurriculumScheduler(start_stage=CURRICULUM_START_STAGE)
    stage = curriculum.stage
    print(f"Curriculum: starting at Stage {curriculum.stage_idx}\n({stage.rows}×{stage.cols}, {stage.n_ghosts} ghosts)")
    print("Initializing VecEnv (spawn before CUDA to prevent hang)...")
    vec_env = VecEnv(NUM_ENVS, rows=stage.rows, cols=stage.cols, n_ghosts=stage.n_ghosts, n_power=stage.n_power, static_pacman=(CURRICULUM_START_STAGE == 0))
    print("Initializing networks...")
    actor  = GhostActor().to(DEVICE)
    critic = GhostCritic().to(DEVICE)
    actor_rollout = GhostActor().to(DEVICE)
    critic_rollout = GhostCritic().to(DEVICE)
    actor_rollout.load_state_dict(actor.state_dict())
    critic_rollout.load_state_dict(critic.state_dict())
    actor_rollout.eval()
    critic_rollout.eval()
    opt_actor  = torch.optim.Adam(actor.parameters(), lr=LR)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=LR*2)
    start_update = 1
    episodes     = 0
    total_steps  = 0
    if "--resume" in sys.argv:
        ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "ckpt_*.pt")), key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[1]))
        if ckpts:
            ckpt_path = ckpts[-1]
            print(f"Resuming from {ckpt_path} ...")
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            actor.load_state_dict(ckpt["actor"])
            try:
                critic_sd = ckpt["critic"]
                new_sd = critic.state_dict()
                filtered_sd = {k: v for k, v in critic_sd.items() if k in new_sd and v.shape == new_sd[k].shape}
                new_sd.update(filtered_sd)
                critic.load_state_dict(new_sd)
                if len(filtered_sd) == len(critic_sd) and len(filtered_sd) == len(new_sd):
                    try:
                        opt_critic.load_state_dict(ckpt["opt_critic"])
                    except Exception as e:
                        print(f"Warning: Could not restore critic optimizer: {e}")
            except Exception as e:
                print(f"Warning: Could not fully restore critic weights: {e}")
            opt_actor.load_state_dict(ckpt["opt_actor"])
            curriculum.load_state_dict(ckpt["curriculum"])
            start_update = ckpt["update"] + 1
            episodes     = ckpt.get("episodes", 0)
            total_steps  = ckpt.get("total_steps", ckpt["update"] * ROLLOUT_STEPS * NUM_ENVS)
            #restore curriculum stage in VecEnv
            stage = curriculum.stage
            vec_env.set_curriculum(stage.rows, stage.cols,stage.n_ghosts, stage.n_power, static_pacman=(start_update <= 50 and CURRICULUM_START_STAGE == 0))
            print(f"Resumed at update {start_update}, stage {curriculum.stage_idx} ({stage.rows}×{stage.cols}, {stage.n_ghosts}g)")
        else:
            print("No checkpoints found, starting from scratch.")
    print("VecEnv initialized. Starting training...")
    t0 = time.time()

    def ppo_worker(update, b_sp, b_gsp_unique, b_gsp_ids, b_ve, b_cve, b_vm, b_ht, b_act, b_olp, b_adv, b_ret, lam_bc, mean_ret, mean_pac, episodes, total_steps, t_rollout, t0_ref, bc_prob, realized_merge_rate):
        t_ppo_start = time.time()
        metrics = {"actor_loss": 0, "value_loss": 0, "bc_loss": 0, "entropy": 0, "n_batches": 0}    
        N_total = b_sp.shape[0]
        from collections import defaultdict
        uid_to_indices = defaultdict(list)
        b_gsp_ids_np = b_gsp_ids.cpu().numpy()
        for i in range(N_total):
            uid_to_indices[b_gsp_ids_np[i]].append(i)
        unique_uids = list(uid_to_indices.keys())
        global critic_warmup_remaining
        for epoch_i in range(PPO_EPOCHS):
            np.random.shuffle(unique_uids)
            uid_batches = []
            cur_batch = []
            cur_count = 0
            for uid in unique_uids:
                cur_batch.append(uid)
                cur_count += len(uid_to_indices[uid])
                if cur_count >= MINI_BATCH:
                    uid_batches.append(cur_batch)
                    cur_batch = []
                    cur_count = 0
            if cur_batch:
                uid_batches.append(cur_batch)
            for batch_uids in uid_batches:
                idx = []
                for uid in batch_uids:
                    idx.extend(uid_to_indices[uid])
                idx = np.array(idx)
                mb_sp  = b_sp[idx]
                mb_ve  = b_ve[idx]
                mb_cve = b_cve[idx]
                mb_gsp_ids = b_gsp_ids[idx]
                mb_vm  = b_vm[idx]
                mb_ht  = b_ht[idx]
                mb_act = b_act[idx]
                mb_olp = b_olp[idx]
                mb_adv = b_adv[idx]
                mb_ret = b_ret[idx]
                new_lp, ent, pool, vec, flat_logits = actor.evaluate_actions(mb_sp, mb_ve, mb_vm, mb_act)
                unique_ids, inv_idx = torch.unique(mb_gsp_ids, return_inverse=True)
                mb_gsp_unique = b_gsp_unique[unique_ids]
                mb_c_pool = critic.encode_spatial(mb_gsp_unique)
                v_pred = critic.forward_from_pool(mb_c_pool[inv_idx], mb_cve).squeeze(-1)
                ratio = torch.exp(new_lp - mb_olp)
                s1 = ratio * mb_adv
                s2 = ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * mb_adv
                a_loss = -torch.min(s1, s2).mean()
                v_loss = F.smooth_l1_loss(v_pred, mb_ret)
                mb_ht_masked = mb_ht * mb_vm.float()
                ht_flat     = mb_ht_masked.view(mb_ht_masked.shape[0], -1)
                ht_row_sums = ht_flat.sum(dim=1)
                valid_bc    = ht_row_sums > 1e-6
                if valid_bc.any():
                    ht_valid  = ht_flat[valid_bc]
                    ht_prob   = (ht_valid / ht_valid.sum(dim=1, keepdim=True)).detach()
                    fl_bc     = flat_logits[valid_bc].clamp(min=-1e4)
                    log_pi    = F.log_softmax(fl_bc, dim=-1)
                    bc        = -(ht_prob * log_pi).sum(dim=-1).mean()
                    bc        = bc * valid_bc.float().mean()
                else:
                    bc = torch.tensor(0.0, device=DEVICE)
                loss_actor = a_loss - ENT_COEF * ent.mean() + lam_bc * bc
                loss_critic = VF_COEF * v_loss
                opt_critic.zero_grad()
                loss_critic.backward()
                if critic_warmup_remaining <= 0:
                    opt_actor.zero_grad()
                    loss_actor.backward()
                    grad_norm_a = nn.utils.clip_grad_norm_(actor.parameters(), MAX_GRAD_NORM)
                else:
                    grad_norm_a = torch.tensor(0.0)
                grad_norm_c = nn.utils.clip_grad_norm_(critic.parameters(), MAX_GRAD_NORM)
                if torch.isfinite(grad_norm_c) and (critic_warmup_remaining > 0 or torch.isfinite(grad_norm_a)):
                    opt_critic.step()
                    if critic_warmup_remaining <= 0:
                        opt_actor.step()
                else:
                    opt_critic.zero_grad()
                    if critic_warmup_remaining <= 0:
                        opt_actor.zero_grad()
                metrics["actor_loss"] += a_loss.item()
                metrics["value_loss"] += v_loss.item()
                metrics["bc_loss"]    += bc.item()
                metrics["entropy"]    += ent.mean().item()
                metrics["n_batches"]  += 1
        t_ppo = time.time() - t_ppo_start
        if critic_warmup_remaining > 0:
            critic_warmup_remaining -= 1
        result_queue.put({"update": update, "metrics": metrics,"mean_ret": mean_ret, "mean_pac": mean_pac,
        "episodes": episodes,"total_steps": total_steps, "t_rollout": t_rollout, "t_ppo": t_ppo, "lam_bc": lam_bc, "bc_prob": bc_prob, "realized_merge_rate": realized_merge_rate, "wall_s": round(time.time() - t0_ref, 1)})
    current_returns = [0.0] * NUM_ENVS
    batch_transfer = BatchTransfer(DEVICE)
    for update in range(start_update, 50_001):
        decay_step_curr = max(0, update - 50)
        anneal_frac = math.exp(-decay_step_curr / BC_ANNEAL_UPDATES)
        bc_prob = anneal_frac if anneal_frac >= 0.05 else 0.0
        if update == 51 and CURRICULUM_START_STAGE == 0:
            print("Transitioning to moving Pacman (static_pacman = False)...")
            vec_env.set_curriculum(stage.rows, stage.cols, stage.n_ghosts, stage.n_power, static_pacman=False)
        t_start_rollout = time.time()
        #per-env, per-step storage (lists of length ROLLOUT_STEPS)
        buf_spatial   = [[] for _ in range(NUM_ENVS)]
        buf_gsp       = [[] for _ in range(NUM_ENVS)]
        buf_gsp_ids   = [[] for _ in range(NUM_ENVS)]
        all_gsp_unique_list = []
        buf_vector    = [[] for _ in range(NUM_ENVS)]
        buf_cve       = [[] for _ in range(NUM_ENVS)]
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
        ep_heuristic_merges = 0
        ep_total_auctions = 0
        for _ in range(ROLLOUT_STEPS):
            step_actions = [{} for _ in range(NUM_ENVS)]
            #collect all alive ghosts across all environments
            batch_sp, batch_ve, batch_cve, batch_vm = [], [], [], []
            batch_gsp_unique = [] #one per active env
            batch_env_idx = []    #which env each ghost belongs to
            batch_gids = []       #ghost id within its env
            env_n_ghosts = []     #how many ghosts per env (for splitting)
            active_n_ghosts = []  #how many ghosts per active env
            for e in range(NUM_ENVS):
                obs = vec_env.current_obs[e]
                gids, sp, ve, vm, ht, global_sp, grid_shape = obs
                n_g = len(gids)
                env_n_ghosts.append(n_g)
                if n_g == 0:
                    buf_values[e].append({})
                    buf_gids[e].append([])
                    buf_spatial[e].append(np.empty((0, SPATIAL_CH, MAX_H, MAX_W), dtype=np.float32))
                    buf_gsp[e].append(np.empty((0, 11, MAX_H, MAX_W), dtype=np.float32))
                    buf_gsp_ids[e].append([])
                    buf_vector[e].append(np.empty((0, VEC_DIM), dtype=np.float32))
                    buf_cve[e].append(np.empty((0, CRITIC_VEC_DIM), dtype=np.float32))
                    buf_mask[e].append(np.empty((0, MAX_H, MAX_W), dtype=bool))
                    buf_htarget[e].append(np.empty((0, MAX_H, MAX_W), dtype=np.float32))
                    buf_actions[e].append(np.empty((0, K_NOMINATIONS), dtype=np.int64))
                    buf_logprobs[e].append(np.empty((0,), dtype=np.float32))
                    continue
                #Pad trimmed observations to current stage size for CNN
                sp_padded = _pad_spatial(sp, target_h=stage.rows, target_w=stage.cols)
                vm_padded = _pad_spatial(vm, target_h=stage.rows, target_w=stage.cols)
                ht_padded = _pad_spatial(ht, target_h=stage.rows, target_w=stage.cols)
                gsp_padded = _pad_spatial(global_sp, target_h=stage.rows, target_w=stage.cols)
                gsp_padded_batch = np.repeat(gsp_padded[np.newaxis, ...], n_g, axis=0)
                all_gsp_unique_list.append(gsp_padded)
                uid = len(all_gsp_unique_list) - 1
                joint_ve = np.zeros((MAX_GHOSTS, VEC_DIM), dtype=np.float32)
                for i, gid in enumerate(gids):
                    joint_ve[gid] = ve[i]
                joint_ve_flat = joint_ve.flatten()
                cve_batch = []
                for gid in gids:
                    one_hot = np.zeros(MAX_GHOSTS, dtype=np.float32)
                    one_hot[gid] = 1.0
                    cve_batch.append(np.concatenate([joint_ve_flat, one_hot]))
                cve_batch = np.array(cve_batch, dtype=np.float32)
                batch_sp.append(sp_padded)
                batch_gsp_unique.append(gsp_padded[np.newaxis, ...])
                active_n_ghosts.append(n_g)
                batch_ve.append(ve)
                batch_cve.append(cve_batch)
                batch_vm.append(vm_padded.astype(bool))
                batch_env_idx.extend([e] * n_g)
                batch_gids.extend(gids)
                #stprepadded obs for PPO buffer
                buf_spatial[e].append(sp_padded)
                buf_gsp[e].append(gsp_padded_batch)
                buf_gsp_ids[e].append([uid] * n_g)
                buf_vector[e].append(ve)
                buf_cve[e].append(cve_batch)
                buf_mask[e].append(vm_padded.astype(bool))
                buf_htarget[e].append(ht_padded)
                buf_gids[e].append(gids)
            #run a single batched forward pass for all ghosts across all envs
            if batch_sp:
                all_sp = np.concatenate(batch_sp, axis=0)
                all_gsp_unique = np.concatenate(batch_gsp_unique, axis=0)
                all_ve = np.concatenate(batch_ve, axis=0)
                all_cve = np.concatenate(batch_cve, axis=0)
                all_vm = np.concatenate(batch_vm, axis=0)
                t_sp, t_gsp_unique, t_ve, t_cve, t_vm = batch_transfer.transfer(all_sp, all_gsp_unique, all_ve, all_cve, all_vm)
                with torch.inference_mode():
                    idx, lp, scores, pool, vec = actor_rollout(
                        t_sp, t_ve, t_vm, K=K_NOMINATIONS)
                    val = _critic_value(critic_rollout, t_gsp_unique, t_cve, active_n_ghosts)
                idx_np = idx.cpu().numpy()
                sc_np  = scores.cpu().numpy()
                lp_np  = lp.cpu().numpy()
                val_np = val.cpu().numpy()
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
            obs_list, rew_list, done_list, info_list = vec_env.step(step_actions, bc_prob)
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
                    ep_heuristic_merges += info_list[e].get("heuristic_merges", 0)
                    ep_total_auctions += info_list[e].get("total_auctions", 0)
                    current_returns[e] = 0.0
        total_steps += ROLLOUT_STEPS * NUM_ENVS
        boot_sp, boot_gsp_unique, boot_ve, boot_cve, boot_vm = [], [], [], [], []
        boot_env_idx, boot_gids_list, boot_n_ghosts = [], [], []
        for e in range(NUM_ENVS):
            if len(buf_rewards[e]) == 0:
                continue
            obs = vec_env.current_obs[e]
            gids, sp, ve, vm, ht, global_sp, grid_shape = obs
            if len(gids) > 0:
                boot_n_ghosts.append(len(gids))
                sp_padded = _pad_spatial(sp, target_h=stage.rows, target_w=stage.cols)
                vm_padded = _pad_spatial(vm, target_h=stage.rows, target_w=stage.cols).astype(bool)
                gsp_padded = _pad_spatial(global_sp, target_h=stage.rows, target_w=stage.cols)
                joint_ve = np.zeros((MAX_GHOSTS, VEC_DIM), dtype=np.float32)
                for i, gid in enumerate(gids):
                    joint_ve[gid] = ve[i]
                joint_ve_flat = joint_ve.flatten()
                cve_batch = []
                for gid in gids:
                    one_hot = np.zeros(MAX_GHOSTS, dtype=np.float32)
                    one_hot[gid] = 1.0
                    cve_batch.append(np.concatenate([joint_ve_flat, one_hot]))
                boot_sp.append(sp_padded)
                boot_gsp_unique.append(gsp_padded[np.newaxis, ...])
                boot_ve.append(ve)
                boot_cve.append(np.array(cve_batch, dtype=np.float32))
                boot_vm.append(vm_padded)
                boot_gids_list.append(gids)
        all_last_v = [{} for _ in range(NUM_ENVS)]
        if boot_sp:
            cat_sp = np.concatenate(boot_sp, axis=0)
            cat_gsp_unique = np.concatenate(boot_gsp_unique, axis=0)
            cat_ve = np.concatenate(boot_ve, axis=0)
            cat_cve = np.concatenate(boot_cve, axis=0)
            cat_vm = np.concatenate(boot_vm, axis=0)
            t_sp, t_gsp_unique, t_ve, t_cve, t_vm = batch_transfer.transfer(cat_sp, cat_gsp_unique, cat_ve, cat_cve, cat_vm)
            with torch.inference_mode():
                _, _, _, pool, vec = actor_rollout(t_sp, t_ve, t_vm, K=K_NOMINATIONS)
                val = _critic_value(critic_rollout, t_gsp_unique, t_cve, boot_n_ghosts)
            v_np = val.cpu().numpy()
            offset = 0
            gids_iter = iter(boot_gids_list)
            n_ghosts_iter = iter(boot_n_ghosts)
            for e in range(NUM_ENVS):
                if len(buf_rewards[e]) == 0:
                    continue
                obs = vec_env.current_obs[e]
                if len(obs[0]) == 0:
                    continue
                n_g = next(n_ghosts_iter)
                gids = next(gids_iter)
                all_last_v[e] = {gids[i]: float(v_np[offset + i]) for i in range(n_g)}
                offset += n_g
        #flatten per-env rollouts into a single batch 
        all_sp, all_gsp, all_ve, all_vm, all_ht = [], [], [], [], []
        all_cve, all_gsp_ids = [], []
        all_act, all_lp, all_adv, all_ret = [], [], [], []
        for e in range(NUM_ENVS):
            T = len(buf_rewards[e])
            if T == 0:
                continue
            last_v = all_last_v[e]
            adv_dict_list, ret_dict_list = compute_gae(buf_rewards[e], buf_values[e], buf_dones[e], last_v, GAMMA, GAE_LAMBDA)
            for t in range(T):
                if len(buf_spatial[e]) <= t:
                    continue
                sp_t  = buf_spatial[e][t]     #(N, C, H, W)
                gsp_t = buf_gsp[e][t]
                gids_t = buf_gids[e][t]
                n_g   = len(gids_t)
                for i in range(n_g):
                    gid = gids_t[i]
                    all_sp.append(sp_t[i])
                    all_gsp.append(gsp_t[i])
                    all_gsp_ids.append(buf_gsp_ids[e][t][i])
                    all_ve.append(buf_vector[e][t][i])
                    all_cve.append(buf_cve[e][t][i])
                    all_vm.append(buf_mask[e][t][i])
                    all_ht.append(buf_htarget[e][t][i])
                    all_act.append(buf_actions[e][t][i])
                    all_lp.append(buf_logprobs[e][t][i])
                    all_adv.append(adv_dict_list[t].get(gid, 0.0))
                    all_ret.append(ret_dict_list[t].get(gid, 0.0))
        if not all_sp:
            continue
        #build dataset
        arr_sp  = np.array(all_sp, dtype=np.float32)
        arr_gsp = np.array(all_gsp, dtype=np.float32)
        arr_gsp_unique = np.array(all_gsp_unique_list, dtype=np.float32)
        arr_gsp_ids = np.array(all_gsp_ids, dtype=np.int64)
        arr_ve  = np.array(all_ve, dtype=np.float32)
        arr_cve = np.array(all_cve, dtype=np.float32)
        arr_vm  = np.array(all_vm, dtype=bool)
        arr_ht  = np.array(all_ht, dtype=np.float32)
        arr_act = np.array(all_act, dtype=np.int64)
        arr_olp = np.array(all_lp, dtype=np.float32)
        arr_adv = np.array(all_adv, dtype=np.float32)
        arr_ret = np.array(all_ret, dtype=np.float32)
        ds_sp, ds_gsp, ds_gsp_unique, ds_gsp_ids, ds_ve, ds_cve, ds_vm, ds_ht, ds_act, ds_olp, ds_adv, ds_ret = batch_transfer.transfer(
            arr_sp, arr_gsp, arr_gsp_unique, arr_gsp_ids, arr_ve, arr_cve, arr_vm, arr_ht, arr_act, arr_olp, arr_adv, arr_ret)
        N_total = ds_sp.shape[0]
        indices = np.arange(N_total)
        #normalize advantages GLOBALLY across the entire batch, not per-minibatch
        ds_adv = (ds_adv - ds_adv.mean()) / (ds_adv.std() + 1e-8)
        lam_bc = max(BC_FLOOR, BC_INIT * anneal_frac)
        realized_merge_rate = (ep_heuristic_merges / ep_total_auctions) if ep_total_auctions > 0 else 0.0
        t_rollout = time.time() - t_start_rollout
        mean_ret = round(float(np.mean(ep_returns)), 3) if ep_returns else None
        mean_pac = round(float(np.mean(ep_pacman_scores)), 1) if ep_pacman_scores else None
        if train_thread is not None:
            train_thread.join()
            train_thread = None
            res = result_queue.get()
            #sync weights to rollout actors
            actor_rollout.load_state_dict(actor.state_dict())
            critic_rollout.load_state_dict(critic.state_dict())
            p_up = res["update"]
            m = res["metrics"]
            nb = max(1, m["n_batches"])
            row = {
                "update":     p_up,
                "episodes":   res["episodes"],
                "steps":      res["total_steps"],
                "wall_s":     res["wall_s"],
                "actor_loss": round(m["actor_loss"] / nb, 5),
                "value_loss": round(m["value_loss"] / nb, 5),
                "bc_loss":    round(m["bc_loss"] / nb, 5),
                "entropy":    round(m["entropy"] / nb, 5),
                "bc_coef":    round(res["lam_bc"], 4),
                "bc_prob":    round(res["bc_prob"], 4),
                "merge_rate": round(res["realized_merge_rate"], 4),
                "mean_return": res["mean_ret"],
                "pacman_score": res["mean_pac"],
                "curriculum_stage": curriculum.stage_idx,
                "grid_size": f"{curriculum.stage.rows}x{curriculum.stage.cols}",
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            curriculum.record_return(res["mean_ret"] if res["lam_bc"] <= BC_ADVANCE_GATE else None)
            if curriculum.should_advance():
                curriculum.advance()
                stage = curriculum.stage
                print(f"\n{'='*60}")
                print(f"CURRICULUM ADVANCE → Stage {curriculum.stage_idx} "
                      f"({stage.rows}×{stage.cols}, {stage.n_ghosts} ghosts)")
                print(f"{'='*60}\n")
                is_static_pacman = (p_up <= 50)
                vec_env.set_curriculum(stage.rows, stage.cols,stage.n_ghosts, stage.n_power, static_pacman=is_static_pacman)
                current_returns = [0.0] * NUM_ENVS
                for pg in opt_actor.param_groups:
                    pg['lr'] *= 0.5
                for pg in opt_critic.param_groups:
                    pg['lr'] *= 0.5
                global critic_warmup_remaining
                critic_warmup_remaining = 20
                with open(log_path, "a") as f:
                    f.write(json.dumps({"curriculum_advance": curriculum.stage_idx, "update": p_up, "new_grid": f"{stage.rows}x{stage.cols}", "new_lr": opt_actor.param_groups[0]['lr']}) + "\n")
            if p_up % 10 == 0:
                elapsed = res["wall_s"]
                mins, secs = divmod(int(elapsed), 60)
                hrs, mins = divmod(mins, 60)
                runtime = f"{hrs}h {mins:02d}m {secs:02d}s" if hrs else f"{mins}m {secs:02d}s"
                if res["lam_bc"] > 0.25:
                    phase = "\033[95mHybrid RL + IL\033[0m"
                elif res["lam_bc"] > 0.02:
                    phase = "\033[93mIL → RL Transition\033[0m"
                else:
                    phase = "\033[92mReinforcement Learning\033[0m"
                stg = curriculum.stage
                cur_lr = opt_actor.param_groups[0]['lr']
                print(f"\n┌─── Update {p_up:>5} / 50k ── {runtime} ─────────────────────────────────")
                print(f"│  Phase: {phase}   Curriculum: Stage {curriculum.stage_idx} ({stg.rows}×{stg.cols}, {stg.n_ghosts}g)")
                print(f"│  Episodes: {res['episodes']:<8}  Steps: {res['total_steps']:<10}  LR: {cur_lr:.2e}")
                print(f"│  BC Coef:   {res['lam_bc']:.4f}    Policy Loss: {row['actor_loss']:>+.5f}")
                print(f"│  BC Prob:   {res['bc_prob']:.4f} ({row['merge_rate']:.1%} merge)    Value Loss: {row['value_loss']:.5f}")
                print(f"│  BC Loss:   {row['bc_loss']:.5f}    Entropy: {row['entropy']:.5f}")
                ret_str = f"{row['mean_return']:.3f}" if row['mean_return'] is not None else "—"
                pac_str = f"{row['pacman_score']:.1f}" if row['pacman_score'] is not None else "—"
                print(f"│  Ghost Return: {ret_str:<10}  Pacman Score: {pac_str}")
                print(f"│  Timings: Rollout {res['t_rollout']:.1f}s | PPO {res['t_ppo']:.1f}s")
                print(f"└{'─'*64}")
            if p_up % 100 == 0:
                path = os.path.join(CKPT_DIR, f"ckpt_{p_up}.pt")
                torch.save({"actor": actor.state_dict(),
                             "critic": critic.state_dict(),
                             "opt_actor": opt_actor.state_dict(),
                             "opt_critic": opt_critic.state_dict(),
                             "update": p_up,
                             "episodes": res['episodes'],
                             "total_steps": res['total_steps'],
                             "curriculum": curriculum.state_dict()}, path)
                with open(log_path, "a") as f:
                    f.write(json.dumps({"checkpoint": path, "update": p_up}) + "\n")
                print(f"  💾 Checkpoint saved: {path}")
        train_thread = threading.Thread(target=ppo_worker, args=(
            update, ds_sp, ds_gsp_unique, ds_gsp_ids, ds_ve, ds_cve, ds_vm, ds_ht, ds_act, ds_olp, ds_adv, ds_ret,
            lam_bc, mean_ret, mean_pac, episodes, total_steps, t_rollout, t0, bc_prob, realized_merge_rate))
        train_thread.start()
    if train_thread is not None:
        train_thread.join()
    vec_env.close()

if __name__ == "__main__":
    train()