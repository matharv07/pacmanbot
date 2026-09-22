#!/usr/bin/env python3
"""
Behaviour probe: plays full games with a checkpoint (or the pure heuristic / random nominations)
and measures WHAT the ghosts do, not just whether they win.

  python3 tests/behaviour_probe.py --ckpt checkpoints/ckpt_1000.pt --mode rl --n 30
  python3 tests/behaviour_probe.py --mode heuristic --n 30
  python3 tests/behaviour_probe.py --mode random --n 30

Per decision step (every DECISION_INTERVAL frames) and per alive ghost it records:
  * belief-following: distance of the closest nomination / the active CBBA target to the ghost's
    own belief-map peak, and how far that peak is from the TRUE Pacman (belief quality)
  * policy shape (rl only): entropy of the masked cell distribution, effective #cells, and the
    probability mass the policy puts on the belief map's top-10 cells
  * loitering: displacement over the trailing 5 decisions, |v|, fallback-mode flag, no-task flag
  * denial: "opportunities" where the ghost knows of a power pellet within 6 cells while Pacman is
    not a threat, and whether the ghost is actually heading for it / holding a CONVERT task
  * deaths: what the ghost knew when it died (powered flag, distance, frames since power start)
"""
import os, sys, math, argparse, time, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', 'hide')
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed

from curriculum import STAGES
from allocator import TaskType, ORIGIN_HEURISTIC, ORIGIN_RL_ENDORSE, ORIGIN_RL_NOVEL
from obs import flatten_cand_cells

DECISION_INTERVAL = 6
K = 3
MAX_FRAMES = 3000
DENIAL_RADIUS = 6.0
THREAT_RADIUS = 8.0

def _pad(arr, H, W):
    h, w = arr.shape[-2], arr.shape[-1]
    if h == H and w == W: return arr
    out = np.zeros(arr.shape[:-2] + (H, W), dtype=arr.dtype); out[..., :h, :w] = arr
    if arr.ndim == 4: out[:, 0, h:, :] = 1.0; out[:, 0, :, w:] = 1.0
    elif arr.ndim == 3: out[0, h:, :] = 1.0; out[0, :, w:] = 1.0
    return out

def _bm_peak(g):
    bm = g.belief_map
    if not getattr(bm, '_initialised', False) or len(bm._b_flat) == 0: return None, 0.0
    i = int(np.argmax(bm._b_flat)); p = float(bm._b_flat[i])
    if p <= 1e-6: return None, 0.0
    r, c = bm._open_cells[i]; return (float(r), float(c)), p

def _bm_top_mass_on_cells(g, rc_list, H, W, topn=10):
    """probability mass the policy puts on the belief map's top-n cells; rc_list is a (H*W,) prob vector"""
    bm = g.belief_map
    if not getattr(bm, '_initialised', False) or len(bm._b_flat) == 0: return float('nan')
    k = min(topn, len(bm._b_flat)); top = np.argpartition(bm._b_flat, -k)[-k:]
    mass = 0.0
    for i in top:
        r, c = bm._open_cells[i]; r, c = int(r), int(c)
        if 0 <= r < H and 0 <= c < W: mass += float(rc_list[r * W + c])
    return mass

def _nominated_cells(ghost, act):
    """World-space targets this ghost actually nominated: the candidates it pointed at, plus any off-menu
    cell. Index 0 of the action is now a list of candidate SLOTS, not (row, col) pairs."""
    if not act:
        return []
    cands = getattr(ghost, '_rl_candidates', None) or []
    out = []
    for slot in act[0]:
        if 0 <= int(slot) < len(cands):
            t = cands[int(slot)]
            out.append((float(t.target_pos[0]), float(t.target_pos[1])))
    for (r, c) in act[2]:
        out.append((float(r) + 0.5, float(c) + 0.5))
    return out

def _run_game(mode, actor, env, stage, seed, log, critic=None):
    np.random.seed(seed); torch.manual_seed(seed)
    import random as _r; _r.seed(seed)
    obs = env.reset()
    if mode == 'heuristic':
        for g in env.ghosts.values():
            g.rl_mode = False; g.cbba_agent.rl_mode = False
    H, W = stage.rows, stage.cols
    pos_hist = {gid: collections.deque(maxlen=6) for gid in env.ghosts}
    eff_hist = {gid: collections.deque(maxlen=6) for gid in env.ghosts}
    power_before = len(env.world.power_pellets)
    activations = 0; powered_frames = 0; frames_known_by_any = 0
    deaths = []
    dead_seen = set()
    n_dec = 0
    prev_powered = False; power_start_frame = -1; phase_dist = {}; learned_at = {}
    while obs is not None:
        gids, sp, ve, vm, ht, hs, cf, cc, cm, cbc, gsp, grid_shape = obs
        if not gids: break
        action = {}
        probs_per_ghost = {}
        if mode in ('rl', 'floor'):
            t_sp = torch.from_numpy(_pad(sp.astype(np.float32), H, W))
            t_vm = torch.from_numpy(_pad(vm.astype(np.float32), H, W).astype(bool))
            t_ve = torch.from_numpy(ve.astype(np.float32))
            t_cf = torch.from_numpy(cf.astype(np.float32))
            t_cc = torch.from_numpy(flatten_cand_cells(cc, W).astype(np.int64))
            t_cm = torch.from_numpy(cm.astype(bool))
            with torch.inference_mode():
                (idx, lp, scores, nidx, _nlp, nsc, _p, _v,
                 speed, _slp, direction, _dlp, gate, _glp, c_clog) = actor(t_sp, t_ve, t_vm, t_cf, t_cc, t_cm, K_cand=K, K_novel=1)
                feats, pool, vec = actor.encode(t_sp, t_ve)
                raw_logits = actor.logits_from_features(feats, pool, t_vm).view(len(gids), -1)
                cand_logits, safe = actor.candidate_logits(feats, pool, vec, t_cf, t_cc, t_cm)
            use_np = np.zeros(len(gids), dtype=bool); adv_np = np.zeros(len(gids), dtype=np.float32)
            if critic is not None and mode == 'rl':
                from net import gate_for_eval
                from obs import build_cve
                gsp_p = _pad(gsp.astype(np.float32), H, W)
                use_np, adv_np = gate_for_eval(critic, gsp_p, build_cve(gids, ve), t_cf, t_cm, c_clog, idx[:, 0])
            for _u, _a in zip(use_np, adv_np):
                log['pick_used'].append(float(_u)); log['pick_adv'].append(float(_a))
            idx_np = idx.numpy(); sc = scores.float().numpy(); spd = speed.float().numpy(); dr = direction.float().numpy(); gt = gate.float().numpy()
            nidx_np = nidx.numpy(); nsc_np = nsc.float().numpy()
            cl_np = cand_logits.float().numpy(); safe_np = safe.numpy()
            for i, gid in enumerate(gids):
                novel_pairs = [(int(x // W), int(x % W)) for x in nidx_np[i]]
                picks = [int(x) for x in idx_np[i]]
                if mode == 'floor':
                    picks, novel_pairs = [], []   #ablation: heuristic floor decides the auction on its own
                action[gid] = (picks, sc[i], novel_pairs, nsc_np[i], float(spd[i].item()), float(dr[i].item()), float(gt[i].item()), bool(use_np[i]))
                #how committed the pointer head is, in units of its own maximum
                n_live = int(cm[i].sum())
                log['n_cands'].append(n_live)
                if n_live >= 2:
                    cfl = cl_np[i][safe_np[i]]
                    cp = np.exp(cfl - cfl.max()); cp /= cp.sum()
                    ce = float(-(cp[cp > 0] * np.log(cp[cp > 0])).sum())
                    log['cand_entropy'].append(ce / math.log(n_live))
                    log['cand_top'].append(float(cp.max()))
                fl = raw_logits[i].double().numpy()
                fin = fl[np.isfinite(fl)]
                log['logit_mean'].append(float(fin.mean())); log['logit_std'].append(float(fin.std()))
                #sc is now per-CANDIDATE; the saturation check belongs to the spatial (off-menu) head
                log['score_sat'].append(float(np.mean(nsc_np[i].reshape(-1)[np.isfinite(fl)] >= 0.999)))
                m = fin.max(); p = np.where(np.isfinite(fl), np.exp(fl - m), 0.0); p /= p.sum()
                probs_per_ghost[gid] = p
                ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
                log['cell_entropy'].append(ent); log['eff_cells'].append(math.exp(ent))
                log['throttle'].append(float(spd[i].item())); log['gate'].append(float(gt[i].item()))
        elif mode == 'random':
            for i, gid in enumerate(gids):
                valid = np.flatnonzero(vm[i].reshape(-1))
                pick = np.random.choice(valid, size=min(1, len(valid)), replace=False)
                novel_pairs = [(int(x // vm.shape[-1]), int(x % vm.shape[-1])) for x in pick]
                smap = np.zeros((H, W), dtype=np.float32)
                for (r, c) in novel_pairs: smap[r, c] = np.random.uniform(0.3, 1.0)
                live = np.flatnonzero(cm[i])
                picks = list(np.random.choice(live, size=min(K, len(live)), replace=False)) if len(live) else []
                cs = np.zeros(cm.shape[1], dtype=np.float32)
                for sl in picks: cs[sl] = np.random.uniform(0.3, 1.0)
                action[gid] = ([int(x) for x in picks], cs, novel_pairs, smap, 1.0, 0.5, 0.0)
        #--- per-ghost behaviour snapshot for THIS decision (state the action was chosen in) ---
        n_dec += 1
        pac = env.player; true_pac = (pac.y, pac.x)
        any_known = False
        for gid in gids:
            g = env.ghosts[gid]
            pos_hist[gid].append((g.y, g.x))
            d_true = math.hypot(g.y - true_pac[0], g.x - true_pac[1])
            knows = g.known_pacman is not None
            any_known |= knows
            task = g.cbba_agent.get_active_task()
            speed_mag = math.hypot(g.vx, g.vy)
            log['speed'].append(speed_mag / max(1e-6, g.max_speed))
            log['fallback'].append(1.0 if g.in_fallback_mode else 0.0)
            log['no_task'].append(1.0 if task is None else 0.0)
            log['task_type'][int(task.task_type) if task else -1] += 1
            #ATTRIBUTION: who actually won this ghost's executed task, and does it close on Pacman?
            org = int(getattr(task, 'origin', ORIGIN_HEURISTIC)) if task is not None else -1
            log['origin'][org] += 1
            if task is not None:
                log['origin_score'][org].append(float(task.score))
            eff_hist[gid].append((org, d_true))
            if len(eff_hist[gid]) == eff_hist[gid].maxlen:
                old_org, old_d = eff_hist[gid][0]
                if old_org >= 0:
                    log['closing'][old_org].append(d_true - old_d)
            if len(pos_hist[gid]) == pos_hist[gid].maxlen:
                (y0, x0) = pos_hist[gid][0]; disp = math.hypot(g.y - y0, g.x - x0)
                engaged = knows and d_true < 4.5
                if not engaged and not g.dead:
                    log['disp30'].append(disp); log['loiter'].append(1.0 if disp < 1.5 else 0.0)
            peak, ppeak = _bm_peak(g)
            if peak is not None:
                log['bm_peak_err'].append(math.hypot(peak[0] - true_pac[0], peak[1] - true_pac[1]))
                if not knows:
                    log['bm_peak_err_unseen'].append(math.hypot(peak[0] - true_pac[0], peak[1] - true_pac[1]))
                    nom_cells = _nominated_cells(g, action.get(gid))
                    if nom_cells:
                        dn = min(math.hypot(y - peak[0], x - peak[1]) for (y, x) in nom_cells)
                        log['nom_to_peak'].append(dn)
                        dg = np.mean([math.hypot(y - g.y, x - g.x) for (y, x) in nom_cells])
                        log['nom_to_self'].append(float(dg))
                    if task is not None:
                        log['task_to_peak'].append(math.hypot(task.target_pos[0] - peak[0], task.target_pos[1] - peak[1]))
                        log['task_to_self'].append(math.hypot(task.target_pos[0] - g.y, task.target_pos[1] - g.x))
                    if gid in probs_per_ghost:
                        log['policy_mass_top10bm'].append(_bm_top_mass_on_cells(g, probs_per_ghost[gid], H, W))
            elif not knows:
                log['bm_empty'] += 1
            #denial opportunities: knows of a power pellet nearby and pacman is not a threat right now
            if g.known_power_pellets and not g.pacman_powered:
                best = None; bd = 1e9
                for (px, py) in g.known_power_pellets:
                    d = math.hypot(py - g.y, px - g.x)
                    if d < bd: bd = d; best = (py, px)
                threat = knows and math.hypot(g.known_pacman[0] - g.y, g.known_pacman[1] - g.x) < THREAT_RADIUS
                if bd < DENIAL_RADIUS and not threat:
                    has_convert = any(k[0] == int(TaskType.CONVERT) for k in g.cbba_agent.path)
                    heading = 0.0
                    if speed_mag > 1e-3: heading = ((best[0] - g.y) * g.vy + (best[1] - g.x) * g.vx) / (speed_mag * max(bd, 1e-6))
                    going = has_convert or heading > 0.5 or (task is not None and math.hypot(task.target_pos[0] - best[0], task.target_pos[1] - best[1]) < 1.5)
                    log['denial_opps'] += 1
                    log['denial_taken'] += 1 if going else 0
                    log['denial_dist'].append(bd)
            log['knows_pac'].append(1.0 if knows else 0.0)
            if knows: log['los_dist'].append(d_true)
            #ghost believes pacman powered vs truth, and how late it found out (only for ghosts Pacman could actually reach)
            if pac.powered:
                log['powered_aware'].append(1.0 if g.pacman_powered else 0.0)
                if g.pacman_powered and gid not in learned_at and power_start_frame >= 0:
                    learned_at[gid] = env.frame
                    if d_true < 16: log['aware_latency'].append(env.frame - power_start_frame)
        if any_known: frames_known_by_any += 1
        alive_g = [gg for gg in env.ghosts.values() if not gg.dead]
        n_al = len(alive_g)
        if n_al >= 2:
            import ghost as _gh
            rad = float(_gh.RADIUS)
            seen_i, comps = set(), []
            for i0 in range(n_al):
                if i0 in seen_i: continue
                stack, csz = [i0], 0
                while stack:
                    k = stack.pop()
                    if k in seen_i: continue
                    seen_i.add(k); csz += 1
                    for j0 in range(n_al):
                        if j0 not in seen_i and math.hypot(alive_g[k].y - alive_g[j0].y, alive_g[k].x - alive_g[j0].x) <= rad:
                            stack.append(j0)
                comps.append(csz)
            log['connectivity'].append(sum(c * c for c in comps) / float(n_al * n_al))
        obs, rew, done, info = env.step(action, want_bc=False)
        #--- frame-level facts we can only see after stepping ---
        if pac.powered and not prev_powered:
            power_start_frame = env.frame; activations += 1; learned_at = {}
            phase_dist = {gid: math.hypot(g.y - pac.y, g.x - pac.x) for gid, g in env.ghosts.items() if not g.dead}
            for gid, d in phase_dist.items(): log['dist_at_activation'].append(d)
        prev_powered = pac.powered
        if pac.powered: powered_frames += DECISION_INTERVAL
        for gid, g in env.ghosts.items():
            if g.dead and gid not in dead_seen:
                dead_seen.add(gid)
                d = math.hypot(g.y - pac.y, g.x - pac.x)
                fc = getattr(g, '_flee_cache', None)
                deaths.append({'frame': env.frame, 'knew_powered': bool(g.pacman_powered), 'knew_pac': g.known_pacman is not None,
                               'since_power': env.frame - power_start_frame if power_start_frame >= 0 else -1,
                               'dist_at_activation': phase_dist.get(gid, float('nan')),
                               'was_fleeing': bool(fc is not None and g.frame - fc[0] < 8),
                               'near_power_pellet': any(math.hypot(py - g.y, px - g.x) < 2.5 for (px, py) in g.known_power_pellets)})
        if done or env.frame >= MAX_FRAMES: break
    caught = bool(env.player.dead)
    alive = sum(1 for g in env.ghosts.values() if not g.dead)
    if caught: outcome = 'kill'
    elif alive == 0: outcome = 'wipe'
    elif (len(env.world.pellets) + len(env.world.power_pellets)) == 0: outcome = 'cleared'
    else: outcome = 'timeout'
    power_left = len(env.world.power_pellets)
    #pacman activations: count via score jumps is fragile; use powered edges recorded in the loop instead
    return {'seed': seed, 'outcome': outcome, 'frames': env.frame, 'alive': alive, 'deaths': deaths,
            'pac_score': float(env.player.score), 'power_start': power_before, 'power_left': power_left,
            'powered_frames': powered_frames, 'known_frac': frames_known_by_any / max(1, n_dec), 'activations': activations,
            'n_dec': n_dec}

def _new_log():
    return {'cell_entropy': [], 'eff_cells': [], 'logit_mean': [], 'logit_std': [], 'score_sat': [], 'throttle': [], 'gate': [], 'speed': [], 'fallback': [], 'no_task': [],
            'n_cands': [], 'cand_entropy': [], 'cand_top': [],
            'task_type': collections.Counter(), 'disp30': [], 'loiter': [], 'bm_peak_err': [], 'bm_peak_err_unseen': [],
            'nom_to_peak': [], 'nom_to_self': [], 'task_to_peak': [], 'task_to_self': [], 'policy_mass_top10bm': [], 'bm_empty': 0,
            'denial_opps': 0, 'denial_taken': 0, 'denial_dist': [], 'dist_at_activation': [], 'aware_latency': [], 'connectivity': [], 'pick_used': [], 'pick_adv': [],
            'origin': collections.Counter(), 'origin_score': collections.defaultdict(list), 'closing': collections.defaultdict(list), 'knows_pac': [], 'los_dist': [], 'powered_aware': []}

def _worker(mode, ckpt, stage_idx, seeds, radio=None, lidar=None):
    os.environ['OMP_NUM_THREADS'] = '1'; torch.set_num_threads(1)
    if radio is not None:
        import ghost as _gh
        _gh.RADIUS = float(radio)
    if lidar is not None:
        import ghost as _gh
        _gh.MAX_RAY_DIST = float(lidar)
    from worker import Env
    from net import GhostActor
    stage = STAGES[stage_idx]
    actor = None; critic = None
    if mode in ('rl', 'floor'):
        from net import GhostCritic
        if os.path.basename(str(ckpt)) == 'fresh':
            torch.manual_seed(0)
            actor = GhostActor(); critic = GhostCritic()
        else:
            ck = torch.load(ckpt, map_location='cpu', weights_only=False)
            actor = GhostActor(); actor.load_state_dict(ck['actor'])
            if 'critic' in ck:
                try:
                    critic = GhostCritic(); critic.load_state_dict(ck['critic'])
                except Exception:
                    critic = None   #old checkpoint: no Q-critic -> picks never executed (heuristic floor)
        actor.eval()
        if critic is not None: critic.eval()
    elif mode == 'floor':
        if os.path.exists(ckpt):
            try:
                ck = torch.load(ckpt, map_location='cpu', weights_only=False)
                actor = GhostActor(); actor.load_state_dict(ck['actor']); actor.eval()
            except Exception:
                actor = GhostActor(); actor.eval()
        else:
            actor = GhostActor(); actor.eval()
    env = Env(env_id=seeds[0], num_ghosts=stage.n_ghosts, world_height=float(stage.rows), world_width=float(stage.cols), obs_resolution=1.0, n_power=stage.n_power)
    log = _new_log(); games = []
    for s in seeds:
        games.append(_run_game(mode, actor, env, stage, s, log, critic))
    return games, log

def _merge(dst, src):
    for k, v in src.items():
        if isinstance(v, collections.Counter): dst[k].update(v)
        elif isinstance(v, collections.defaultdict):
            for kk, vv in v.items(): dst[k][kk].extend(vv)
        elif isinstance(v, list): dst[k].extend(v)
        else: dst[k] += v

def _q(xs, p): return float(np.percentile(xs, p)) if len(xs) else float('nan')
def _m(xs): return float(np.mean(xs)) if len(xs) else float('nan')

def report(mode, games, log):
    n = len(games)
    oc = collections.Counter(g['outcome'] for g in games)
    deaths = [d for g in games for d in g['deaths']]
    print(f"\n==================== {mode.upper()}  ({n} games) ====================")
    print(f"outcomes: " + ", ".join(f"{k} {v / n:.0%}" for k, v in sorted(oc.items())))
    kf = [g['frames'] for g in games if g['outcome'] == 'kill']
    print(f"kill rate {oc['kill'] / n:.3f} | mean frames {_m([g['frames'] for g in games]):.0f} | pac score {_m([g['pac_score'] for g in games]):.0f}")
    print(f"TARGET METRICS  time-to-kill {_m(kf):.0f} frames (median {_q(kf, 50):.0f}, p90 {_q(kf, 90):.0f}, kills only)"
          f" | ghosts lost {len(deaths) / n:.2f}/game | pacman score {_m([g['pac_score'] for g in games]):.0f}")
    print(f"ghost deaths/game {len(deaths) / n:.2f} | powered share of game {_m([g['powered_frames'] / max(1, g['frames']) for g in games]):.0%} | some ghost knows Pacman {_m([g['known_frac'] for g in games]):.0%} of decisions")
    conv = [g['power_start'] - g['power_left'] - g['activations'] for g in games]
    print(f"power pellets: start {_m([g['power_start'] for g in games]):.1f} | eaten by Pacman {_m([g['activations'] for g in games]):.1f}/game | converted by ghosts {_m(conv):.1f}/game | left at end {_m([g['power_left'] for g in games]):.1f}")
    if deaths:
        print(f"deaths: knew-powered {np.mean([d['knew_powered'] for d in deaths]):.0%} | knew-pacman-pos {np.mean([d['knew_pac'] for d in deaths]):.0%} | "
              f"median frames after power start {np.median([d['since_power'] for d in deaths if d['since_power'] >= 0]) if any(d['since_power'] >= 0 for d in deaths) else float('nan'):.0f} | "
              f"within 2.5 of a known power pellet {np.mean([d['near_power_pellet'] for d in deaths]):.0%} | had a flee target computed in last 8 frames {np.mean([d['was_fleeing'] for d in deaths]):.0%}")
        da = np.array([d['dist_at_activation'] for d in deaths if not math.isnan(d['dist_at_activation'])])
        all_da = np.array(log['dist_at_activation'])
        if len(da) and len(all_da):
            print("P(death this phase | ghost distance to Pacman when it ate the pellet):")
            for lo, hi in [(0, 5), (5, 8), (8, 12), (12, 16), (16, 25), (25, 99)]:
                n_all = int(((all_da >= lo) & (all_da < hi)).sum()); n_d = int(((da >= lo) & (da < hi)).sum())
                if n_all: print(f"    {lo:>2}-{hi:<2} cells: {n_d / n_all:5.0%}  ({n_d}/{n_all})")
    print(f"radio-graph connectivity (mean frac of swarm reachable): {_m(log['connectivity']):.0%}")
    print(f"powered-awareness while Pacman IS powered: {_m(log['powered_aware']):.0%} of ghost-decisions | ghosts within 16 cells learned of it after median {_q(log['aware_latency'], 50):.0f} frames (p75 {_q(log['aware_latency'], 75):.0f}) of the 40-frame window")
    print("--- movement ---")
    print(f"|v|/vmax mean {_m(log['speed']):.2f} | fallback-mode {_m(log['fallback']):.0%} | no active task {_m(log['no_task']):.0%} | "
          f"loiter (<1.5 cells moved in 30 frames, not engaged) {_m(log['loiter']):.0%} | disp30 median {_q(log['disp30'], 50):.1f} cells")
    tt = log['task_type']; tot = sum(tt.values()) or 1
    names = {-1: 'none', 0: 'HUNT', 1: 'CONVERT', 2: 'EVADE', 3: 'EXPLORE', 4: 'DYNAMIC', 5: 'FLANK'}
    print("active task mix: " + ", ".join(f"{names.get(k, k)} {v / tot:.0%}" for k, v in sorted(tt.items())))
    print("--- belief map ---")
    print(f"peak error vs TRUE Pacman: all {_q(log['bm_peak_err'], 50):.1f} med | when unseen {_q(log['bm_peak_err_unseen'], 50):.1f} med, {_m(log['bm_peak_err_unseen']):.1f} mean, p25 {_q(log['bm_peak_err_unseen'], 25):.1f}, p75 {_q(log['bm_peak_err_unseen'], 75):.1f} | empty-belief decisions {log['bm_empty']}")
    if log['nom_to_peak']:
        print(f"closest nomination -> own belief peak (unseen): median {_q(log['nom_to_peak'], 50):.1f} cells, mean {_m(log['nom_to_peak']):.1f}, within 3 cells {np.mean(np.array(log['nom_to_peak']) <= 3.0):.0%} | nominations are {_m(log['nom_to_self']):.1f} cells from the ghost on average")
    if log['task_to_peak']:
        print(f"active task target -> own belief peak (unseen): median {_q(log['task_to_peak'], 50):.1f}, within 3 cells {np.mean(np.array(log['task_to_peak']) <= 3.0):.0%} | task target {_m(log['task_to_self']):.1f} cells from ghost")
    if log['policy_mass_top10bm']:
        print(f"policy prob mass on belief top-10 cells: mean {_m(log['policy_mass_top10bm']):.3f} (uniform over ~800 open cells would be ~0.012)")
    if log['cell_entropy']:
        print(f"policy cell entropy: mean {_m(log['cell_entropy']):.2f} nats -> effective cells {_m(log['eff_cells']):.0f} | throttle mean {_m(log['throttle']):.2f} | hijack gate rate {_m(log['gate']):.2f}")
        print(f"raw logits: mean {_m(log['logit_mean']):.1f} (std within map {_m(log['logit_std']):.2f}) -> sigmoid CBBA scores saturated at 1.0 for {_m(log['score_sat']):.0%} of open cells")
    if log['n_cands']:
        print(f"POINTER HEAD  {_m(log['n_cands']):.1f} live candidates/decision"
              f" | entropy {_m(log['cand_entropy']):.2f} of max (1.00 = undecided, 0.00 = fully committed)"
              f" | top-candidate prob {_m(log['cand_top']):.2f}")
    ONAMES = {-1: 'no task', ORIGIN_HEURISTIC: 'heuristic', ORIGIN_RL_ENDORSE: 'RL-endorsed', ORIGIN_RL_NOVEL: 'RL-novel'}
    o_tot = sum(log['origin'].values()) or 1
    if log['pick_used']:
        pa = np.array(log['pick_adv']); pu = np.array(log['pick_used'])
        print(f"EXECUTE GATE  pick executed on {pu.mean():.0%} of decisions | counterfactual adv of the pick: mean {pa.mean():+.3f}, "
              f"when executed {pa[pu > 0.5].mean() if (pu > 0.5).any() else float('nan'):+.3f}, when deferred {pa[pu < 0.5].mean() if (pu < 0.5).any() else float('nan'):+.3f} (normalised-return units)")
    print("--- RL attribution (who won the executed task) ---")
    for k in sorted(log['origin']):
        sc = log['origin_score'].get(k, [])
        cl = log['closing'].get(k, [])
        cl_s = f"{_m(cl):+.2f}" if cl else "  n/a"
        sc_s = f"{_m(sc):.2f}" if sc else " n/a"
        print(f"  {ONAMES.get(k, k):<12} {log['origin'][k] / o_tot:>5.0%} of decisions | mean winning score {sc_s} | dist to Pacman over next 5 decisions {cl_s}")
    rl_share = (log['origin'][ORIGIN_RL_ENDORSE] + log['origin'][ORIGIN_RL_NOVEL]) / o_tot
    print(f"  RL share of executed tasks: {rl_share:.1%}   (negative 'dist' = closing in; compare RL rows against heuristic)")
    print("--- power pellet denial ---")
    if log['denial_opps']:
        print(f"opportunities (pellet known within {DENIAL_RADIUS:.0f}, Pacman not a threat): {log['denial_opps']} ghost-decisions, ghost heading for it / holding CONVERT: {log['denial_taken'] / log['denial_opps']:.0%} | median pellet dist {_q(log['denial_dist'], 50):.1f}")
    print(f"LOS: knows Pacman {_m(log['knows_pac']):.0%} of ghost-decisions; when known, distance median {_q(log['los_dist'], 50):.1f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='checkpoints/ckpt_1000.pt')
    ap.add_argument('--mode', default='rl', choices=['rl', 'heuristic', 'random', 'floor'])
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--stage', type=int, default=len(STAGES) - 1)
    ap.add_argument('--seed0', type=int, default=0)
    ap.add_argument('--radio', type=float, default=None, help='override ghost comms radius (default: repo value)')
    ap.add_argument('--lidar', type=float, default=None, help='override ghost sensing range (default: repo value)')
    args = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(root, args.ckpt)
    seeds = list(range(args.seed0, args.seed0 + args.n))
    chunks = [seeds[i::args.workers] for i in range(args.workers) if seeds[i::args.workers]]
    t0 = time.time(); games = []; log = _new_log()
    with ProcessPoolExecutor(max_workers=len(chunks)) as ex:
        futs = [ex.submit(_worker, args.mode, ckpt, args.stage, c, args.radio, args.lidar) for c in chunks]
        for f in as_completed(futs):
            g, l = f.result(); games.extend(g); _merge(log, l)
            print(f"  {len(games)}/{args.n} games ({time.time() - t0:.0f}s)", end='\r', flush=True)
    games.sort(key=lambda g: g['seed'])
    report(f"{args.mode} {os.path.basename(ckpt) if args.mode == 'rl' else ''} stage{args.stage}", games, log)
    print("per-game: " + " ".join(f"{g['outcome'][0]}{g['frames']}/{g['alive']}a" for g in games))

if __name__ == '__main__':
    main()
