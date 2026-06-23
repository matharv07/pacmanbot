#!/usr/bin/env python3

import os
import sys
import glob
import argparse
import time
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from net import GhostActor
from worker import Env
from curriculum import STAGES
import traceback

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', 'hide')
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

DECISION_INTERVAL  = 6
K_NOMINATIONS      = 3
MAX_EPISODE_FRAMES = 3000

def _pad_spatial(arr, target_h, target_w):
    h, w = arr.shape[-2], arr.shape[-1]
    if h == target_h and w == target_w:
        return arr
    pad_h = target_h - h
    pad_w = target_w - w
    if arr.ndim == 2:
        return np.pad(arr, ((0, pad_h), (0, pad_w)))
    elif arr.ndim == 3:
        return np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)))
    elif arr.ndim == 4:
        return np.pad(arr, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)))
    return arr

def _run_episode(actor, env, stage):
    obs = env.reset()
    while True:
        if obs is None:
            break
        gids, sp, ve, vm, ht, global_sp, grid_shape = obs
        if not gids:
            break
        sp_p = _pad_spatial(sp.astype(np.float32), stage.rows, stage.cols)
        vm_p = _pad_spatial(vm.astype(np.float32), stage.rows, stage.cols).astype(bool)
        t_sp = torch.from_numpy(sp_p)
        t_ve = torch.from_numpy(ve.astype(np.float32))
        t_vm = torch.from_numpy(vm_p)
        with torch.inference_mode():
            idx, lp, scores, _pool, _vec = actor(t_sp, t_ve, t_vm, K=K_NOMINATIONS)
        idx_np    = idx.numpy()
        scores_np = scores.numpy()
        action_dict = {}
        for i, gid in enumerate(gids):
            pairs = [(int(x // stage.cols), int(x % stage.cols)) for x in idx_np[i]]
            action_dict[gid] = (pairs, scores_np[i])
        obs, _rewards, done, info = env.step(action_dict, bc_prob=0.0)
        if done:
            surviving = sum(1 for g in env.ghosts.values() if not g.dead)
            pac_score = float(info.get('pacman_score', 0))
            pacman_caught = bool(env.player.dead)
            return env.frame, surviving, pac_score, pacman_caught
        if env.frame >= MAX_EPISODE_FRAMES:
            surviving = sum(1 for g in env.ghosts.values() if not g.dead)
            pac_score = float(info.get('pacman_score', 0))
            return env.frame, surviving, pac_score, False
    surviving = sum(1 for g in env.ghosts.values() if not g.dead)
    return env.frame, surviving, 0.0, False

def _worker_chunk(ckpt_path: str, n_games: int, stage_override=None, seed_offset: int = 0):
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    np.random.seed(seed_offset)
    torch.manual_seed(seed_offset)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    stage_idx = ckpt['curriculum']['stage_idx']
    stage     = STAGES[stage_override] if stage_override is not None else STAGES[stage_idx]
    actor = GhostActor().cpu()
    actor.load_state_dict(ckpt['actor'])
    actor.eval()
    env = Env(env_id=seed_offset, num_ghosts=stage.n_ghosts, grid_rows=stage.rows, grid_cols=stage.cols, n_power=stage.n_power)
    results = []
    for i in range(n_games):
        np.random.seed(seed_offset * 1000 + i)
        r = _run_episode(actor, env, stage)
        results.append(r)
    return ckpt_path, stage_idx, stage, results

def _aggregate(chunks):
    frames_all, surv_all, pac_all, caught_all = [], [], [], []
    for frames, surv, pac, caught in chunks:
        frames_all.append(frames if caught else MAX_EPISODE_FRAMES)
        surv_all.append(surv if caught else 0)
        pac_all.append(pac)
        caught_all.append(caught)
    n = len(frames_all)
    n_kills = sum(caught_all)

    return {'n'             : n,
        'n_kills'       : n_kills,
        'kill_rate'     : n_kills / n,
        'mean_frames'   : float(np.mean(frames_all)),
        'mean_surviving': float(np.mean(surv_all)),
        'mean_pac_score': float(np.mean(pac_all))}

def _print_table(rows, title="Ghost-Team Checkpoint Comparison"):
    cols = [('Checkpoint',   '<', 14),
        ('Stage',        '^',  7),
        ('Grid',         '^',  8),
        ('Ghosts',       '^',  7),
        ('Kill%',        '^',  7),
        ('Frames↓',      '^', 10),
        ('Surv.Ghosts',  '^', 13),
        ('PacScore',     '^', 10),]
    hfmt = '  '.join(f'{{:{a}{w}}}' for _, a, w in cols)
    dfmt = '  '.join(f'{{:{a}{w}}}' for _, a, w in cols)
    bar = '─' * (sum(w for _, _, w in cols) + 2 * (len(cols) - 1))
    print(f'\n{"─"*6} {title} {"─"*6}')
    print(bar)
    print(hfmt.format(*[c for c, _, _ in cols]))
    print(bar)
    for r in rows:
        frames_s = f"{r['mean_frames']:.0f}" if not np.isnan(r['mean_frames']) else '—'
        cells = [r['label'],f"S{r['stage']}",r['grid'],str(r['ghosts']),f"{r['kill_rate']*100:.1f}%",frames_s,f"{r['mean_surviving']:.2f}",f"{r['mean_pac_score']:.1f}",]
        print(dfmt.format(*cells))
    print(bar)
    print()
    print('  Kill%       = % episodes a ghost caught Pacman')
    print(f'  Frames↓     = mean game frames to catch Pacman across ALL episodes')
    print(f'                Non-kill episodes (timeout / pellets eaten / all ghosts dead)')
    print(f'                contribute {MAX_EPISODE_FRAMES} frames as a penalty')
    print('  Surv.Ghosts = mean ghosts alive at end; non-kill episodes count as 0')
    print('  PacScore    = mean Pacman score across ALL episodes (lower = better for ghosts)')
    print()

def main():
    ap = argparse.ArgumentParser(description='Evaluate ghost-team checkpoints.')
    ap.add_argument('--n',        type=int,  default=100,  help='Games per checkpoint (default: 100)')
    ap.add_argument('--chunk',    type=int,  default=25,   help='Games per worker chunk (default: 25)')
    ap.add_argument('--workers',  type=int,  default=None, help='Max parallel workers (default: cpu count)')
    ap.add_argument('--ckpts',    type=int,  nargs='*',    help='Specific update numbers, e.g. 1100 1600')
    ap.add_argument('--stage',    type=int,  default=4,    help='Force all checkpoints onto this stage index (default: 4 = 33x41 / 7 ghosts)')
    ap.add_argument('--ckpt_dir', type=str,  default='checkpoints', help='Checkpoint directory')
    args = ap.parse_args()
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.ckpt_dir)
    if args.ckpts:
        ckpt_paths = [os.path.join(ckpt_dir, f'ckpt_{u}.pt') for u in sorted(args.ckpts)]
        missing = [p for p in ckpt_paths if not os.path.exists(p)]
        if missing:
            print(f'ERROR: checkpoints not found: {missing}')
            sys.exit(1)
    else:
        ckpt_paths = sorted(glob.glob(os.path.join(ckpt_dir, 'ckpt_*.pt')),
                            key=lambda p: int(os.path.basename(p).split('_')[1].split('.')[0]))
    if not ckpt_paths:
        print(f'No checkpoints found in {ckpt_dir}')
        sys.exit(1)
    n_games     = args.n
    chunk_sz    = min(args.chunk, n_games)
    max_workers = args.workers or os.cpu_count() or 4
    tasks = []
    for ckpt_path in ckpt_paths:
        n_chunks = max(1, n_games // chunk_sz)
        rem      = n_games % chunk_sz
        for ci in range(n_chunks):
            games = chunk_sz + (rem if ci == 0 else 0)
            tasks.append((ckpt_path, games, args.stage, ci * 1000))
    n_total_workers = min(len(tasks), max_workers)
    print(f'\n  Evaluating {len(ckpt_paths)} checkpoint(s) × {n_games} games each')
    print(f'  {len(tasks)} total chunks  →  {n_total_workers} parallel workers\n')
    raw:  dict[str, list]  = defaultdict(list)
    meta: dict[str, tuple] = {}
    t0 = time.time()
    completed = 0
    with ProcessPoolExecutor(max_workers=n_total_workers) as ex:
        futures = {ex.submit(_worker_chunk, *task): task[0] for task in tasks}
        for fut in as_completed(futures):
            ckpt_path = futures[fut]
            try:
                path_ret, stage_idx, stage, chunk_results = fut.result()
                raw[ckpt_path].extend(chunk_results)
                meta[ckpt_path] = (stage_idx, stage)
            except Exception as e:
                print(f'  ✗ {os.path.basename(ckpt_path)}: {e}')
                traceback.print_exc()
            completed += 1
            elapsed = time.time() - t0
            pct     = completed / len(tasks) * 100
            print(f'  [{pct:5.1f}%]  {completed}/{len(tasks)} chunks done  ({elapsed:.0f}s elapsed)', end='\r')
    print()
    rows = []
    for ckpt_path in ckpt_paths:
        if ckpt_path not in raw:
            continue
        stage_idx, stage = meta[ckpt_path]
        agg = _aggregate(raw[ckpt_path])
        rows.append({'label': os.path.basename(ckpt_path).replace('.pt', ''), 'stage': stage_idx, 'grid' : f'{stage.rows}×{stage.cols}', 'ghosts': stage.n_ghosts, **agg})
    _print_table(rows)
    if len(rows) > 1:
        best_kill  = max(rows, key=lambda r: r['kill_rate'])
        best_speed = min((r for r in rows if not np.isnan(r['mean_frames'])),
                         key=lambda r: r['mean_frames'], default=None)
        best_surv  = max(rows, key=lambda r: r['mean_surviving'])
        best_pac   = min(rows, key=lambda r: r['mean_pac_score'])
        print('Best performers:')
        print(f'Highest kill rate  → {best_kill["label"]} ({best_kill["kill_rate"]*100:.1f}%)')
        if best_speed:
            print(f'Fastest kill       → {best_speed["label"]} ({best_speed["mean_frames"]:.0f} frames)')
        print(f'Most ghosts alive  → {best_surv["label"]} ({best_surv["mean_surviving"]:.2f})')
        print(f'Lowest pac score   → {best_pac["label"]} ({best_pac["mean_pac_score"]:.1f})')
        print()

if __name__ == '__main__':
    main()