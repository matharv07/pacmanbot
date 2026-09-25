#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import math
import collections
import numpy as np
import torch

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', 'hide')
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

from curriculum import STAGES
from worker import Env
from net import GhostActor
from obs import (flatten_cand_cells, build_spatial, build_vector, build_valid_mask,
                 actions_to_tasks, MAX_H, MAX_W, SPATIAL_CH, VEC_DIM)
from allocator import TaskType

def monitor_game(ckpt_path, stage_num=3, seed=42, max_frames=2000):
    print(f"\n=================================================================")
    print(f"MONITORING GAME: {ckpt_path} | Stage {stage_num} | Seed {seed}")
    print(f"=================================================================")
    
    stage = STAGES[stage_num]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load actor
    actor = GhostActor(VEC_DIM).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("actor", ckpt.get("model", ckpt))
    actor.load_state_dict(state_dict)
    actor.eval()
    
    # Init Env
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = Env(0, num_ghosts=stage.n_ghosts,
              world_height=stage.world_height, world_width=stage.world_width,
              n_power=stage.n_power,
              obs_resolution=stage.obs_resolution,
              pac_speed=getattr(stage, 'pac_speed', 1.0))
    env.reset()
    
    # Tracking metrics across the game
    speed_samples = []
    low_speed_events = [] # (|v| < 0.4)
    near_miss_events = [] # ghost near pacman but missed
    corner_stutter_events = [] # sharp velocity changes / oscillations
    ghost_congestion_events = [] # multiple ghosts within 1.2 cells of each other
    overshoot_events = [] # ghost bypassed pacman
    los_frames = collections.defaultdict(int)
    strike_frames = collections.defaultdict(int)
    
    frame = 0
    decision_step = 0
    pacman_alive = True
    
    prev_ghost_pos = {gid: (g.y, g.x) for gid, g in env.ghosts.items()}
    prev_ghost_vel = {gid: (0.0, 0.0) for gid, g in env.ghosts.items()}
    
    while frame < max_frames and not env.player.dead:
        # Decision step every 6 frames
        if frame % 6 == 0:
            decision_step += 1
            alive_gids = [gid for gid, g in env.ghosts.items() if not g.dead]
            if not alive_gids:
                break
                
            action_dict = {}
            for gid in alive_gids:
                g = env.ghosts[gid]
                g.rl_mode = True
                
                rows = int(stage.world_height * stage.obs_resolution)
                cols = int(stage.world_width * stage.obs_resolution)
                recent_noms = np.zeros((rows, cols), dtype=np.float32)
                sp = build_spatial(g, recent_noms, rows, cols, stage.obs_resolution)
                ve = build_vector(g)
                vm = build_valid_mask(g, rows, cols, stage.obs_resolution)
                
                # Pad to MAX_H, MAX_W
                sp_pad = np.zeros((SPATIAL_CH, MAX_H, MAX_W), dtype=np.float32)
                sp_pad[:, :rows, :cols] = sp
                vm_pad = np.zeros((MAX_H, MAX_W), dtype=bool)
                vm_pad[:rows, :cols] = vm
                
                t_sp = torch.from_numpy(sp_pad).unsqueeze(0).to(device)
                t_ve = torch.from_numpy(ve).unsqueeze(0).to(device)
                t_vm = torch.from_numpy(vm_pad).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    out = actor(t_sp, t_ve, t_vm, K=3, return_restruct=True)
                    picks, lps, sc_map, pool, vec, spd_idx, spd_lp, flat_logits, res_act, _, _ = out
                    
                    picks_np = picks[0].cpu().numpy()
                    scores_np = sc_map[0].cpu().numpy()[:rows, :cols]
                    spd_val = spd_idx[0].item() if isinstance(spd_idx, torch.Tensor) else spd_idx
                    res_val = res_act[0].item() if isinstance(res_act, torch.Tensor) else 0
                    
                    action_dict[gid] = (picks_np, scores_np, spd_val, bool(res_val > 0.5))
            
            # Step env
            _, rewards, done, info = env.step(action_dict)
        
        frame += 6 # each step advances 6 frames internally
        
        # Monitor ghost physical movements after the step
        for gid, g in env.ghosts.items():
            if g.dead:
                continue
            
            v_mag = math.hypot(g.vx, g.vy)
            speed_samples.append(v_mag)
            
            d_pac = math.hypot(g.y - env.player.y, g.x - env.player.x)
            has_los = (g.known_pacman is not None)
            if has_los:
                los_frames[gid] += 6
            if getattr(g, '_is_striking', False):
                strike_frames[gid] += 6
                
            # Flaw detection 1: Ghost sluggishness or stall (< 0.45 speed while pursuing or roaming)
            if v_mag < 0.45:
                low_speed_events.append({
                    'frame': frame, 'gid': gid, 'speed': v_mag,
                    'pos': (round(g.y, 2), round(g.x, 2)),
                    'd_pac': round(d_pac, 2),
                    'active_task': str(g.cbba_agent.get_active_task()),
                    'is_powered': env.player.powered
                })
                
            # Flaw detection 2: Direction reversal / Corner stutter
            old_vy, old_vx = prev_ghost_vel[gid]
            if math.hypot(old_vy, old_vx) > 0.3 and v_mag > 0.3:
                dot = (g.vy * old_vy + g.vx * old_vx) / (math.hypot(old_vy, old_vx) * v_mag)
                if dot < -0.5: # 180 degree reversal or sharp oscillation
                    corner_stutter_events.append({
                        'frame': frame, 'gid': gid, 'dot': round(dot, 2),
                        'pos': (round(g.y, 2), round(g.x, 2)),
                        'd_pac': round(d_pac, 2)
                    })
            
            # Flaw detection 3: Near Miss / Missed Strike (within 3.0 cells of Pacman, but did not catch)
            if d_pac <= 3.0 and not env.player.dead and not env.player.powered:
                # check if ghost is moving AWAY from Pacman
                dr = env.player.y - g.y
                dc = env.player.x - g.x
                d_norm = math.hypot(dr, dc)
                if d_norm > 0.1 and v_mag > 0.1:
                    cos_align = (g.vy * (dr/d_norm) + g.vx * (dc/d_norm)) / v_mag
                    if cos_align < 0.0: # ghost is moving AWAY despite being within 3 cells!
                        near_miss_events.append({
                            'frame': frame, 'gid': gid, 'd_pac': round(d_pac, 2),
                            'cos_align': round(cos_align, 2),
                            'active_task': str(g.cbba_agent.get_active_task())
                        })
            
            prev_ghost_pos[gid] = (g.y, g.x)
            prev_ghost_vel[gid] = (g.vy, g.vx)
            
        # Flaw detection 4: Multi-ghost congestion in narrow 1-cell corridors
        alive_list = [g for g in env.ghosts.values() if not g.dead]
        for i in range(len(alive_list)):
            for j in range(i + 1, len(alive_list)):
                g1, g2 = alive_list[i], alive_list[j]
                d_inter = math.hypot(g1.y - g2.y, g1.x - g2.x)
                if d_inter < 1.0: # crowding/bumping against each other
                    ghost_congestion_events.append({
                        'frame': frame, 'g1': g1.gid, 'g2': g2.gid, 'dist': round(d_inter, 2),
                        'pos': (round(g1.y, 2), round(g1.x, 2))
                    })
    
    kill = bool(env.player.dead)
    print(f"GAME RESULT: {'KILL (WIN)' if kill else 'ESCAPE (LOSS)'}")
    print(f"Total Frames: {frame} | Pacman Score: {env.player.score} | Ghosts Dead: {sum(1 for g in env.ghosts.values() if g.dead)}")
    print(f"Average Ghost Speed: {np.mean(speed_samples):.3f} (Max possible: 1.000)")
    print(f"Low Speed Frames (|v| < 0.45): {len(low_speed_events)} / {len(speed_samples)} ({len(low_speed_events)/len(speed_samples)*100:.1f}%)")
    print(f"Sudden Stutter/Reversals: {len(corner_stutter_events)}")
    print(f"Near-Miss Divergences (d <= 3.0 but moving away): {len(near_miss_events)}")
    print(f"Inter-Ghost Crowding Events (d < 1.0): {len(ghost_congestion_events)}")
    
    if near_miss_events:
        print("\n--- SAMPLE NEAR-MISS EVENTS (Ghost close to Pacman but steering away) ---")
        for ev in near_miss_events[:5]:
            print(f"  Frame {ev['frame']}: Ghost {ev['gid']} dist={ev['d_pac']} cos_align={ev['cos_align']} task={ev['active_task']}")
            
    if low_speed_events:
        print("\n--- SAMPLE LOW SPEED / STALL EVENTS ---")
        for ev in low_speed_events[:5]:
            print(f"  Frame {ev['frame']}: Ghost {ev['gid']} speed={ev['speed']:.2f} pos={ev['pos']} d_pac={ev['d_pac']} task={ev['active_task']}")

    if ghost_congestion_events:
        print("\n--- SAMPLE CROWDING / CONGESTION EVENTS ---")
        for ev in ghost_congestion_events[:5]:
            print(f"  Frame {ev['frame']}: Ghosts {ev['g1']} & {ev['g2']} dist={ev['dist']} pos={ev['pos']}")
            
    return {
        'kill': kill, 'frames': frame, 'score': env.player.score,
        'mean_speed': float(np.mean(speed_samples)),
        'low_speed_pct': len(low_speed_events)/len(speed_samples)*100,
        'stutters': len(corner_stutter_events),
        'near_misses': len(near_miss_events),
        'congestions': len(ghost_congestion_events)
    }

if __name__ == '__main__':
    ckpts = ['checkpoints/ckpt_815_stage_3.pt', 'checkpoints/ckpt_1000.pt']
    for cp in ckpts:
        if os.path.exists(cp):
            res = monitor_game(cp, stage_num=3, seed=100)
