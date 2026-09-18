#!/usr/bin/env python3
import os
import sys
import math
import numpy as np
import torch

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
os.environ['SDL_VIDEODRIVER'] = 'dummy'

from net import GhostActor
from worker import Env
from curriculum import STAGES
from test import _pad_spatial, K_NOMINATIONS

def run_diagnostics(ckpt_path: str = None, stage_idx: int = 1, seed: int = 42, max_frames: int = 250):
    print(f"=== Running Pure RL Diagnostics on Stage {stage_idx} (seed={seed}) ===")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    actor_vec_dim = 10
    actor = GhostActor(vec_dim=actor_vec_dim).cpu()
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if 'vec_mlp.0.weight' in ckpt['actor']:
            actor_vec_dim = ckpt['actor']['vec_mlp.0.weight'].shape[1]
            actor = GhostActor(vec_dim=actor_vec_dim).cpu()
        try:
            actor.load_state_dict(ckpt['actor'])
            print(f"Loaded checkpoint from {ckpt_path}")
        except Exception as e:
            print(f"Notice: Checkpoint has incompatible architecture ({e}). Mapping matching weights...")
            actor_sd = ckpt['actor']
            new_actor_sd = actor.state_dict()
            for k, v in actor_sd.items():
                if k in new_actor_sd:
                    if v.shape == new_actor_sd[k].shape:
                        new_actor_sd[k] = v
                    elif "vec_mlp.0.weight" in k and v.ndim == 2 and new_actor_sd[k].ndim == 2:
                        min_out = min(v.shape[0], new_actor_sd[k].shape[0])
                        min_in = min(v.shape[1], new_actor_sd[k].shape[1])
                        new_actor_sd[k][:min_out, :min_in] = v[:min_out, :min_in]
            actor.load_state_dict(new_actor_sd)
    else:
        print("Running diagnostics with freshly initialized weights.")
    actor.eval()
    
    stage = STAGES[min(len(STAGES) - 1, stage_idx)]
    env = Env(env_id=seed, num_ghosts=stage.n_ghosts, world_height=float(stage.rows), world_width=float(stage.cols), obs_resolution=stage.obs_resolution, n_power=stage.n_power)
    
    obs = env.reset()
    
    g0_positions = []
    g0_velocities = []
    b_sums = []
    los_transitions = 0
    top_cell_teleports = 0
    ping_pong_reversals = 0
    
    total_steps = 0
    
    while env.frame < max_frames:
        if obs is None:
            break
        gids, sp, ve, vm, ht, hs, global_sp, grid_shape = obs
        if not gids:
            break
            
        sp_p = _pad_spatial(sp.astype(np.float32), stage.rows, stage.cols)
        vm_p = _pad_spatial(vm.astype(np.float32), stage.rows, stage.cols).astype(bool)
        t_sp = torch.from_numpy(sp_p)
        t_ve = torch.from_numpy(ve[:, :actor_vec_dim].astype(np.float32))
        t_vm = torch.from_numpy(vm_p)
        
        with torch.inference_mode():
            idx, lp, scores, _pool, _vec, speed, _speed_lp, direction, _dir_lp = actor(t_sp, t_ve, t_vm, K=K_NOMINATIONS)
            
        idx_np    = idx.cpu().numpy()
        scores_np = scores.float().cpu().numpy()
        speed_np  = speed.float().cpu().numpy()
        dir_np    = direction.float().cpu().numpy()
        
        action_dict = {}
        for i, gid in enumerate(gids):
            pairs = [(int(x // stage.cols), int(x % stage.cols)) for x in idx_np[i]]
            action_dict[gid] = (pairs, scores_np[i], float(speed_np[i].item()), float(dir_np[i].item()))
            
        # Step environment
        obs, rewards, done, info = env.step(action_dict, bc_prob=0.0)
        total_steps += 1
        
        # Check Ghost 0
        g0 = env.ghosts.get(0)
        if g0 and not g0.dead:
            g0_positions.append((g0.y, g0.x))
            g0_velocities.append((g0.vy, g0.vx))
            
            # Check for immediate 180° reversal between steps
            if len(g0_velocities) >= 2:
                v1 = g0_velocities[-2]
                v2 = g0_velocities[-1]
                mag1 = math.hypot(v1[0], v1[1])
                mag2 = math.hypot(v2[0], v2[1])
                if mag1 > 0.1 and mag2 > 0.1:
                    dot = (v1[0]*v2[0] + v1[1]*v2[1]) / (mag1 * mag2)
                    if dot < -0.85:
                        ping_pong_reversals += 1
                        
            # Check belief map sum
            bsum = float(g0.belief_map._b_flat.sum())
            b_sums.append(bsum)
            
            # Check top cells
            top = g0.belief_map.top_cells(n=1)
            if top:
                top_pos = top[0]
                # Defect was top cell snapping to top-left corner (0.4, 0.4)
                if abs(top_pos[0] - 0.4) < 0.15 and abs(top_pos[1] - 0.4) < 0.15:
                    # Only flag if pacman is far from (0.4, 0.4)
                    if math.hypot(env.player.y - 0.4, env.player.x - 0.4) > 3.0:
                        top_cell_teleports += 1
                        
            if g0.known_pacman is None and getattr(g0, '_had_los_prev', False):
                los_transitions += 1
                
        # Pure RL task check: Verify no tasks have heuristic scores (> 1.0)
        for gid in gids:
            g = env.ghosts[gid]
            act = g.cbba_agent.get_active_task()
            if act is not None:
                assert act.score <= 1.01, f"Found heuristic task with score {act.score} in RL mode!"
                
        if done:
            break
            
    print(f"\n--- Diagnostic Results (Frames={env.frame}, Steps={total_steps}) ---")
    print(f"Ping-pong reversals (Ghost 0): {ping_pong_reversals}")
    min_bsum = min(b_sums) if b_sums else 0.0
    max_bsum = max(b_sums) if b_sums else 0.0
    print(f"Belief map sum min: {min_bsum:.6f}, max: {max_bsum:.6f}")
    print(f"Top cell teleport to (0.4, 0.4) count: {top_cell_teleports}")
    print(f"LOS loss transitions: {los_transitions}")
    print(f"Pacman caught: {env.player.dead}")
    
    # Assertions
    assert abs(min_bsum - 1.0) < 1e-4, f"Belief map sum underflow/leakage! Min was {min_bsum}"
    assert abs(max_bsum - 1.0) < 1e-4, f"Belief map sum overflow! Max was {max_bsum}"
    assert top_cell_teleports == 0, f"Detected {top_cell_teleports} top cell teleports to (0.4, 0.4)!"
    assert ping_pong_reversals <= 3, f"Excessive ping-pong reversals: {ping_pong_reversals}"
    print("\n✓ ALL DIAGNOSTIC CHECKS PASSED!")
    return True

if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    stage = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    run_diagnostics(ckpt_path=ckpt, stage_idx=stage)
