import os
import sys
import torch
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from curriculum import STAGES, CurriculumScheduler
from worker import Env
from net import GhostActor, GhostCritic
from test import _run_episode, _pad_spatial

from obs import SPATIAL_CH, VEC_DIM

def test_curriculum_logic():
    print("Testing curriculum logic...")
    cs = CurriculumScheduler(start_stage=0)
    assert cs.stage_idx == 0
    # Test kill rate gating (needs at least min_updates=100 and ADVANCE_WINDOW=100)
    for _ in range(100):
        cs.record_return(mean_return=5.0, kill_rate=0.75)
    assert cs.should_advance(), "Curriculum should advance with 75% kill rate"
    cs.advance()
    assert cs.stage_idx == 1, f"Expected Stage 1, got {cs.stage_idx}"
    print("✓ Curriculum test passed!")

def test_actor_critic_shapes_and_logprobs():
    print("Testing actor-critic forward and evaluate_actions...")
    actor = GhostActor()
    critic = GhostCritic()
    
    # 2 ghosts, 22x27 grid
    H, W = 22, 27
    sp = torch.randn(2, SPATIAL_CH, H, W)
    ve = torch.randn(2, VEC_DIM)
    vm = torch.ones(2, H, W, dtype=torch.bool)
    
    idx, lp, scores, pool, vec, speed, speed_lp = actor(sp, ve, vm, K=3)
    assert idx.shape == (2, 3), f"idx shape mismatch: {idx.shape}"
    assert lp.shape == (2, 3), f"lp shape mismatch: {lp.shape}"
    assert speed.shape == (2, 1), f"speed shape mismatch: {speed.shape}"
    assert speed_lp.shape == (2, 1), f"speed_lp shape mismatch: {speed_lp.shape}"
    
    # Check evaluate_actions
    eval_lp, eval_ent, _pool, _vec, flat_logits, speed_params = actor.evaluate_actions(sp, ve, vm, idx, speed)
    assert eval_lp.shape == (2,), f"eval_lp shape mismatch: {eval_lp.shape}"
    assert eval_ent.shape == (2,), f"eval_ent shape mismatch: {eval_ent.shape}"
    
    # Check that rollout logprob formula matches evaluate_actions
    rollout_lp = lp.mean(dim=1) + 0.1 * speed_lp.squeeze(-1)
    diff = torch.abs(rollout_lp - eval_lp).max().item()
    print(f"Log-prob difference between rollout and evaluate_actions: {diff:.6f}")
    assert diff < 1e-4, f"Mismatch between rollout log-prob and evaluate_actions: {diff}"
    print("✓ Actor-critic shapes and log-prob consistency passed!")

def test_run_episode_integration():
    print("Testing end-to-end episode execution...")
    stage = STAGES[0]
    actor = GhostActor()
    actor.eval()
    env = Env(env_id=0, num_ghosts=stage.n_ghosts, world_height=float(stage.rows), world_width=float(stage.cols), obs_resolution=stage.obs_resolution, n_power=stage.n_power)
    
    frames, surviving, pac_score, pacman_caught = _run_episode(actor, env, stage)
    print(f"Episode completed: frames={frames}, surviving_ghosts={surviving}, pac_score={pac_score}, caught={pacman_caught}")
    assert frames > 0, "Episode terminated immediately without frames"
    print("✓ Run episode integration passed!")

if __name__ == "__main__":
    test_curriculum_logic()
    test_actor_critic_shapes_and_logprobs()
    test_run_episode_integration()
    print("\nAll integration tests passed successfully!")
