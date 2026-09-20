import os
import sys
import torch
import numpy as np

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
    assert len(STAGES) == 4, f"Expected 4 curriculum stages, got {len(STAGES)}"

    # Test that partial/None updates do not corrupt the rolling window
    cs.record_return(mean_return=None, kill_rate=None)
    cs.record_return(mean_return=25.0, kill_rate=None)
    cs.record_return(mean_return=None, kill_rate=0.8)
    assert len(cs._return_history) == 0
    assert len(cs._kill_history) == 0
    assert cs._updates_in_stage == 3

    # Test that min_updates cannot be bypassed even with 95% kill rate
    for _ in range(40):
        cs.record_return(mean_return=80.0, kill_rate=0.95)
    assert len(cs._return_history) == 40
    assert cs._updates_in_stage == 43
    assert not cs.should_advance(), f"Curriculum must NOT advance before min_updates ({cs.stage.min_updates}) is reached!"

    # Reach min_updates with high performance
    for _ in range(80):
        cs.record_return(mean_return=75.0, kill_rate=0.90)
    assert cs._updates_in_stage >= cs.stage.min_updates
    assert cs.should_advance(), "Curriculum should advance once min_updates is reached with high kill rate"
    cs.advance()
    assert cs.stage_idx == 1, f"Expected Stage 1, got {cs.stage_idx}"
    assert not cs.is_final, "Stage 1 is intermediate"

    # Fast forward through remaining intermediate stages to final stage
    while not cs.is_final:
        for _ in range(cs.stage.min_updates):
            cs.record_return(mean_return=50.0, kill_rate=0.85)
        assert cs.should_advance()
        cs.advance()
    assert cs.stage_idx == len(STAGES) - 1, f"Expected final stage {len(STAGES) - 1}, got {cs.stage_idx}"
    assert cs.is_final, "Final stage should be terminal"
    assert not cs.should_advance(), "Terminal stage should never advance"

    # Test state_dict recovery and clamping
    cs_load = CurriculumScheduler(start_stage=0)
    corrupted_state = {
        "stage_idx": 10,  # out-of-bounds stage index
        "updates_in_stage": 40,
        "return_history": [20.0] * 30,
        "kill_history": [0.7] * 40
    }
    cs_load.load_state_dict(corrupted_state)
    assert cs_load.stage_idx == len(STAGES) - 1, "Out-of-bounds stage index should be clamped to terminal stage"
    assert len(cs_load._return_history) == 30
    assert len(cs_load._kill_history) == 30
    print("✓ Curriculum test passed!")

def test_actor_critic_shapes_and_logprobs():
    print("Testing actor-critic forward and evaluate_actions...")
    actor = GhostActor()
    critic = GhostCritic()
    H, W = 22, 27
    sp = torch.randn(2, SPATIAL_CH, H, W)
    ve = torch.randn(2, VEC_DIM)
    vm = torch.ones(2, H, W, dtype=torch.bool)
    idx, lp, scores, pool, vec, speed, speed_lp, direction, dir_lp, gate, gate_lp = actor(sp, ve, vm, K=3)
    assert idx.shape == (2, 3), f"idx shape mismatch: {idx.shape}"
    assert lp.shape == (2, 3), f"lp shape mismatch: {lp.shape}"
    assert speed.shape == (2, 1), f"speed shape mismatch: {speed.shape}"
    assert speed_lp.shape == (2, 1), f"speed_lp shape mismatch: {speed_lp.shape}"
    assert direction.shape == (2, 1), f"direction shape mismatch: {direction.shape}"
    assert dir_lp.shape == (2, 1), f"dir_lp shape mismatch: {dir_lp.shape}"
    assert gate.shape == (2, 1), f"gate shape mismatch: {gate.shape}"
    assert gate_lp.shape == (2, 1), f"gate_lp shape mismatch: {gate_lp.shape}"
    eval_lp, eval_ent, _pool, _vec, flat_logits, speed_params, _cell_ent = actor.evaluate_actions(sp, ve, vm, idx, speed, direction, gate)
    assert eval_lp.shape == (2, 6), f"eval_lp shape mismatch: {eval_lp.shape}"
    assert eval_ent.shape == (2,), f"eval_ent shape mismatch: {eval_ent.shape}"
    rollout_lp = torch.cat([lp, speed_lp, dir_lp, gate_lp], dim=1)
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

def test_predictor_sequence_and_env_sync():
    print("Testing MovementPredictor sequence BPTT and worker weight sync...")
    from net import MovementPredictor, PREDICTOR_IN_DIM, PREDICTOR_HIDDEN_DIM
    predictor = MovementPredictor(in_dim=PREDICTOR_IN_DIM, hidden_dim=PREDICTOR_HIDDEN_DIM)
    B, T = 4, 8
    x_seq = torch.randn(B, T, PREDICTOR_IN_DIM)
    base_v = torch.randn(B, T, 2)
    gt_v = torch.randn(B, T, 2)
    pred_seq, final_h = predictor.forward_sequence(x_seq, base_vel_seq=base_v)
    assert pred_seq.shape == (B, T, 2), f"Expected (B, T, 2), got {pred_seq.shape}"
    assert final_h.shape == (B, PREDICTOR_HIDDEN_DIM)
    loss = torch.nn.functional.smooth_l1_loss(pred_seq, gt_v)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in predictor.parameters())
    # Test Env synchronization
    env = Env(env_id=0, num_ghosts=3, world_height=15.0, world_width=15.0, n_power=2)
    env.reset()
    # Modify a parameter in predictor
    with torch.no_grad():
        for p in predictor.parameters():
            p.add_(0.5)
    env.sync_predictor(predictor.state_dict())
    for g in env.ghosts.values():
        for p_env, p_main in zip(g.belief_map.predictor.parameters(), predictor.parameters()):
            assert torch.allclose(p_env, p_main)
    print("✓ Predictor sequence BPTT and env sync passed!")

if __name__ == "__main__":
    test_curriculum_logic()
    test_actor_critic_shapes_and_logprobs()
    test_run_episode_integration()
    test_predictor_sequence_and_env_sync()
    print("\nAll integration tests passed successfully!")