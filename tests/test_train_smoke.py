import os
import sys
import torch
import numpy as np
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from curriculum import STAGES, CurriculumScheduler
from net import GhostActor, GhostCritic
from worker import Env
from obs import flatten_cand_cells
from reward import RewardShaper

def test_single_env_rollout_and_step():
    print("Testing single env rollout and step dynamics...")
    stage = STAGES[0]
    actor = GhostActor()
    critic = GhostCritic()
    env = Env(env_id=0, num_ghosts=stage.n_ghosts, world_height=float(stage.rows), world_width=float(stage.cols), obs_resolution=stage.obs_resolution, n_power=stage.n_power)
    obs = env.reset()
    assert obs is not None
    gids, sp, ve, vm, ht, hs, cf, cc, cm, cbc, global_sp, grid_shape = obs
    assert len(gids) == stage.n_ghosts
    assert cm.any(), "the heuristic floor must offer the actor at least one candidate to arbitrate over"
    t_sp = torch.from_numpy(sp)
    t_ve = torch.from_numpy(ve)
    t_vm = torch.from_numpy(vm)
    t_cf = torch.from_numpy(cf)
    t_cc = torch.from_numpy(flatten_cand_cells(cc, stage.cols))
    t_cm = torch.from_numpy(cm)
    with torch.no_grad():
        (idx, lp, scores, nidx, nlp, nsc, pool, vec,
         speed, speed_lp, direction, dir_lp, gate, gate_lp, cand_logits) = actor(t_sp, t_ve, t_vm, t_cf, t_cc, t_cm)
    action_dict = {}
    for i, gid in enumerate(gids):
        novel_pairs = [(int(x // stage.cols), int(x % stage.cols)) for x in nidx[i].numpy()]
        action_dict[gid] = ([int(x) for x in idx[i].numpy()], scores[i].numpy(), novel_pairs, nsc[i].numpy(),
                            float(speed[i].item()), float(direction[i].item()), float(gate[i].item()))
    obs, rewards, done, info = env.step(action_dict, want_bc=False)
    print(f"Step successful. Rewards: {rewards}, Done: {done}, Info: {info}")
    assert "pacman_caught" in info
    assert "pacman_score" in info
    print("✓ Single env rollout and step test passed!")

def test_checkpoint_save_and_eval():
    print("Testing checkpoint save and test.py _worker_chunk...")
    stage = STAGES[0]
    actor = GhostActor()
    critic = GhostCritic()
    cs = CurriculumScheduler(start_stage=0)
    os.makedirs("test_ckpt", exist_ok=True)
    ckpt_path = "test_ckpt/test_model.pt"
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "curriculum": cs.state_dict(), "update": 1}, ckpt_path)
    from test import _worker_chunk, _aggregate
    res_path, s_idx, stg, chunk_results = _worker_chunk(ckpt_path, n_games=2, stage_override=0, seed_offset=42)
    agg = _aggregate(chunk_results)
    print(f"Evaluation aggregate results on 2 games: {agg}")
    assert agg["n"] == 2
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    if os.path.exists("test_ckpt"):
        os.rmdir("test_ckpt")
    print("✓ Checkpoint evaluation test passed!")

if __name__ == "__main__":
    test_single_env_rollout_and_step()
    test_checkpoint_save_and_eval()
    print("\nAll smoke tests passed!")
