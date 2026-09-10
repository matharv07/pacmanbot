import os
import sys
import math
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from worker import Env
from net import MovementPredictor, PREDICTOR_IN_DIM, PREDICTOR_HIDDEN_DIM
from beliefmap import BeliefMap, extract_movement_features
from pathfinder import astar_belief


def test_no_world_reference():
    print("Testing that BeliefMap has no self.world attribute (blind SLAM)...")
    bm = BeliefMap(0, rows=30, cols=30)
    assert not hasattr(bm, 'world'), "BeliefMap still has self.world attribute!"

    class DummyWorld:
        height = 25
        width = 35

    bm2 = BeliefMap(1, DummyWorld())
    assert bm2.rows == 25 and bm2.cols == 35, f"Expected rows=25, cols=35, got {bm2.rows}, {bm2.cols}"
    assert not hasattr(bm2, 'world'), "BeliefMap stored self.world when passed world object!"
    print("✓ BeliefMap is verified 100% blind with no world reference!")


def test_blind_topology_initialization():
    print("Testing blind topology initialization (dense UNKNOWN hypothesis grid)...")
    bm = BeliefMap(0, rows=20, cols=20)
    bm.init_full_topology()
    assert bm.n_nodes > 0, "Belief map has 0 nodes"
    assert bm._open_arr.shape[0] == bm.n_nodes, "Node array count mismatch"
    assert np.all(bm._walkable_mask), "Topology did not initialize as 100% walkable hypothesis"
    assert len(bm._disabled_wall_nodes) == 0, "Discovered walls exist prior to sensor sweep"
    assert abs(float(bm._b_flat.sum()) - 1.0) < 1e-4, "Initial belief does not sum to 1.0"
    print(f"✓ Dense blind grid initialized: {bm.n_nodes} nodes, all marked UNKNOWN and walkable!")


def test_feature_extraction_and_predictor_training():
    print("Testing feature extraction and MovementPredictor forward/backward...")
    env = Env(env_id=0, num_ghosts=3, world_height=22.0, world_width=27.0, n_power=2)
    env.reset()
    pac_pos = (float(env.player.y), float(env.player.x))
    cur_v = (0.5, -0.5)
    prev_v = (0.0, 0.0)
    ghost_positions = [(g.y, g.x) for g in env.ghosts.values()]
    pellets = env.world.pellets
    feats = extract_movement_features(pacman_pos=pac_pos, current_vel=cur_v, prev_vel=prev_v, known_walls=set(), 
    map_width=env.world.width, map_height=env.world.height,known_ghosts=ghost_positions, known_pellets=pellets, is_powered=False)
    assert feats.shape == (PREDICTOR_IN_DIM,), f"Expected shape ({PREDICTOR_IN_DIM},), got {feats.shape}"
    assert np.all(np.isfinite(feats)), "Features contain NaN or Inf"
    assert np.all(feats[4:12] >= 0.0) and np.all(feats[4:12] <= 1.0), "Clearances out of [0, 1] range"
    predictor = MovementPredictor(in_dim=PREDICTOR_IN_DIM, hidden_dim=PREDICTOR_HIDDEN_DIM)
    x = torch.from_numpy(feats).unsqueeze(0)
    hx = torch.zeros(1, PREDICTOR_HIDDEN_DIM)
    base_v = torch.tensor([[cur_v[0], cur_v[1]]], dtype=torch.float32)
    pred_v, new_hx = predictor(x, hx, base_vel=base_v)
    assert pred_v.shape == (1, 2), f"Expected pred_v shape (1, 2), got {pred_v.shape}"
    assert new_hx.shape == (1, PREDICTOR_HIDDEN_DIM), f"Expected hx shape (1, {PREDICTOR_HIDDEN_DIM}), got {new_hx.shape}"
    target_v = torch.tensor([[0.7, -0.3]], dtype=torch.float32)
    l1 = F.smooth_l1_loss(pred_v, target_v)
    cos = F.cosine_similarity(pred_v, target_v, dim=-1)
    loss = l1 + 0.3 * (1.0 - cos.mean())
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in predictor.parameters())
    assert has_grad, "Predictor parameters received no gradients"
    print("✓ Feature extraction and MovementPredictor forward/backward passed!")

def test_wall_discovery_and_dead_zones():
    print("Testing online wall discovery, node dead-zoning, and edge severing...")
    bm = BeliefMap(0, rows=15, cols=15)
    bm.init_full_topology()
    wall_pt = (7.2, 7.2)
    wall_node_idx = bm._closest_node(wall_pt)
    wall_node = bm._open_cells[wall_node_idx]
    orig_nbrs = list(bm._neighbours.get(wall_node, []))
    assert len(orig_nbrs) > 0, "Wall node had no initial neighbours"
    bm.observe_wall(wall_pt)
    assert wall_node in bm._disabled_wall_nodes, "Wall node not registered in _disabled_wall_nodes"
    assert not bm._walkable_mask[wall_node_idx], "Walkable mask was not cleared for wall node"
    assert bm._b_flat[wall_node_idx] == 0.0, "Probability at wall node was not set to 0.0"
    assert len(bm._neighbours.get(wall_node, [])) == 0, "Wall node outgoing edges were not severed"
    for nbr in orig_nbrs:
        assert wall_node not in bm._neighbours.get(nbr, []), f"Neighbour {nbr} still had edge to severed wall node"
    print("✓ Wall discovery correctly dead-zoned node, severed bidirectional edges, and cleared probability!")


def test_multi_path_redistribution():
    print("Testing multi-path trapped probability redistribution (Path A > Path C > Path B)...")
    bm = BeliefMap(0, rows=15, cols=15)
    bm.init_full_topology()
    center_idx = bm._closest_node((7.0, 7.0))
    center_node = bm._open_cells[center_idx]
    valid_nbrs = list(bm._neighbours.get(center_node, []))
    assert len(valid_nbrs) >= 3, f"Need at least 3 neighbours for T-junction test, got {len(valid_nbrs)}"
    path_A = valid_nbrs[0]
    path_B = valid_nbrs[1]
    path_C = valid_nbrs[2]
    bm._b_flat.fill(0.0)
    bm._b_flat[center_idx] = 1.0
    bm.observe_wall(center_node, ghost_positions=[path_B], known_pellets=[(path_A[1], path_A[0])], is_powered=False)
    idx_A = bm._open_idx_map[path_A]
    idx_B = bm._open_idx_map[path_B]
    idx_C = bm._open_idx_map[path_C]
    prob_A = bm._b_flat[idx_A]
    prob_B = bm._b_flat[idx_B]
    prob_C = bm._b_flat[idx_C]
    print(f"Redistributed probabilities: P(A_pellet)={prob_A:.4f}, P(C_neutral)={prob_C:.4f}, P(B_ghost)={prob_B:.4f}")
    assert prob_A > prob_C, f"Expected P(Path A) > P(Path C), got {prob_A} <= {prob_C}"
    assert prob_C > prob_B, f"Expected P(Path C) > P(Path B), got {prob_C} <= {prob_B}"
    total_prob = float(bm._b_flat.sum())
    assert abs(total_prob - 1.0) < 1e-4, f"Total belief deviated from 1.0: {total_prob}"
    print("✓ Multi-path redistribution verifies: Path A (attractor) > Path C (neutral) > Path B (repulsor)!")


def test_power_pellet_polarity_inversion():
    print("Testing power pellet polarity and threat inversion when powered...")
    bm_unpowered = BeliefMap(0, rows=15, cols=15)
    bm_unpowered.init_full_topology()
    center_idx = bm_unpowered._closest_node((7.0, 7.0))
    center_node = bm_unpowered._open_cells[center_idx]
    nbrs = list(bm_unpowered._neighbours.get(center_node, []))
    path_g = nbrs[0]
    path_neutral = nbrs[1]
    bm_unpowered._b_flat.fill(0.0)
    bm_unpowered._b_flat[center_idx] = 1.0
    bm_unpowered.observe_wall(center_node, ghost_positions=[path_g], is_powered=False)
    prob_g_unpowered = bm_unpowered._b_flat[bm_unpowered._open_idx_map[path_g]]
    prob_neutral_unpowered = bm_unpowered._b_flat[bm_unpowered._open_idx_map[path_neutral]]
    assert prob_neutral_unpowered > prob_g_unpowered, "Ghost path was not repelled when unpowered"
    bm_powered = BeliefMap(0, rows=15, cols=15)
    bm_powered.init_full_topology()
    bm_powered._b_flat.fill(0.0)
    bm_powered._b_flat[center_idx] = 1.0
    bm_powered.observe_wall(center_node, ghost_positions=[path_g], is_powered=True)
    prob_g_powered = bm_powered._b_flat[bm_powered._open_idx_map[path_g]]
    prob_neutral_powered = bm_powered._b_flat[bm_powered._open_idx_map[path_neutral]]
    assert prob_g_powered > prob_neutral_powered, f"Hunt mode failed: ghost path prob {prob_g_powered} <= neutral {prob_neutral_powered}"
    print("✓ Power pellet polarity correctly inverts ghost threat into hunting attractor!")

def test_peer_wall_sync():
    print("Testing peer-to-peer wall synchronization and belief edge severing...")
    env = Env(env_id=0, num_ghosts=2, world_height=20.0, world_width=20.0, n_power=2)
    env.reset()
    g0 = env.ghosts[0]
    g1 = env.ghosts[1]
    wall_to_discover = (10.0, 10.0)
    wall_diff = ("wall", wall_to_discover)
    g1.message_queue.append({"id": ("test_wall_msg", 0), "diffs": [wall_diff], "hop": 0})
    g1._process_messages(env.ghosts)
    assert wall_to_discover in g1.lidar_memory, "Ghost 1 failed to add peer wall to lidar_memory"
    g1_wall_node_idx = g1.belief_map._closest_node(wall_to_discover)
    g1_wall_node = g1.belief_map._open_cells[g1_wall_node_idx]
    assert g1_wall_node in g1.belief_map._disabled_wall_nodes, "Ghost 1 failed to register peer wall in belief dead zones"
    print("✓ Peer wall message successfully propagated to receiver's belief map and severed edges!")

def test_astar_belief_pathfinding():
    print("Testing ghost-local A* on personal discovered belief topology...")
    bm = BeliefMap(0, rows=15, cols=15)
    bm.init_full_topology()
    start = (2.0, 2.0)
    goal = (12.0, 12.0)
    path = astar_belief(bm, start, goal)
    assert len(path) >= 2, f"Failed to find open path: {path}"
    assert path[0] == start, f"Path start {path[0]} != {start}"
    for c in range(0, 12):
        bm.observe_wall((7.0, float(c)))
    detour_path = astar_belief(bm, start, goal)
    assert len(detour_path) >= 2, "A* failed to find detour path around wall barrier"
    for wp in detour_path:
        closest_idx = bm._closest_node(wp)
        node = bm._open_cells[closest_idx]
        assert node not in bm._disabled_wall_nodes, f"Detour waypoint {wp} traverses dead-zoned wall {node}"
    print(f"✓ A* found valid detour of {len(detour_path)} waypoints bypassing discovered barrier!")


def test_belief_diffusion_and_speed_bound():
    print("Testing belief diffusion and physical reachability...")
    env = Env(env_id=0, num_ghosts=3, world_height=33.0, world_width=41.0, n_power=4)
    env.reset()
    bm = env.ghosts[0].belief_map
    start_pos = bm._open_cells[10]
    bm.observe(start_pos, (1.0, 0.0))
    assert bm.probability_at(start_pos) > 0.5, "Belief was not placed at observed node"
    for _ in range(5):
        bm.diffuse(ghost_pos=(float(env.ghosts[0].y), float(env.ghosts[0].x)), 
                known_pellets=env.ghosts[0].known_pellets, known_power=env.ghosts[0].known_power_pellets)
    total_prob = float(bm._b_flat.sum())
    assert abs(total_prob - 1.0) < 1e-4, f"Belief sum {total_prob} deviated from 1.0"
    active_mask = bm._b_flat > 1e-5
    active_nodes = bm._open_arr[active_mask]
    dists = np.hypot(active_nodes[:, 0] - start_pos[0], active_nodes[:, 1] - start_pos[1])
    max_travel = np.max(dists) if len(dists) > 0 else 0.0
    print(f"Max reachability distance after 5 diffuse calls: {max_travel:.2f} units (theoretical physical max ~15.0 units)")
    assert max_travel <= 16.0, f"Probability spread too fast ({max_travel:.2f} units > 16.0 units), violating physical speed limit!"
    print("✓ Belief diffusion respects continuous speed limit!")


def test_belief_multi_agent_merge():
    print("Testing multi-agent belief merging and payload...")
    env = Env(env_id=0, num_ghosts=3, world_height=33.0, world_width=41.0, n_power=4)
    env.reset()
    g0 = env.ghosts[0]
    g1 = env.ghosts[1]
    target_node = g0.belief_map._open_cells[5]
    g0.belief_map.observe(target_node, (0.0, 1.0))
    payload = g0.belief_map.get_payload()
    assert "cells" in payload, "Payload missing cells"
    assert "pred_dir" in payload, "Payload missing pred_dir"
    assert "hx" in payload, "Payload missing hx"
    assert len(payload["hx"]) == PREDICTOR_HIDDEN_DIM, f"Payload hx dim mismatch: {len(payload['hx'])}"
    initial_prob = g1.belief_map.probability_at(target_node)
    g1.belief_map.merge(sender_gid=0, payload=payload, frame=10)
    merged_prob = g1.belief_map.probability_at(target_node)
    assert merged_prob > initial_prob, f"Merge failed to increase probability at target node ({initial_prob} -> {merged_prob})"
    total_prob = float(g1.belief_map._b_flat.sum())
    assert abs(total_prob - 1.0) < 1e-4, f"Merged belief sum {total_prob} != 1.0"
    print("✓ Multi-agent payload and merge passed!")


if __name__ == "__main__":
    test_no_world_reference()
    test_blind_topology_initialization()
    test_feature_extraction_and_predictor_training()
    test_wall_discovery_and_dead_zones()
    test_multi_path_redistribution()
    test_power_pellet_polarity_inversion()
    test_peer_wall_sync()
    test_astar_belief_pathfinding()
    test_belief_diffusion_and_speed_bound()
    test_belief_multi_agent_merge()
    print("\nAll BeliefMap structural & blind SLAM tests passed successfully!")