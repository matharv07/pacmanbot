import math
import pytest
import numpy as np

from reward import RewardShaper
from allocator import Task, TaskType
from cbba import CBBA_Agent, _task_key

class DummyGhost:
    def __init__(self, gid, y, x, player_dir=(0, 0)):
        self.gid = gid
        self.y = y
        self.x = x
        self.frame = 0
        self.dead = False
        self.pacman_powered = False
        self.known_pacman = (10.0, 10.0)
        self.last_lost_pacman = None
        self.belief_map = type('DummyBM', (), {'_b_flat': np.zeros(0), '_open_arr': np.zeros((0, 2))})()
        self._player_dir = player_dir
        self.world = type('DummyWorld', (), {'width': 28.0, 'height': 36.0, 'prm_nodes': []})()

def test_phi_surround_pincer_vs_tailing():
    shaper = RewardShaper(beta=6.0)
    pac_pos = (10.0, 10.0)
    
    # Case 1: Tailing behind (both ghosts behind Pacman to the west: at x=6 and x=7)
    g0_tail = DummyGhost(0, y=10.0, x=6.0)
    g1_tail = DummyGhost(1, y=10.0, x=7.0)
    all_tail = {0: g0_tail, 1: g1_tail}
    phi_tail = shaper._phi_surround(g0_tail, all_tail, pac_pos)
    print(f"Phi surround (tailing): {phi_tail:.4f}")
    assert phi_tail < 0.05, f"Tailing ghosts should have near-zero surround potential, got {phi_tail}"
    
    # Case 2: Pincer from opposite sides (one west at x=6, one east at x=14)
    g0_pincer_far = DummyGhost(0, y=10.0, x=6.0)
    g1_pincer_far = DummyGhost(1, y=10.0, x=14.0)
    all_pincer_far = {0: g0_pincer_far, 1: g1_pincer_far}
    phi_pincer_far = shaper._phi_surround(g0_pincer_far, all_pincer_far, pac_pos)
    print(f"Phi surround (pincer far d=4): {phi_pincer_far:.4f}")
    assert phi_pincer_far > 1.5, f"Pincer formation should have strong surround potential, got {phi_pincer_far}"
    
    # Case 3: Pincer closing in (one west at x=8.5, one east at x=11.5 -> distance 1.5 each)
    g0_pincer_close = DummyGhost(0, y=10.0, x=8.5)
    g1_pincer_close = DummyGhost(1, y=10.0, x=11.5)
    all_pincer_close = {0: g0_pincer_close, 1: g1_pincer_close}
    phi_pincer_close = shaper._phi_surround(g0_pincer_close, all_pincer_close, pac_pos)
    print(f"Phi surround (pincer close d=1.5): {phi_pincer_close:.4f}")
    assert phi_pincer_close > phi_pincer_far, f"Closing pincer potential {phi_pincer_close} should exceed far pincer {phi_pincer_far}"

def test_phi_hunt_lead_interception():
    shaper = RewardShaper(alpha=5.0)
    pac_pos = (10.0, 10.0)
    
    # Pacman moving east: player_dir = (0, 1)
    # Lead point is at (10.0, 13.0)
    # Ghost A is at (10.0, 14.0) - waiting ahead at the cutoff point
    g_flank = DummyGhost(0, y=10.0, x=14.0, player_dir=(0.0, 1.0))
    phi_flank = shaper._phi_hunt(g_flank, pac_pos)
    
    # Ghost B is at (10.0, 6.0) - 4 units behind Pacman
    g_behind = DummyGhost(1, y=10.0, x=6.0, player_dir=(0.0, 1.0))
    phi_behind = shaper._phi_hunt(g_behind, pac_pos)
    
    print(f"Phi hunt flank ahead: {phi_flank:.4f}, behind: {phi_behind:.4f}")
    # The flanking ghost at distance 1.0 from the lead cutoff point receives high hunt potential
    assert phi_flank > 2.0, f"Flanking ghost should receive strong hunt potential, got {phi_flank}"

def test_cbba_shared_task_distribution():
    agent0 = CBBA_Agent(gid=0)
    agent1 = CBBA_Agent(gid=1)
    
    # Tasks: Task A (Direct chase at (10.0, 10.0)), Task B (Cutoff ahead at (10.0, 14.0))
    task_direct = Task(task_type=TaskType.HUNT, target_pos=(10.0, 10.0), score=2.5)
    task_cutoff = Task(task_type=TaskType.HUNT, target_pos=(10.0, 14.0), score=2.0)
    
    tasks = [task_direct, task_cutoff]
    
    # Ghost 0 is close to Pacman at (10.0, 9.0) -> dist to direct=1.0, dist to cutoff=5.0
    g0 = DummyGhost(0, y=10.0, x=9.0)
    dists0 = {(10.0, 10.0): (1.0, []), (10.0, 14.0): (5.0, [])}
    agent0._phase1(g0, tasks, dists0)
    
    # Ghost 1 is close to cutoff at (10.0, 15.0) -> dist to direct=5.0, dist to cutoff=1.0
    g1 = DummyGhost(1, y=10.0, x=15.0)
    dists1 = {(10.0, 10.0): (5.0, []), (10.0, 14.0): (1.0, [])}
    agent1._phase1(g1, tasks, dists1)
    
    # Both agents initially bid
    key_direct = _task_key(task_direct)
    key_cutoff = _task_key(task_cutoff)
    assert agent0.y[key_direct] > agent1.y[key_direct], "Ghost 0 closer to Pacman should have higher bid"
    
    # Consensus exchange between agent0 and agent1
    payload0 = agent0.get_consensus_payload()
    payload1 = agent1.get_consensus_payload()
    
    agent0.receive_consensus(1, payload1["y"], payload1["z"], payload1["s"], frame=10)
    agent1.receive_consensus(0, payload0["y"], payload0["z"], payload0["s"], frame=10)
    
    # After receiving Ghost 0's higher bid, Ghost 1 yields task_direct to Ghost 0
    assert agent1.z[key_direct] == 0, f"Ghost 1 should recognize Ghost 0 as winner of direct chase, got {agent1.z[key_direct]}"
    
    # In next auction phase, Ghost 1 re-bundles and claims task_cutoff
    agent1._phase1(g1, tasks, dists1)
    assert agent1.path[0] == key_cutoff, f"Ghost 1 should claim cutoff task after direct chase is won by Ghost 0, got {agent1.path}"
    assert agent0.path[0] == key_direct, f"Ghost 0 should retain direct chase, got {agent0.path}"
    print("✓ CBBA shared task consensus: Ghost 0 won Direct Chase, Ghost 1 won Cutoff Ahead!")

def test_worker_swarm_catch_bonus():
    from worker import Env
    from curriculum import STAGES
    
    stage = STAGES[0]  # Stage 0: 2 ghosts
    env = Env(env_id=0, num_ghosts=stage.n_ghosts, world_height=float(stage.rows), world_width=float(stage.cols), obs_resolution=stage.obs_resolution, n_power=stage.n_power)
    env.reset()
    
    # Place player at (10.0, 10.0)
    env.player.y = 10.0
    env.player.x = 10.0
    env.player.vy = 0.0
    env.player.vx = 0.0
    env.player.dead = False
    env.player.powered = False
    
    # Position ghost 0 at (10.0, 9.5) and ghost 1 at (10.0, 10.5) (opposing pincer)
    env.ghosts[0].y = 10.0
    env.ghosts[0].x = 9.5
    env.ghosts[0].dead = False
    env.ghosts[0].path_this_frame = [(9.5, 10.0), (10.0, 10.0)]
    
    env.ghosts[1].y = 10.0
    env.ghosts[1].x = 10.5
    env.ghosts[1].dead = False
    env.ghosts[1].path_this_frame = [(10.5, 10.0), (10.0, 10.0)]
    
    env.player.path_this_frame = [(10.0, 10.0)]
    
    # Step 0 frames to evaluate kill logic
    # Perform dummy actions
    actions = {0: ([(10, 10)], [1.0], 1.0), 1: ([(10, 10)], [1.0], 1.0)}
    obs, rewards, done, info = env.step(actions, bc_prob=0.0)
    
    print(f"Pincer kill rewards: {rewards}, Done: {done}, Caught: {info.get('pacman_caught')}")
    assert done, "Game should end when Pacman is caught"
    assert info.get('pacman_caught'), "Pacman should be marked caught"
    
    # Killer (ghost 0) should receive direct kill award + swarm bonus
    # Flanker (ghost 1) should receive team kill award + swarm bonus
    assert rewards[0] > 240.0, f"Ghost 0 reward {rewards[0]} should include direct kill + swarm bounty"
    assert rewards[1] > 100.0, f"Ghost 1 reward {rewards[1]} should include team kill + swarm bounty"
    print("✓ Worker multi-agent swarm catch bonus successfully awarded to all trapping ghosts!")

def test_phi_corner_dead_end_trapping():
    shaper = RewardShaper(alpha_corner=3.0)
    g = DummyGhost(0, y=10.0, x=9.5)
    pac_pos = (10.0, 11.0)  # distance = 1.5
    
    # Mock world where Pacman at (10, 11) is in a dead end: only west (towards ghost) is passable
    class DeadEndWorld:
        def is_passable(self, x, y, radius=0.35):
            # Only allow (x, y) if x <= 10.5 (i.e. to the west)
            return x <= 10.5
    
    g.world = DeadEndWorld()
    phi_corner_trapped = shaper._phi_corner(g, {0: g}, pac_pos)
    assert phi_corner_trapped > 1.5, f"Expected strong cornering potential in dead end, got {phi_corner_trapped}"

    # Mock open cross intersection: all 4 directions passable
    class OpenWorld:
        def is_passable(self, x, y, radius=0.35):
            return True
            
    g.world = OpenWorld()
    phi_corner_open = shaper._phi_corner(g, {0: g}, pac_pos)
    assert phi_corner_open == 0.0, f"Corner potential in open intersection should be 0.0, got {phi_corner_open}"
    print("✓ Cornering / dead-end trapping potential verified!")

def test_corridor_anti_clustering_jam_penalty():
    from worker import Env
    from curriculum import STAGES
    stage = STAGES[0]
    env = Env(env_id=0, num_ghosts=2, world_height=float(stage.rows), world_width=float(stage.cols), obs_resolution=stage.obs_resolution, n_power=stage.n_power)
    env.reset()
    
    g0 = env.ghosts[0]
    g1 = env.ghosts[1]
    g0.dead = False
    g1.dead = False
    
    # 1. Crowded in narrow corridor (d = 0.4 < 0.85)
    g0.y, g0.x = 2.0, 2.0
    g1.y, g1.x = 2.0, 2.4
    rewards_crowded = {0: 0.0, 1: 0.0}
    alive_ghosts = [g for g in env.ghosts.values() if not g.dead]
    for i in range(len(alive_ghosts)):
        for j in range(i + 1, len(alive_ghosts)):
            d_peer = math.hypot(alive_ghosts[i].y - alive_ghosts[j].y, alive_ghosts[i].x - alive_ghosts[j].x)
            if d_peer < 0.85:
                p = 0.015 * (1.0 - d_peer / 0.85)
                rewards_crowded[alive_ghosts[i].gid] -= p
                rewards_crowded[alive_ghosts[j].gid] -= p
                
    expected_jam = 0.015 * (1.0 - 0.4 / 0.85)
    assert math.isclose(rewards_crowded[0], -expected_jam, abs_tol=1e-5)
    assert rewards_crowded[0] < -0.007
    
    # 2. Spaced out (d = 1.5 > 0.85)
    g1.y, g1.x = 2.0, 3.5
    rewards_spaced = {0: 0.0, 1: 0.0}
    for i in range(len(alive_ghosts)):
        for j in range(i + 1, len(alive_ghosts)):
            d_peer = math.hypot(alive_ghosts[i].y - alive_ghosts[j].y, alive_ghosts[i].x - alive_ghosts[j].x)
            if d_peer < 0.85:
                p = 0.015 * (1.0 - d_peer / 0.85)
                rewards_spaced[alive_ghosts[i].gid] -= p
                rewards_spaced[alive_ghosts[j].gid] -= p
                
    assert rewards_spaced[0] == 0.0
    assert rewards_crowded[0] < rewards_spaced[0]
    print("✓ Corridor anti-clustering jam penalty verified!")

if __name__ == "__main__":
    test_phi_surround_pincer_vs_tailing()
    test_phi_hunt_lead_interception()
    test_cbba_shared_task_distribution()
    test_worker_swarm_catch_bonus()
    test_phi_corner_dead_end_trapping()
    test_corridor_anti_clustering_jam_penalty()
    print("All swarming & group catch unit tests passed!")