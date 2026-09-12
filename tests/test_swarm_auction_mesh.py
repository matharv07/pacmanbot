import pytest
import math
import numpy as np
from allocator import Task, TaskType
from cbba import CBBA_Agent, _task_key
from reward import RewardShaper
from worker import Env
from curriculum import STAGES

class DummyWorld:
    def __init__(self, h=20, w=20):
        self.height = h
        self.width = w
    def is_passable(self, x, y, radius=0.35):
        return 0 <= x < self.width and 0 <= y < self.height

class DummyGhost:
    def __init__(self, gid, y=5.0, x=5.0):
        self.gid = gid
        self.y = y
        self.x = x
        self.frame = 10
        self.dead = False
        self.dead_agents = set()
        self.known_agents = {}
        self.last_heartbeat = {gid: 10}
        self.world = DummyWorld()
        self.cbba_agent = CBBA_Agent(gid)
        self.known_power_pellets = set()
        self.known_pacman = None
        self.last_lost_pacman = None
        self.pacman_powered = False
        self.prm_last_seen = {}
        self.prm_known_count = 0

    def is_agent_dead(self, other_gid: int) -> bool:
        if other_gid in self.dead_agents:
            return True
        if other_gid in self.last_heartbeat and (self.frame - self.last_heartbeat[other_gid] > 150):
            self.dead_agents.add(other_gid)
            return True
        return False

    def kill(self):
        self.dead = True
        self.dead_agents.add(self.gid)


def test_dead_winner_orphan_adoption():
    """Verify that when winning agent dies, task is reset to None and surviving ghost adopts it immediately."""
    g0 = DummyGhost(0, y=5.0, x=5.0)
    g1 = DummyGhost(1, y=6.0, x=6.0)
    g0.last_heartbeat[1] = 10
    g1.last_heartbeat[0] = 10

    # Create task at (6.0, 6.0)
    t = Task(task_type=TaskType.HUNT, target_pos=(6.0, 6.0), score=10.0)
    k = _task_key(t)

    # Ghost 0 bids and wins task t
    g0.cbba_agent._phase1(g0, [t], {(6.0, 6.0): (1.4, None)})
    assert g0.cbba_agent.z.get(k) == 0

    # Ghost 1 receives consensus from Ghost 0
    payload0 = g0.cbba_agent.get_consensus_payload()
    g1.cbba_agent.receive_consensus(0, payload0["y"], payload0["z"], payload0["s"], frame=10)
    assert g1.cbba_agent.z.get(k) == 0
    assert k in g1.cbba_agent._task_map

    # Ghost 0 dies!
    g0.kill()
    g1.dead_agents.add(0)

    # Ghost 1 runs step at frame 10 (not auction frame): discovers winner is dead, resets to None
    g1.cbba_agent.step(g1, frame=10)
    assert g1.cbba_agent.z.get(k) is None
    assert g1.cbba_agent.y.get(k) == 0.0

    # In subsequent auction phase (frame 11: (11 + 1) % 6 == 0), surviving ghost adopts it immediately
    g1.cbba_agent.step(g1, frame=11)
    assert g1.cbba_agent.z.get(k) == 1
    assert k in g1.cbba_agent.bundle
    assert g1.cbba_agent.get_active_task() is not None
    assert _task_key(g1.cbba_agent.get_active_task()) == k


def test_unknown_not_prematurely_wiped():
    """Verify that an agent temporarily marked UNKNOWN (e.g. around a corner) does NOT have its tasks wiped."""
    g0 = DummyGhost(0, y=5.0, x=5.0)
    g1 = DummyGhost(1, y=10.0, x=10.0)
    g1.last_heartbeat[0] = 10  # recent heartbeat (frame 10)
    g1.frame = 12

    t = Task(task_type=TaskType.HUNT, target_pos=(5.0, 5.0), score=10.0)
    k = _task_key(t)

    # Ghost 0 won task
    g1.cbba_agent._task_map[k] = t
    g1.cbba_agent._task_created_frame[k] = 10
    g1.cbba_agent.y[k] = 8.5
    g1.cbba_agent.z[k] = 0

    # Ghost 1 temporarily lost LOS to Ghost 0, so known_agents is "UNKNOWN"
    g1.known_agents[0] = "UNKNOWN"

    # Ghost 1 calls step at frame 12 (only 2 frames of silence)
    g1.cbba_agent.step(g1, frame=12)

    # Winner must NOT be reset!
    assert g1.cbba_agent.z.get(k) == 0
    assert g1.cbba_agent.y.get(k) == 8.5


def test_prolonged_silence_orphaned():
    """Verify that if an agent is UNKNOWN for prolonged silence (>100 frames), task is orphaned."""
    g0 = DummyGhost(0, y=5.0, x=5.0)
    g1 = DummyGhost(1, y=10.0, x=10.0)
    g1.last_heartbeat[0] = 10
    g1.frame = 120  # 110 frames of silence!

    t = Task(task_type=TaskType.HUNT, target_pos=(5.0, 5.0), score=10.0)
    k = _task_key(t)

    g1.cbba_agent._task_map[k] = t
    g1.cbba_agent._task_created_frame[k] = 100
    g1.cbba_agent.y[k] = 8.5
    g1.cbba_agent.z[k] = 0
    g1.known_agents[0] = "UNKNOWN"

    g1.cbba_agent.step(g1, frame=120)

    # After prolonged silence, winner should be reset to None
    assert g1.cbba_agent.z.get(k) is None
    assert g1.cbba_agent.y.get(k) == 0.0


def test_extended_mesh_potential():
    """Verify that RewardShaper computes connected component size and applies tension gradient."""
    shaper = RewardShaper(beta_mesh=3.0)

    # 3 ghosts in a chain: g0 at (0, 0), g1 at (0, 7), g2 at (0, 14)
    # dist(g0, g1) = 7 (<= 12), dist(g1, g2) = 7 (<= 12). Full multi-hop connected mesh!
    # dist(g0, g2) = 14 (> 12), but connected via g1!
    g0 = DummyGhost(0, y=0.0, x=0.0)
    g1 = DummyGhost(1, y=0.0, x=7.0)
    g2 = DummyGhost(2, y=0.0, x=14.0)
    ghosts = {0: g0, 1: g1, 2: g2}

    # Since min_dist for g0 is 7.0 (<= 8.0), tension is 0.0. All 3 connected: frac_connected = 1.0
    phi0 = shaper._phi_mesh(g0, ghosts)
    assert math.isclose(phi0, 3.0 * (1.0 - 0.0), abs_tol=1e-3)

    # Move g1 to (0, 10.0) -> min_dist for g0 is 10.0 (in tension window 8..12)
    # tension = ((10.0 - 8.0) / 4.0)**2 = (0.5)**2 = 0.25
    g1.x = 10.0
    phi0_tension = shaper._phi_mesh(g0, ghosts)
    expected_phi = 3.0 * (1.0 - 0.5 * 0.25)
    assert math.isclose(phi0_tension, expected_phi, abs_tol=1e-3)
    assert phi0_tension < phi0

    # Move g0 far away to (0, 50.0) -> isolated!
    g0.x = 50.0
    phi0_isolated = shaper._phi_mesh(g0, ghosts)
    # g0 is only connected to itself: frac_connected = 1/3, tension = 1.0
    assert math.isclose(phi0_isolated, 3.0 * (1.0/3.0 - 0.5 * 1.0), abs_tol=1e-3)
    assert phi0_isolated < 0.0


def test_env_mesh_rewards_and_common_pool():
    """Verify that Env applies mesh isolation penalties and shares common pooled tasks."""
    stage = STAGES[2]  # Stage 2: 21x27
    env = Env(env_id=0, num_ghosts=2, world_height=float(stage.rows), world_width=float(stage.cols), obs_resolution=stage.obs_resolution, n_power=stage.n_power)
    env.reset()

    # Move player far away from both ghosts
    env.player.y = 10.0
    env.player.x = 10.0

    # Place ghosts within mesh range: g0 at (2.0, 2.0), g1 at (2.0, 6.0) (dist = 4.0 <= 12.0)
    env.ghosts[0].y = 2.0
    env.ghosts[0].x = 2.0
    env.ghosts[1].y = 2.0
    env.ghosts[1].x = 6.0

    actions = {
        0: ([(2, 2)], [1.0], 1.0),
        1: ([(2, 6)], [1.0], 1.0)
    }
    obs, rewards, done, info = env.step(actions, bc_prob=0.0)
    assert 0 in rewards and 1 in rewards

    # Separate ghosts beyond 12.0: g0 at (2.0, 2.0), g1 at (18.0, 24.0) (dist > 25.0)
    env.ghosts[0].y = 2.0
    env.ghosts[0].x = 2.0
    env.ghosts[1].y = 18.0
    env.ghosts[1].x = 24.0
    env.player.y = 10.0
    env.player.x = 10.0
    obs, rewards_isolated, done, info = env.step(actions, bc_prob=0.0)
    # In isolated state, ghosts receive isolation penalty (-0.02/frame) vs connected (+0.005/frame)
    assert rewards_isolated[0] < rewards[0]


def test_cross_ghost_task_pooling_and_bidding():
    """Verify that ghosts pool nominated tasks and bid on the ones closest to themselves."""
    stage = STAGES[2]  # Stage 2: 21x27
    env = Env(env_id=0, num_ghosts=2, world_height=float(stage.rows), world_width=float(stage.cols), obs_resolution=stage.obs_resolution, n_power=stage.n_power)
    env.reset()

    # Dynamically pick two passable cells that are 5..8 units apart (within mesh radio range 12.0)
    open_cells = [(r, c) for r in range(stage.rows) for c in range(stage.cols) if env.world.is_passable(float(c) + 0.5, float(r) + 0.5, radius=0.35)]
    r0, c0 = open_cells[10]
    r1, c1 = next((r, c) for r, c in open_cells if 5 <= abs(r - r0) + abs(c - c0) <= 8)

    env.ghosts[0].y = float(r0) + 0.5
    env.ghosts[0].x = float(c0) + 0.5
    env.ghosts[1].y = float(r1) + 0.5
    env.ghosts[1].x = float(c1) + 0.5

    # Place Pacman far away
    env.player.y = 18.0
    env.player.x = 20.0

    # Both ghosts nominate both waypoints into the common pool
    actions = {0: ([(r0, c0), (r1, c1)], [5.0, 5.0], 1.0), 1: ([(r0, c0), (r1, c1)], [5.0, 5.0], 1.0)}
    obs, rewards, done, info = env.step(actions, bc_prob=0.0)

    # Within mesh radio range, consensus resolves:
    # Ghost 0 wins task near itself, Ghost 1 wins task near itself
    t0_pos = (float(r0) + 0.5, float(c0) + 0.5)
    t1_pos = (float(r1) + 0.5, float(c1) + 0.5)

    winner_for_t0 = None
    winner_for_t1 = None
    for (t_type, pos), winner in env.ghosts[0].cbba_agent.z.items():
        if math.hypot(pos[0] - t0_pos[0], pos[1] - t0_pos[1]) < 0.2:
            winner_for_t0 = winner
        if math.hypot(pos[0] - t1_pos[0], pos[1] - t1_pos[1]) < 0.2:
            winner_for_t1 = winner

    assert winner_for_t0 == 0
    assert winner_for_t1 == 1

def test_belief_grounded_heuristic_tasks():
    """Verify that heuristic tasks and BC targets follow the belief map modes when Pacman is lost."""
    from allocator import generate_tasks

    stage = STAGES[2]  # Stage 2: 21x27
    env = Env(env_id=0, num_ghosts=2, world_height=float(stage.rows), world_width=float(stage.cols), obs_resolution=stage.obs_resolution, n_power=stage.n_power)
    env.reset()

    g0 = env.ghosts[0]
    # Ensure Pacman is NOT directly seen or recently lost by Ghost 0
    g0.known_pacman = None
    g0.last_lost_pacman = None
    g0.pacman_powered = False

    # Find a passable target cell in the world for the belief mode
    open_cells = [(r, c) for r in range(stage.rows) for c in range(stage.cols) if env.world.is_passable(float(c) + 0.5, float(r) + 0.5, radius=0.35)]
    target_r, target_c = open_cells[20]
    target_pos = (float(target_r) + 0.5, float(target_c) + 0.5)

    # Initialize and set the belief map mode at target_cell
    g0.belief_map._ensure_initialised()
    g0.belief_map._b_flat[:] = 0.0001
    idx = g0.belief_map._open_idx_map.get((target_r, target_c))
    if idx is None:
        idx = g0.belief_map._closest_node((target_r, target_c))
        target_pos = g0.belief_map._open_cells[idx]
    g0.belief_map._b_flat[idx] = 1.0

    # 1. Verify allocator.generate_tasks directly
    tasks, dists = generate_tasks(g0, frame=env.frame)
    hunt_tasks = [t for t in tasks if t.task_type == TaskType.HUNT]
    assert len(hunt_tasks) > 0, "Expected hunt tasks generated from belief map mode"
    # Verify that at least one hunt task targets the belief mode
    assert any(math.hypot(t.target_pos[0] - target_pos[0], t.target_pos[1] - target_pos[1]) < 2.0 for t in hunt_tasks)

    # 2. Verify worker.py's _cached_ht generates target heatmap peak around belief mode
    actions = {0: ([], [], 1.0), 1: ([], [], 1.0)}
    env.step(actions, bc_prob=1.0)

    cached_map = env._cached_ht[0]
    assert np.max(cached_map) > 0.0, "Expected non-zero target heatmap for Ghost 0"

    # Peak of cached_map should be centered near the belief mode target_pos or its cutoff intercept points
    peak_r, peak_c = np.unravel_index(np.argmax(cached_map), cached_map.shape)
    peak_y = (float(peak_r) + 0.5) / stage.obs_resolution
    peak_x = (float(peak_c) + 0.5) / stage.obs_resolution
    dist_to_peak = math.hypot(peak_y - target_pos[0], peak_x - target_pos[1])
    assert dist_to_peak <= 4.5, f"Target peak at ({peak_y:.1f}, {peak_x:.1f}) too far from belief mode ({target_pos[0]:.1f}, {target_pos[1]:.1f})"

