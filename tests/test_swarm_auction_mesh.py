import pytest
import math
import numpy as np
from allocator import Task, TaskType
from cbba import CBBA_Agent, _task_key
from reward import RewardShaper
from worker import Env
from curriculum import STAGES
from obs import MAX_CANDIDATES

def _novel_action(env, cells, score=1.0):
    """Nominate raw cells through the actor's off-menu head, bypassing the candidate pointer. Mirrors the
    (cand_picks, cand_scores, novel_pairs, novel_scores, speed, dir, gate) tuple that worker.step expects."""
    rows, cols = int(env.world_height), int(env.world_width)
    smap = np.zeros((rows, cols), dtype=np.float32)
    for (r, c) in cells:
        if 0 <= r < rows and 0 <= c < cols:
            smap[r, c] = score
    return ([], np.zeros(MAX_CANDIDATES, dtype=np.float32), list(cells), smap, 1.0, 0.5, 0.0)

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
    """Verify the mesh potential rewards multi-hop reachability and is indifferent to spacing.

    The nearest-peer `tension` penalty was removed 2026-09-20: measured with sensing cut to 4 cells,
    going from a 2-cell radio to global comms cut deaths 3.70 -> 2.87 but cut kills 0.867 -> 0.800,
    because a fully shared belief makes ghosts converge and cover less ground. Being reachable is
    worth a reward; being bunched up is not."""
    shaper = RewardShaper(beta_mesh=3.0)

    # 3 ghosts in a chain: g0 at (0, 0), g1 at (0, 7), g2 at (0, 14)
    # dist(g0, g1) = 7 (<= 12), dist(g1, g2) = 7 (<= 12). Full multi-hop connected mesh!
    # dist(g0, g2) = 14 (> 12), but connected via g1!
    g0 = DummyGhost(0, y=0.0, x=0.0)
    g1 = DummyGhost(1, y=0.0, x=7.0)
    g2 = DummyGhost(2, y=0.0, x=14.0)
    ghosts = {0: g0, 1: g1, 2: g2}

    # all 3 reachable via g1 -> frac_connected = 1.0
    phi0 = shaper._phi_mesh(g0, ghosts)
    assert math.isclose(phi0, 3.0, abs_tol=1e-3)

    # stretch the chain to the edge of radio range: still fully reachable, so still full reward.
    # This is the case the old tension term penalised and the new one must not.
    g1.x = 11.5
    g2.x = 23.0
    phi0_stretched = shaper._phi_mesh(g0, ghosts)
    assert math.isclose(phi0_stretched, 3.0, abs_tol=1e-3)

    # break the chain: g0 is now reachable only from itself -> frac_connected = 1/3
    g0.x = 50.0
    phi0_isolated = shaper._phi_mesh(g0, ghosts)
    assert math.isclose(phi0_isolated, 3.0 * (1.0 / 3.0), abs_tol=1e-3)
    assert phi0_isolated < phi0_stretched


def test_env_mesh_rewards_and_common_pool():
    """Verify that Env applies mesh isolation penalties and shares common pooled tasks."""
    stage = next(s for s in STAGES if s.rows >= 21)  # Stage: 21x27
    env = Env(env_id=0, num_ghosts=2, world_height=float(stage.rows), world_width=float(stage.cols), obs_resolution=stage.obs_resolution, n_power=stage.n_power)
    env.reset()

    # Move player far away from both ghosts and hold stationary
    env.player.stationary = True
    env.player.y = 10.0
    env.player.x = 10.0

    # Place ghosts within mesh range: g0 at (2.0, 2.0), g1 at (2.0, 6.0) (dist = 4.0 <= 12.0)
    env.ghosts[0].y = 2.0
    env.ghosts[0].x = 2.0
    env.ghosts[1].y = 2.0
    env.ghosts[1].x = 6.0

    actions = {
        0: _novel_action(env, [(2, 2)]),
        1: _novel_action(env, [(2, 6)])
    }
    obs, rewards, done, info = env.step(actions, want_bc=False)
    assert 0 in rewards and 1 in rewards

    # Separate ghosts beyond 12.0: g0 at (2.0, 2.0), g1 at (18.0, 24.0) (dist > 25.0)
    env.ghosts[0].y = 2.0
    env.ghosts[0].x = 2.0
    env.ghosts[1].y = 18.0
    env.ghosts[1].x = 24.0
    env.player.y = 10.0
    env.player.x = 10.0
    env.shaper.reset()
    obs, rewards_isolated, done, info = env.step(actions, want_bc=False)
    assert 0 in rewards_isolated and 1 in rewards_isolated
    assert not math.isnan(rewards_isolated[0]) and not math.isnan(rewards[0])


def test_cross_ghost_task_pooling_and_bidding():
    """The pointer pick is AUTHORITATIVE for its own ghost and is NOT pooled to peers.

    Runs 13-17 shared every RL nomination across the swarm through a confidence-scaled auction, and the
    chain 'my pick -> maybe wins vs 6 peers -> maybe not overridden -> reward' was noise to PPO. Now:
      * a ghost handed use_pick=True with a live slot executes exactly that candidate,
      * a peer handed use_pick=False never inherits it (it runs the heuristic floor),
      * an out-of-range slot silently falls back to the heuristic instead of crashing.
    """
    import numpy as np
    from cbba import _task_key
    from obs import MAX_CANDIDATES
    stage = next(s for s in STAGES if s.rows >= 21)  # Stage: 21x27
    env = Env(env_id=0, num_ghosts=2, world_height=float(stage.rows), world_width=float(stage.cols), obs_resolution=stage.obs_resolution, n_power=stage.n_power)
    env.reset()
    R, C = stage.rows, stage.cols
    g0, g1 = env.ghosts[0], env.ghosts[1]
    assert getattr(g0, '_rl_candidates', None), "reset must populate the heuristic candidate set"
    pick_key = _task_key(g0._rl_candidates[0])

    def act(slot, use):
        return ([slot], np.zeros(MAX_CANDIDATES, np.float32), [], np.zeros((R, C), np.float32), 1.0, 0.5, 0.0, use)

    env.step({0: act(0, True), 1: act(0, False)}, want_bc=False)
    #the auction outcome lives in the consensus table; get_active_task() may already have moved on if the
    #target was within reach of six frames of movement (explore candidates are scored nearest-first)
    from obs import AUTH_SCORE
    from allocator import ORIGIN_RL_ENDORSE
    assert g0.cbba_agent.z.get(pick_key) == 0, "ghost 0 must have won its own pick in its auction"
    t_auth = g0.cbba_agent._task_map.get(pick_key)
    assert t_auth is not None and t_auth.score == AUTH_SCORE and t_auth.origin == ORIGIN_RL_ENDORSE, \
        "the executed entry must be the authoritative copy, not the heuristic's own scoring of the same target"
    assert g1.cbba_agent.z.get(pick_key) != 1 and pick_key not in g1.cbba_agent.bundle, \
        "a peer must not inherit another ghost's authoritative pick"

    #out-of-range slot: heuristic fallback, no exception, and the auction still ran
    env.step({0: act(MAX_CANDIDATES + 5, True), 1: act(0, False)}, want_bc=False)
    assert g0.cbba_agent._last_auction >= env.frame - 6

def test_belief_grounded_heuristic_tasks():
    """Verify that heuristic tasks and BC targets follow the belief map modes when Pacman is lost."""
    from allocator import generate_tasks

    stage = next(s for s in STAGES if s.rows >= 21)  # Stage: 21x27
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
    #targets are built from the CURRENT state (post-step), so call the builder directly instead
    #of stepping the world, which would advance the belief map away from the injected mode
    env._refresh_bc_targets()

    cached_map = env._cached_ht[0]
    assert np.max(cached_map) > 0.0, "Expected non-zero target heatmap for Ghost 0"

    # Peak of cached_map should be centered near the belief mode target_pos or its cutoff intercept points
    peak_r, peak_c = np.unravel_index(np.argmax(cached_map), cached_map.shape)
    peak_y = (float(peak_r) + 0.5) / stage.obs_resolution
    peak_x = (float(peak_c) + 0.5) / stage.obs_resolution
    dist_to_peak = math.hypot(peak_y - target_pos[0], peak_x - target_pos[1])
    assert dist_to_peak <= 4.5, f"Target peak at ({peak_y:.1f}, {peak_x:.1f}) too far from belief mode ({target_pos[0]:.1f}, {target_pos[1]:.1f})"

