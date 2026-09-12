import pytest
import math
import numpy as np
from allocator import Task, TaskType, _find_flee_pos, _score_evade_track, generate_tasks
from cbba import CBBA_Agent, _task_key
from ghost import Ghost
from worker import Env
from curriculum import STAGES

class DummyWorld:
    def __init__(self, h=20, w=20):
        self.height = h
        self.width = w
    def is_passable(self, x, y, radius=0.35):
        return 0 <= x < self.width and 0 <= y < self.height

class DummyGhost:
    def __init__(self, gid=0, y=10.0, x=10.0):
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
        self.vx = 0.0
        self.vy = 0.0
        self.radius = 0.35

    def is_agent_dead(self, other_gid: int) -> bool:
        return other_gid in self.dead_agents


def test_powered_evasion_task_generation():
    """Verify allocator never targets Pacman when powered and produces high-priority flee tasks."""
    g = DummyGhost(0, y=10.0, x=10.0)
    g.pacman_powered = True
    g.known_pacman = (10.0, 14.0)  # Pacman is 4 units away along x-axis

    # 1. Test _find_flee_pos picks a safe corner/node away from Pacman
    flee_p = _find_flee_pos(g, g.known_pacman)
    assert flee_p is not None
    # Distance from flee_p to Pacman must be greater than current ghost distance to Pacman (4.0)
    d_flee_pac = math.hypot(flee_p[0] - 10.0, flee_p[1] - 14.0)
    assert d_flee_pac > 4.0, f"Flee position {flee_p} must be further from Pacman than ghost (4.0)"

    # 2. Test _score_evade_track never targets Pacman
    dists = {g.known_pacman: (4.0, None)}
    task = _score_evade_track(g, dists, frame=10)
    assert task is not None
    assert task.task_type == TaskType.EVADE_TRACK
    assert task.target_pos != g.known_pacman, "EVADE_TRACK task must NEVER target Pacman's position"
    assert task.target_pos == flee_p
    assert task.score >= 8.0, "Emergency evade task score must be high priority"


def test_cbba_purges_hunt_tasks_when_powered():
    """Verify CBBA immediately drops HUNT tasks from bundle and path when Pacman is powered."""
    g = DummyGhost(0, y=5.0, x=5.0)
    g.pacman_powered = False

    t_hunt = Task(task_type=TaskType.HUNT, target_pos=(10.0, 10.0), score=10.0)
    k_hunt = _task_key(t_hunt)

    # Ghost bids and wins hunt task
    g.cbba_agent._phase1(g, [t_hunt], {(10.0, 10.0): (7.0, None)})
    assert k_hunt in g.cbba_agent.bundle
    assert k_hunt in g.cbba_agent.path
    assert g.cbba_agent.z[k_hunt] == 0

    # Pacman becomes powered!
    g.pacman_powered = True
    active_task = g.cbba_agent.step(g, frame=11)

    # HUNT task must be purged
    assert k_hunt not in g.cbba_agent.bundle
    assert k_hunt not in g.cbba_agent.path
    assert g.cbba_agent.z.get(k_hunt) is None
    assert active_task is None or active_task.task_type != TaskType.HUNT


def test_safety_map_projects_danger_around_powered_pacman():
    """Verify beliefmap.update_safety_map assigns near-zero safety around powered Pacman."""
    from beliefmap import BeliefMap

    bm = BeliefMap(0, rows=20, cols=20)
    bm.init_full_topology()

    pac_pos = (10.0, 10.0)
    # Update safety map with powered=True
    bm.update_safety_map({}, current_frame=10, powered=True, pacman_pos=pac_pos)

    # Safety near Pacman should be low (< 0.5)
    pac_idx = bm._closest_node(pac_pos)
    assert bm._safety[pac_idx] < 0.5, f"Safety near powered Pacman should be < 0.5, got {bm._safety[pac_idx]}"

    # Safety far from Pacman (e.g. corner (1.0, 1.0)) should be high (> 0.8)
    corner_idx = bm._closest_node((1.0, 1.0))
    assert bm._safety[corner_idx] > 0.8, f"Safety far from powered Pacman should be > 0.8, got {bm._safety[corner_idx]}"


def test_ghost_terminal_evasion_in_env():
    """Verify that in the real environment, ghosts close to powered Pacman actively steer away."""
    stage = STAGES[2]
    env = Env(env_id=0, num_ghosts=2, world_height=float(stage.rows), world_width=float(stage.cols), obs_resolution=stage.obs_resolution, n_power=stage.n_power)
    env.reset()

    # Place Pacman and Ghost 0 near each other in an open area
    open_cells = [(r, c) for r in range(stage.rows) for c in range(stage.cols) if env.world.is_passable(float(c) + 0.5, float(r) + 0.5, radius=0.35)]
    cr, cc = open_cells[30]
    
    # Place Ghost at (cr, cc) and Pacman 3 units to the right
    env.ghosts[0].y = float(cr) + 0.5
    env.ghosts[0].x = float(cc) + 0.5
    env.player.y = env.ghosts[0].y
    env.player.x = env.ghosts[0].x + 3.0
    env.player.powered = True
    env.ghosts[0].pacman_powered = True
    env.ghosts[0].known_pacman = (env.player.y, env.player.x)

    # Call ghost update
    init_ghost_x = env.ghosts[0].x
    pac_x = env.player.x
    env.ghosts[0].update((env.player.y, env.player.x), True, env.ghosts)

    # Ghost should move WEST (decreasing x, away from Pacman who is to the EAST at +3.0)
    # Velocity x must be negative (or x decreases), moving away from Pacman
    assert env.ghosts[0].vx < 0.0 or env.ghosts[0].x < init_ghost_x, f"Ghost should steer away from powered Pacman (vx={env.ghosts[0].vx})"


def test_dead_callout_and_line_of_sight_witnessing():
    """Verify that ghosts only learn of peer deaths via direct line-of-sight or radio mesh,
    and surviving witnesses display a 'Ghost X DOWN!' dead callout."""
    stage = STAGES[2]
    env = Env(env_id=0, num_ghosts=3, world_height=float(stage.rows), world_width=float(stage.cols), obs_resolution=stage.obs_resolution, n_power=stage.n_power)
    env.reset()

    # Place Ghost 0 and Ghost 1 in the same open corridor with direct LOS
    g0 = env.ghosts[0]
    g1 = env.ghosts[1]
    g2 = env.ghosts[2]

    # Find two passable positions with mutual line of sight
    open_cells = [(r, c) for r in range(stage.rows) for c in range(stage.cols) if env.world.is_passable(float(c) + 0.5, float(r) + 0.5, radius=0.35)]
    
    pos0 = open_cells[10]
    # Find a pos1 in open_cells that has clear LOS to pos0
    pos1 = None
    for cand in open_cells:
        if cand != pos0 and math.hypot(cand[0] - pos0[0], cand[1] - pos0[1]) < 6.0:
            if env.world.line_of_sight((pos0[1] + 0.5, pos0[0] + 0.5), (cand[1] + 0.5, cand[0] + 0.5), radius=0.35, step_size=0.5):
                pos1 = cand
                break
    assert pos1 is not None, "Could not find corridor pair with line of sight"

    # Find a pos2 that has NO line of sight to pos0 (behind a wall)
    pos2 = None
    for cand in open_cells:
        if cand != pos0 and math.hypot(cand[0] - pos0[0], cand[1] - pos0[1]) > 4.0:
            if not env.world.line_of_sight((cand[1] + 0.5, cand[0] + 0.5), (pos0[1] + 0.5, pos0[0] + 0.5), radius=0.35, step_size=0.5):
                pos2 = cand
                break
    assert pos2 is not None, "Could not find cell behind wall for ghost 2"

    g0.y, g0.x = float(pos0[0]) + 0.5, float(pos0[1]) + 0.5
    g1.y, g1.x = float(pos1[0]) + 0.5, float(pos1[1]) + 0.5
    g2.y, g2.x = float(pos2[0]) + 0.5, float(pos2[1]) + 0.5

    # Position powered Pacman on top of Ghost 0 to trigger a kill
    env.player.y, env.player.x = g0.y, g0.x
    env.player.powered = True
    env.player.power_timer = 100

    # Clear message queues and callouts
    g1.message_queue.clear()
    g2.message_queue.clear()
    assert g1.callout is None
    assert g2.callout is None
    assert 0 not in g1.dead_agents
    assert 0 not in g2.dead_agents

    # Step the environment - Pacman collides with and eats Ghost 0
    R = int(stage.rows * stage.obs_resolution)
    C = int(stage.cols * stage.obs_resolution)
    dummy_action = {gid: ([], np.zeros((R, C), dtype=np.float32), 1.0) for gid in env.ghosts}
    obs, rewards, done, info = env.step(dummy_action)

    # Ghost 0 is dead
    assert g0.dead == True

    # Ghost 1 had LOS to the kill -> Ghost 1 MUST witness the death!
    assert 0 in g1.dead_agents, "Ghost 1 (with LOS) should have witnessed Ghost 0's death!"
    assert g1.callout == "Ghost 0 DOWN!", f"Ghost 1 should display dead callout, got {g1.callout}"
    assert g1.callout_timer > 0, "Ghost 1 callout timer should be active"

    # Ghost 2 was behind a wall with NO LOS to the kill
    # Check that Ghost 2 did not witness it directly:
    assert g2.callout is None, "Ghost 2 (behind wall) should NOT have a dead callout for a death it did not witness!"


def test_ghosts_lack_omniscient_power_state_and_death_access():
    """Verify ghosts cannot omnisciently sense Pacman's power state or peer deaths without sight or radio."""
    stage = STAGES[2]
    env = Env(env_id=0, num_ghosts=2, world_height=float(stage.rows), world_width=float(stage.cols), obs_resolution=stage.obs_resolution, n_power=stage.n_power)
    env.reset()

    g0 = env.ghosts[0]
    g1 = env.ghosts[1]

    # Place Ghost 0 and Pacman in opposite corners behind multiple walls, far outside radio range (RADIUS=12.0)
    open_cells = [(r, c) for r in range(stage.rows) for c in range(stage.cols) if env.world.is_passable(float(c) + 0.5, float(r) + 0.5, radius=0.35)]
    top_left = open_cells[0]
    bottom_right = open_cells[-1]

    g0.y, g0.x = float(top_left[0]) + 0.5, float(top_left[1]) + 0.5
    env.player.y, env.player.x = float(bottom_right[0]) + 0.5, float(bottom_right[1]) + 0.5
    dist = math.hypot(env.player.y - g0.y, env.player.x - g0.x)
    assert dist > 15.0, f"Distance {dist} should be well beyond MAX_RAY_DIST (12.0)"

    # Pacman becomes powered across the map
    env.player.powered = True
    env.player.power_timer = 200

    # Ensure ghost state starts clean
    g0.pacman_powered = False
    g0.known_pacman = None

    # Step Ghost 0
    g0.update((env.player.y, env.player.x), True, env.ghosts)

    # Ghost 0 MUST NOT know Pacman is powered because it cannot see Pacman!
    assert g0.known_pacman is None, "Ghost 0 should not know Pacman position across map"
    assert g0.pacman_powered == False, "Ghost 0 must NOT have access to powered state without seeing it!"

    # Now test dead agent knowledge: Pacman kills Ghost 1 on the other side of map
    g1.y, g1.x = env.player.y, env.player.x
    d_g0_g1 = math.hypot(g0.y - g1.y, g0.x - g1.x)
    assert d_g0_g1 > 12.0, "Ghost 0 and Ghost 1 should be far apart"

    # Ghost 1 is killed by Pacman
    kill_x, kill_y = g1.x, g1.y
    g1.kill()

    # Neither LOS nor direct radio can reach Ghost 0 across the map (> 12.0 distance)
    d_w = math.hypot(g0.y - kill_y, g0.x - kill_x)
    assert d_w > 12.0
    # Ghost 0 does not witness it
    assert 1 not in g0.dead_agents, "Ghost 0 must not omnisciently know Ghost 1 is dead without seeing it or receiving radio"
    assert g0.callout is None, "Ghost 0 must not call out a death it cannot see"


def test_dead_ghost_list_in_rl_vector():
    """Verify build_vector explicitly encodes peer dead status, unknown status, and team casualty ratio."""
    from obs import build_vector, VEC_DIM, MAX_GHOSTS
    stage = STAGES[2]
    env = Env(env_id=0, num_ghosts=3, world_height=float(stage.rows), world_width=float(stage.cols), obs_resolution=stage.obs_resolution, n_power=stage.n_power)
    env.reset()

    g0 = env.ghosts[0]
    # Configure peers:
    # Ghost 1: Alive with known position
    g0.known_agents[1] = (5.0, 5.0)
    # Ghost 2: Confirmed dead
    g0.dead_agents.add(2)
    g0.known_agents[2] = "UNKNOWN"

    vec = build_vector(g0)
    assert len(vec) == VEC_DIM, f"Vector length {len(vec)} must match VEC_DIM {VEC_DIM}"

    # Features 6..18 encode [is_unknown, is_dead] for the 6 other ghosts (gids 1..6)
    # Ghost 1 (index 0 among other ghosts):
    # known position -> is_unknown=0.0, is_dead=0.0
    u1 = vec[6]
    d1 = vec[7]
    assert u1 == 0.0, f"Ghost 1 with known position should have is_unknown=0.0, got {u1}"
    assert d1 == 0.0, f"Ghost 1 alive should have is_dead=0.0, got {d1}"

    # Ghost 2 (index 1 among other ghosts):
    # confirmed dead -> is_unknown=0.0, is_dead=1.0
    u2 = vec[8]
    d2 = vec[9]
    assert u2 == 0.0, f"Ghost 2 confirmed dead should have is_unknown=0.0, got {u2}"
    assert d2 == 1.0, f"Ghost 2 confirmed dead should have is_dead=1.0, got {d2}"

    # Ghost 3 (uninitialized / unknown):
    # is_unknown=1.0, is_dead=0.0
    u3 = vec[10]
    d3 = vec[11]
    assert u3 == 1.0, f"Ghost 3 unknown should have is_unknown=1.0, got {u3}"
    assert d3 == 0.0, f"Ghost 3 alive should have is_dead=0.0, got {d3}"

    # Casualty ratio at index 18: 1 dead out of 6 peers = 1/6
    casualty_ratio = vec[18]
    assert abs(casualty_ratio - (1.0 / (MAX_GHOSTS - 1))) < 1e-4, f"Casualty ratio should be ~0.1667, got {casualty_ratio}"


def test_peer_not_seen_at_last_known_pos_marked_unknown():
    """Verify that when a ghost looks at a peer's last known position and sees nobody there,
    it immediately marks the peer UNKNOWN (not keeping stale coordinates), but does NOT falsely mark it dead."""
    stage = STAGES[2]
    env = Env(env_id=0, num_ghosts=2, world_height=float(stage.rows), world_width=float(stage.cols), obs_resolution=stage.obs_resolution, n_power=stage.n_power)
    env.reset()

    g0 = env.ghosts[0]
    g1 = env.ghosts[1]

    # Find two passable positions with mutual line of sight in an open corridor
    open_cells = [(r, c) for r in range(stage.rows) for c in range(stage.cols) if env.world.is_passable(float(c) + 0.5, float(r) + 0.5, radius=0.35)]
    pos0 = open_cells[15]
    pos1_stale = None
    for cand in open_cells:
        if cand != pos0 and math.hypot(cand[0] - pos0[0], cand[1] - pos0[1]) < 5.0:
            if env.world.line_of_sight((pos0[1] + 0.5, pos0[0] + 0.5), (cand[1] + 0.5, cand[0] + 0.5), radius=0.35, step_size=0.5):
                pos1_stale = cand
                break
    assert pos1_stale is not None

    g0.y, g0.x = float(pos0[0]) + 0.5, float(pos0[1]) + 0.5
    stale_y, stale_x = float(pos1_stale[0]) + 0.5, float(pos1_stale[1]) + 0.5

    # Ghost 0's belief says Ghost 1 was at pos1_stale, and received a heartbeat recently (2 frames ago)
    g0.known_agents[1] = (stale_y, stale_x)
    g0.last_heartbeat[1] = g0.frame - 2

    # But Ghost 1 is actually far away (e.g. around a corner at open_cells[0])
    g1.y, g1.x = float(open_cells[0][0]) + 0.5, float(open_cells[0][1]) + 0.5

    # Ghost 0 scans lidar (run 2 frames so frame % LIDAR_SWEEP_EVERY == 0 triggers sweep)
    g0.update((env.player.y, env.player.x), False, env.ghosts)
    g0.update((env.player.y, env.player.x), False, env.ghosts)

    # Ghost 0 is looking directly at (stale_y, stale_x) and Ghost 1 is NOT there:
    # Ghost 0 must IMMEDIATELY mark Ghost 1 as UNKNOWN!
    assert g0.known_agents[1] == "UNKNOWN", f"Ghost 0 should immediately mark Ghost 1 UNKNOWN, got {g0.known_agents[1]}"

    # BUT Ghost 1 must NOT be falsely marked dead (Ghost 1 is alive, just moved!)
    assert 1 not in g0.dead_agents, "Ghost 1 should NOT be falsely marked dead just because it moved away!"
    assert not g0.is_agent_dead(1), "is_agent_dead should return False for active live peer"



