import math
import numpy as np
import pytest
from curriculum import STAGES
from worker import Env
from allocator import Task, TaskType
from cbba import CBBA_Agent, _task_key
from ghost import Ghost

def test_stage4_false_death_elimination():
    """Verify that on Stage 4 (33x41, 7 ghosts), prolonged silence does not cause false agent_dead markings."""
    stage = STAGES[4]
    env = Env(0, num_ghosts=stage.n_ghosts, world_height=float(stage.rows), world_width=float(stage.cols), n_power=stage.n_power)
    env.reset()

    false_dead_events = 0
    for step in range(150):
        obs, rewards, done, info = env.step({}, bc_prob=1.0)
        for gid, g in env.ghosts.items():
            if g.dead:
                continue
            for other_gid in range(stage.n_ghosts):
                if other_gid != gid and not env.ghosts[other_gid].dead:
                    if other_gid in g.dead_agents:
                        false_dead_events += 1
        if done:
            break

    assert false_dead_events == 0, f"Expected 0 false dead peer events on Stage 4, got {false_dead_events}"


def test_task_metadata_roundtrip_mesh():
    """Verify that assigned_to, owner, and target_speed round-trip through CBBA consensus payload."""
    agent0 = CBBA_Agent(0)
    agent1 = CBBA_Agent(1)

    t = Task(task_type=TaskType.HUNT, target_pos=(12.0, 15.0), score=5.0, assigned_to=1, owner=0, target_speed=1.0)
    k = _task_key(t)
    agent0._task_map[k] = t
    agent0.y[k] = 5.0
    agent0.z[k] = 0
    agent0.s[0] = 10

    payload = agent0.get_consensus_payload()
    assert "meta" in payload, "Expected 'meta' in CBBA consensus payload"
    assert k in payload["meta"], "Expected task key in meta"
    assert payload["meta"][k] == (1, 0, 1.0)

    # Agent 1 receives consensus from Agent 0
    agent1.receive_consensus(0, payload["y"], payload["z"], payload["s"], frame=10, task_meta=payload.get("meta"))
    assert k in agent1._task_map, "Task should be reconstructed in agent1._task_map"
    t_reconstructed = agent1._task_map[k]
    assert t_reconstructed.assigned_to == 1, f"Expected assigned_to=1, got {t_reconstructed.assigned_to}"
    assert t_reconstructed.owner == 0, f"Expected owner=0, got {t_reconstructed.owner}"


def test_distance_horizon_gate_suppression():
    """Verify that distant ghosts (>22.0 units) suppress bids on unassigned peer hunt tasks."""
    class MockGhost:
        def __init__(self, gid, y, x):
            self.gid = gid
            self.y = y
            self.x = x
            self.frame = 10
            self.world = None

    g = MockGhost(1, y=30.0, x=35.0)
    agent = CBBA_Agent(1)

    # Create remote peer task at (2.0, 2.0) owned by Ghost 0, NOT assigned to Ghost 1
    t_peer = Task(task_type=TaskType.HUNT, target_pos=(2.0, 2.0), score=8.0, assigned_to=-1, owner=0)
    k_peer = _task_key(t_peer)
    agent._task_map[k_peer] = t_peer

    # Distance is hypot(28, 33) = ~43.2 > 22.0
    gain, n = agent._marginal_gain(k_peer, g)
    assert gain == 0.0, f"Expected 0.0 gain for distant peer task (>22 units), got {gain}"

    # If the task was explicitly assigned to Ghost 1, it should NOT be suppressed:
    t_assigned = Task(task_type=TaskType.HUNT, target_pos=(2.0, 2.0), score=8.0, assigned_to=1, owner=0)
    k_assigned = _task_key(t_assigned)
    agent._task_map[k_assigned] = t_assigned
    gain_assigned, _ = agent._marginal_gain(k_assigned, g)
    assert gain_assigned > 0.0, f"Expected positive gain for assigned cutoff task, got {gain_assigned}"


def test_zero_freeze_frames_stage4():
    """Verify that ghosts with active tasks do not freeze with zero velocity on Stage 4."""
    stage = STAGES[4]
    env = Env(0, num_ghosts=stage.n_ghosts, world_height=float(stage.rows), world_width=float(stage.cols), n_power=stage.n_power)
    env.reset()

    frozen_count = 0
    for step in range(100):
        obs, rewards, done, info = env.step({}, bc_prob=1.0)
        for gid, g in env.ghosts.items():
            if g.dead:
                continue
            act = g.cbba_agent.get_active_task()
            spd = (g.vx**2 + g.vy**2)**0.5
            if act is not None and spd < 0.01:
                frozen_count += 1
        if done:
            break

    assert frozen_count == 0, f"Expected 0 freeze frames with active task, got {frozen_count}"
