import math
import numpy as np
import pytest
from allocator import Task, TaskType
from cbba import CBBA_Agent, _task_key
from ghost import Ghost
from obs import (
    build_spatial,
    build_vector,
    build_global_spatial,
    SPATIAL_CH,
    GLOBAL_SPATIAL_CH,
    VEC_DIM,
    CRITIC_VEC_DIM,
    MAX_GHOSTS,
)
from curriculum import STAGES
from worker import Env


def test_spatial_channel_3_peer_intent_and_corridors():
    """Verify that Channel 3 renders peer task commitments as Gaussian blobs and corridor rays."""
    stage = STAGES[0]
    env = Env(env_id=0, num_ghosts=3, world_height=float(stage.rows), world_width=float(stage.cols), obs_resolution=stage.obs_resolution, n_power=stage.n_power)
    env.reset()

    g0 = env.ghosts[0]
    g1 = env.ghosts[1]

    rows = int(stage.rows * stage.obs_resolution)
    cols = int(stage.cols * stage.obs_resolution)
    recent_noms = np.zeros((rows, cols), dtype=np.float32)

    # Initial state: no peer tasks in g0's cbba_agent
    sp0 = build_spatial(g0, recent_noms, rows, cols, stage.obs_resolution)
    assert sp0.shape == (SPATIAL_CH, rows, cols)
    # Channel 3 should be flat 0
    assert np.all(sp0[3] == 0.0), "Channel 3 should be empty when peers have no tasks"

    # Now give Ghost 1 an active task at (5.5, 7.5) with target_speed 0.85
    t1 = Task(task_type=TaskType.HUNT, target_pos=(5.5, 7.5), score=3.0, assigned_to=1, owner=1, target_speed=0.85)
    k1 = _task_key(t1)
    g0.cbba_agent._task_map[k1] = t1
    g0.cbba_agent.z[k1] = 1
    g0.cbba_agent.y[k1] = 3.0

    # Set known position of Ghost 1 to (5.5, 2.5)
    g0.known_agents[1] = (5.5, 2.5)

    sp1 = build_spatial(g0, recent_noms, rows, cols, stage.obs_resolution)
    # 1. Active target blob at (5.5, 7.5)
    target_r = int(5.5 * stage.obs_resolution)
    target_c = int(7.5 * stage.obs_resolution)
    assert sp1[3, target_r, target_c] > 0.5, f"Expected active target blob at ({target_r}, {target_c}), got {sp1[3, target_r, target_c]}"
    # Peak value should be scaled by target_speed (0.85)
    assert abs(sp1[3, target_r, target_c] - 0.85) < 0.05

    # 2. Corridor ray connecting (5.5, 2.5) to (5.5, 7.5) along row 5
    mid_r = int(5.5 * stage.obs_resolution)
    mid_c = int(4.5 * stage.obs_resolution)
    assert sp1[3, mid_r, mid_c] > 0.15, f"Expected corridor ray at ({mid_r}, {mid_c}), got {sp1[3, mid_r, mid_c]}"


def test_channels_8_13_staleness_decay():
    """Verify that peer channels render staleness decay footprints when peer is UNKNOWN."""
    stage = STAGES[0]
    env = Env(env_id=0, num_ghosts=3, world_height=float(stage.rows), world_width=float(stage.cols), obs_resolution=stage.obs_resolution, n_power=stage.n_power)
    env.reset()

    g0 = env.ghosts[0]
    rows = int(stage.rows * stage.obs_resolution)
    cols = int(stage.cols * stage.obs_resolution)
    recent_noms = np.zeros((rows, cols), dtype=np.float32)

    # Ghost 1 is at ch = 8
    ch_g1 = 8

    # Case A: Ghost 1 has known position at cell center (4.5, 4.5)
    g0.known_agents[1] = (4.5, 4.5)
    sp_known = build_spatial(g0, recent_noms, rows, cols, stage.obs_resolution)
    r_target, c_target = int(4.5 * stage.obs_resolution), int(4.5 * stage.obs_resolution)
    assert sp_known[ch_g1, r_target, c_target] > 0.9

    # Case B: Ghost 1 drops into radio shadow (UNKNOWN), but has last known position from 15 frames ago
    g0.known_agents[1] = "UNKNOWN"
    g0._last_known_agent_pos[1] = (4.5, 4.5)
    g0.frame = 100
    g0.last_heartbeat[1] = 85  # delta = 15 frames
    sp_decayed = build_spatial(g0, recent_noms, rows, cols, stage.obs_resolution)

    decay_15 = math.exp(-15.0 / 30.0)  # ~0.606
    val_decayed = sp_decayed[ch_g1, r_target, c_target]
    assert abs(val_decayed - decay_15) < 0.05, f"Expected decayed footprint ~{decay_15}, got {val_decayed}"

    # Case C: Ghost 1 lost for 150 frames (decay <= 0.05) -> should fade out completely
    g0.last_heartbeat[1] = 100 - 150
    sp_faded = build_spatial(g0, recent_noms, rows, cols, stage.obs_resolution)
    assert sp_faded[ch_g1, r_target, c_target] == 0.0, "Channel should be 0 after long loss"


def test_vector_dimensions_and_agent_id_one_hot():
    """Verify vector feature dimension (163), agent ID one-hot, own tasks, and peer telemetry."""
    stage = STAGES[1]
    env = Env(env_id=0, num_ghosts=3, world_height=float(stage.rows), world_width=float(stage.cols), obs_resolution=stage.obs_resolution, n_power=stage.n_power)
    env.reset()

    for gid in (0, 1, 2):
        g = env.ghosts[gid]
        vec = build_vector(g)
        assert len(vec) == VEC_DIM, f"Vector length {len(vec)} does not match VEC_DIM {VEC_DIM}"
        assert VEC_DIM == 163, f"VEC_DIM expected 163, got {VEC_DIM}"

        # Agent ID one-hot: indices 27..33
        id_slice = vec[27:34]
        assert len(id_slice) == MAX_GHOSTS
        for i in range(MAX_GHOSTS):
            expected = 1.0 if i == gid else 0.0
            assert id_slice[i] == expected, f"Agent {gid} one-hot at {i} expected {expected}, got {id_slice[i]}"

    # Verify critic vec dim
    assert CRITIC_VEC_DIM == MAX_GHOSTS * VEC_DIM + MAX_GHOSTS
    assert CRITIC_VEC_DIM == 1148


def test_global_spatial_channel_count_and_pacman_power():
    """Verify build_global_spatial output tensor shape (12, H, W) and Pacman power channel."""
    stage = STAGES[0]
    env = Env(env_id=0, num_ghosts=stage.n_ghosts, world_height=float(stage.rows), world_width=float(stage.cols), obs_resolution=stage.obs_resolution, n_power=stage.n_power)
    env.reset()

    rows = int(stage.rows * stage.obs_resolution)
    cols = int(stage.cols * stage.obs_resolution)

    # 1. Normal state: Pacman not powered
    gsp = build_global_spatial(env, rows, cols, stage.obs_resolution)
    assert gsp.shape == (GLOBAL_SPATIAL_CH, rows, cols)
    assert GLOBAL_SPATIAL_CH == 12
    # Channel 4 is pacman power status
    assert np.all(gsp[4] == 0.0), "Pacman power channel should be 0 when unpowered"

    # 2. Powered state
    env.player.powered = True
    env.player.power_timer = 20
    gsp_powered = build_global_spatial(env, rows, cols, stage.obs_resolution)
    assert np.all(gsp_powered[4] == 0.5), "Pacman power channel should reflect power_timer / 40.0"

    # 3. Ghosts in channels 5..11
    for gid, g in env.ghosts.items():
        if gid < MAX_GHOSTS:
            ch = 5 + gid
            r = int(g.y * stage.obs_resolution)
            c = int(g.x * stage.obs_resolution)
            assert gsp[ch, r, c] > 0.5, f"Ghost {gid} expected at channel {ch}, ({r}, {c})"
