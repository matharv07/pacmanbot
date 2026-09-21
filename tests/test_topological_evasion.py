import pytest
import math
import numpy as np
import random
from world import World
from ghost import Ghost
from pathfinder import find_topological_flee_target, astar
from curriculum import STAGES

def test_topological_flee_picks_safe_node():
    """Verify find_topological_flee_target selects a safe node with positive lead margin."""
    #seeded: the maze is random, and without this the test's outcome depended on how much RNG the
    #tests before it consumed, so the suite failed intermittently and could not be trusted as a gate
    random.seed(7); np.random.seed(7)
    w = World(21, 27, resolution=0.5)
    w.generate(n_obstacles=15, n_power=8)

    assert len(w.prm_nodes) > 10
    ghost_pos = tuple(w.prm_nodes[5])
    pac_pos = tuple(w.prm_nodes[6])

    target = find_topological_flee_target(w, ghost_pos, pac_pos, radius=0.35)
    assert target is not None
    assert target != pac_pos

    path = astar(w, ghost_pos, target, radius=0.35)
    assert len(path) >= 2, "Should find valid path towards flee target"

def test_topological_flee_avoids_dead_ends():
    """Verify that topological flee prefers high-degree junctions over degree-1 dead ends."""
    w = World(21, 27, resolution=0.5)
    w.generate(n_obstacles=10, n_power=4)

    # Artificially test node selection
    ghost_pos = tuple(w.prm_nodes[0])
    pac_pos = tuple(w.prm_nodes[1])

    target = find_topological_flee_target(w, ghost_pos, pac_pos, radius=0.35)
    if target is not None:
        target_idx = w.prm_node_idx.get(target)
        if target_idx is not None:
            deg = len(w.prm_graph.get(w.prm_nodes[target_idx], []))
            assert deg >= 2, f"Target node should have degree >= 2 (open corridor), got {deg}"

def test_ghost_evasion_controller_sets_topological_velocity():
    """Verify ghost.update sets desired velocity along topological escape path when pacman is powered."""
    w = World(21, 27, resolution=0.5)
    w.generate(n_obstacles=12, n_power=6)

    # Spawn ghost and pacman near each other
    g_pos = w.prm_nodes[3]
    p_pos = w.prm_nodes[4]
    grid = np.zeros((21, 27), dtype=np.int8)

    ghost = Ghost(0, grid, (int(g_pos[0]), int(g_pos[1])), (255, 0, 0), (int(p_pos[0]), int(p_pos[1])), world=w)
    ghost.y = float(g_pos[0])
    ghost.x = float(g_pos[1])
    ghost.pacman_powered = True
    ghost.known_pacman = (float(p_pos[0]), float(p_pos[1]))

    # Call ghost update with player_pos, powered=True, all_ghosts={0: ghost}
    p_pos_tuple = (float(p_pos[0]), float(p_pos[1]))
    ghost.update(p_pos_tuple, True, {0: ghost})

    # Verify ghost set non-zero desired velocity or moved
    v_mag = math.hypot(ghost.vx, ghost.vy)
    assert v_mag > 0.05, f"Ghost should be moving to evade powered pacman, got {v_mag}"

def test_3_stage_curriculum_configuration():
    """Verify STAGES is a monotone ramp ending on the README's full game."""
    # Stage 0: 13x17, 2 ghosts — solo-skills stage (survive, deny, track); kills need cooperation
    assert STAGES[0].rows == 13
    assert STAGES[0].cols == 17
    assert STAGES[0].n_ghosts == 2
    # board size, swarm size and power pellet count all ramp monotonically
    for a, b in zip(STAGES, STAGES[1:]):
        assert b.rows >= a.rows and b.cols >= a.cols
        assert b.n_ghosts > a.n_ghosts
        assert b.n_power > a.n_power

    # Final stage: 33x41, 7 ghosts, 28 power pellets
    assert STAGES[-1].rows == 33
    assert STAGES[-1].cols == 41
    assert STAGES[-1].n_ghosts == 7
    assert STAGES[-1].n_power == 28
