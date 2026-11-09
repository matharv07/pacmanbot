#!/usr/bin/env python3
"""
svanm2_sim_env.py — High-Fidelity SvanM2 Quadruped Simulation Environment for MAPPO Training.

Wraps native MuJoCo physics and Livox Mid-360 LiDAR observation pipeline into an
RL training environment adhering to SvanM2 physical dynamic constraints:
  • Realistic forward, lateral, and yaw rate envelopes:
      vx in [-0.2, 1.0] m/s, vy in [-0.35, 0.35] m/s, wz in [-1.6, 1.6] rad/s.
  • Acceleration / jerk damping and non-instantaneous turning radius limits.
  • 3D Livox Mid-360 LiDAR raycast pointclouds and 2.5D elevation/traversability mapping.
  • Decentralized peer-to-peer mesh networking (packet drops, range limits).
  • Compatible with standard pacmanbot Gym/worker training loops.
"""

import sys
import os
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PACMAN_SIM_DIR = _REPO_ROOT / "pacman_sim"
for p in [str(_REPO_ROOT), str(_PACMAN_SIM_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from game_engine import PacmanGameEngine
from arena_builder import ContinuousArena
from svanm2_runner import SvanM2Policy, SvanM2Runner, DEFAULT_QPOS
from lidar3d import LivoxMid360Lidar, ElevationTraversabilityMapper
from mesh_comms import DecentralizedMeshNetwork
from obs import (
    build_spatial,
    build_vector,
    build_valid_mask,
    actions_to_tasks,
    MAX_H,
    MAX_W,
    MAX_GHOSTS,
    SPATIAL_CH,
    VEC_DIM,
)
from allocator import Task, TaskType
from worker import RewardShaper


class SvanM2TrainingEnv:
    """MuJoCo + SvanM2 multi-agent training environment for pacmanbot."""

    def __init__(
        self,
        stage: int = 1,
        seed: Optional[int] = None,
        scale: float = 1.5,
        decision_substeps: int = 6,  # 6 MuJoCo/controller steps per high-level RL decision (120ms @ 50Hz)
    ):
        self.stage_idx = stage
        self.seed = seed
        self.scale = scale
        self.decision_substeps = decision_substeps
        self.dt_control = 0.02  # 50 Hz control cycle

        # Create MuJoCo Game Engine
        self.engine = PacmanGameEngine(stage=stage, seed=seed, scale=scale)
        self.n_ghosts = self.engine.n_ghosts
        self.shaper = RewardShaper()
        self.recent_nom = {gid: np.zeros((MAX_H, MAX_W), dtype=np.float32) for gid in range(self.n_ghosts)}
        self.frame = 0

    def reset(self):
        """Resets the simulation arena, SvanM2 robots, and Pac-Man."""
        self.engine.reset()
        self.frame = 0
        self.recent_nom = {gid: np.zeros((MAX_H, MAX_W), dtype=np.float32) for gid in range(self.n_ghosts)}
        return self._get_obs()

    def _get_obs(self) -> Dict[str, Any]:
        """Constructs spatial and vector observations for all active SvanM2 ghosts."""
        spatial_obs = np.zeros((self.n_ghosts, SPATIAL_CH, MAX_H, MAX_W), dtype=np.float32)
        vector_obs = np.zeros((self.n_ghosts, VEC_DIM), dtype=np.float32)
        valid_masks = np.zeros((self.n_ghosts, MAX_H, MAX_W), dtype=bool)

        coord = self.engine.coordinator
        for gid in range(self.n_ghosts):
            g = coord.ghosts[gid]
            if not g.dead:
                s = build_spatial(g, self.recent_nom[gid], rows=MAX_H, cols=MAX_W, obs_resolution=1.0)
                # Channel 5: Replace/blend heuristic safety map with Livox 2.5D traversability costmap
                if gid in coord.latest_traversability:
                    trav = coord.latest_traversability[gid]["traversability"]
                    th, tw = trav.shape
                    rh = min(MAX_H, th)
                    rw = min(MAX_W, tw)
                    s[5, :rh, :rw] = np.maximum(s[5, :rh, :rw], trav[:rh, :rw])

                spatial_obs[gid] = s
                vector_obs[gid] = build_vector(g, self.recent_nom[gid], all_ghosts=coord.ghosts)
                valid_masks[gid] = build_valid_mask(g, rows=MAX_H, cols=MAX_W)

        return {
            "spatial": spatial_obs,
            "vector": vector_obs,
            "valid_mask": valid_masks,
        }

    def step(self, actions: Dict[int, Any]) -> Tuple[Dict[str, Any], np.ndarray, bool, Dict[str, Any]]:
        """Applies high-level RL actions over decision_substeps of SvanM2 MuJoCo simulation.

        Args:
            actions: Dict mapping gid to (waypoint_actions, speed_mult)
        """
        coord = self.engine.coordinator
        
        # 1. Apply actions to CBBA bundle and waypoint nomination
        for gid, act in actions.items():
            if gid in coord.ghosts and not coord.ghosts[gid].dead:
                g = coord.ghosts[gid]
                if isinstance(act, tuple) and len(act) >= 2:
                    wps, spd = act[0], act[1]
                    g.current_speed_mult = float(spd)
                elif isinstance(act, (list, np.ndarray)):
                    wps = act
                else:
                    wps = []

                if len(wps) > 0:
                    tasks = actions_to_tasks(wps, g, rows=MAX_H, cols=MAX_W)
                    for t in tasks:
                        g.cbba_agent.add_task_to_bundle(t)

        # 2. Advance high-fidelity MuJoCo physics over the decision interval (6 substeps @ 50 Hz = 120ms)
        for _ in range(self.decision_substeps):
            self.engine.step_simulation(dt=self.dt_control)
            self.frame += 1
            if self.engine.game_over:
                break

        # 3. Construct returns
        obs = self._get_obs()
        rewards = np.zeros(self.n_ghosts, dtype=np.float32)
        done = self.engine.game_over
        info = {
            "score": self.engine.score,
            "pellets_left": len(self.engine.arena.pellets),
            "pacman_caught": (self.engine.winner == "ghosts"),
        }

        # Calculate rewards per ghost
        for gid in range(self.n_ghosts):
            g = coord.ghosts[gid]
            if g.dead:
                rewards[gid] -= 2.0
            elif self.engine.winner == "ghosts":
                rewards[gid] += 25.0
            else:
                # Dense distance-reduction shaping
                if g.known_pacman is not None:
                    d = math.hypot(g.y - g.known_pacman[0], g.x - g.known_pacman[1])
                    rewards[gid] += float(np.clip(1.0 / (d + 1.0), 0.0, 1.0))

        return obs, rewards, done, info

