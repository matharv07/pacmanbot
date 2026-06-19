"""
Potential-based reward shaping for the MAPPO ghost pursuit pipeline.

Every shaping term is formulated as  r(t) = Φ(s_{t+1}) - Φ(s_t), so that
the optimal policy is invariant to the shaping (Ng et al., 1999)
"""

import math
import numpy as np
from pathfinder import dijkstra_multi

class RewardShaper:
    """Tracks per-ghost potentials and returns the shaping delta each step."""

    def __init__(self, alpha=0.4, beta=1.2, gamma_ex=0.005, delta=0.02, epsilon=0.05, gamma=0.99):
        """
        Parameters
        ----------
        alpha    : hunt shaping weight (distance to Pacman prediction)
        beta     : encirclement shaping weight (circular variance)
        gamma_ex : exploration shaping weight — SMALL so it doesn't
                   drown out pursuit signals (raw cell count is O(100))
        delta    : belief-entropy shaping weight
        gamma    : RL discount factor for Ng et al. invariant shaping
        """
        self.alpha    = alpha
        self.beta     = beta
        self.gamma_ex = gamma_ex
        self.delta    = delta
        self.epsilon  = epsilon
        self.gamma    = gamma
        self._prev: dict[int, float] = {}

    @staticmethod
    def _pac_target(ghost):
        t = ghost.known_pacman
        if t is not None:
            return t
        t = ghost.last_lost_pacman
        if t is not None:
            return t
        if hasattr(ghost.belief_map, 'top_cells'):
            top = ghost.belief_map.top_cells(n=1)
            if top:
                return top[0]
        return None

    def _phi_hunt(self, ghost) -> float:
        target = self._pac_target(ghost)
        if target is None:
            return 0.0
        #use cached Dijkstra distance if available (from CBBA auction)
        if hasattr(ghost, 'cbba_agent') and target in ghost.cbba_agent._dist_cache:
            d = ghost.cbba_agent._dist_cache[target]
            if math.isinf(d) or math.isnan(d):
                d = abs(ghost.row - target[0]) + abs(ghost.col - target[1])
        else:
            #fallback to manhattan if cache miss
            d = abs(ghost.row - target[0]) + abs(ghost.col - target[1])
        if math.isinf(d) or math.isnan(d):
            d = 999.0
        diag = len(ghost.grid) + len(ghost.grid[0])
        if getattr(ghost, 'pacman_powered', False):
            return 0.0
        return -self.alpha * (d / diag)

    def _phi_flee(self, ghost) -> float:
        if not getattr(ghost, 'pacman_powered', False):
            return 0.0
        target = self._pac_target(ghost)
        if target is None:
            return 0.0
        #use cached Dijkstra distance if available (from CBBA auction)
        if hasattr(ghost, 'cbba_agent') and target in ghost.cbba_agent._dist_cache:
            d = ghost.cbba_agent._dist_cache[target]
            if math.isinf(d) or math.isnan(d):
                d = abs(ghost.row - target[0]) + abs(ghost.col - target[1])
        else:
            #fallback to manhattan if cache miss
            d = abs(ghost.row - target[0]) + abs(ghost.col - target[1])
        if math.isinf(d) or math.isnan(d):
            d = 999.0
        diag = len(ghost.grid) + len(ghost.grid[0])
        # positive potential for distance when powered -> encourages fleeing
        return self.alpha * (d / diag)

    def _phi_surround(self, ghost, all_ghosts) -> float:
        if getattr(ghost, 'pacman_powered', False):
            return 0.0
        target = self._pac_target(ghost)
        if target is None:
            return 0.0
        pr, pc = target
        angles = []
        for g in all_ghosts.values():
            if g.dead:
                continue
            dy, dx = g.row - pr, g.col - pc
            if dy == 0 and dx == 0:
                continue
            angles.append(math.atan2(dy, dx))
        if len(angles) < 3:
            return 0.0
        # circular variance = 1 - || mean unit-vector ||
        N = len(angles)
        R = math.hypot(sum(math.cos(a) for a in angles) / N,
                       sum(math.sin(a) for a in angles) / N)
        return self.beta * (1.0 - R)

    def _phi_explore(self, ghost) -> float:
        #fraction of open cells known (0-1), NOT raw counts — keeps scale O(1)
        p = ghost.personal_map
        open_cells = np.sum(p != 1)   #exclude walls from denominator
        total_open = max(open_cells, 1)
        known = np.sum((p != -1) & (p != 1))   #not unknown, not wall
        return self.gamma_ex * (known / total_open)

    def _phi_belief(self, ghost) -> float:
        if not hasattr(ghost.belief_map, '_b'):
            return 0.0
        b = ghost.belief_map._b
        p = b[b > 0]
        if p.size == 0:
            return 0.0
        entropy = -float(np.sum(p * np.log(p + 1e-12)))
        return -self.delta * entropy

    def _phi_dispersion(self, ghost, all_ghosts) -> float:
        """Repulsive potential to prevent ghosts from clumping up during search."""
        if len(all_ghosts) < 3:
            return 0.0
        
        min_dist = 999.0
        for gid, g in all_ghosts.items():
            if gid == ghost.gid or g.dead:
                continue
            dist = abs(ghost.row - g.row) + abs(ghost.col - g.col)
            if dist < min_dist:
                min_dist = dist
        rows = len(ghost.grid)
        cols = len(ghost.grid[0])
        repulsion_radius = max(2.0, min(rows, cols) * 0.15)        
        if min_dist < repulsion_radius:
            return -self.gamma_ex * ((repulsion_radius - min_dist) / repulsion_radius)
        return 0.0

    def potential(self, ghost, all_ghosts) -> float:
        return (self._phi_hunt(ghost) + 
                self._phi_surround(ghost, all_ghosts) + 
                self._phi_explore(ghost) + 
                self._phi_belief(ghost) +
                self._phi_dispersion(ghost, all_ghosts) +
                self._phi_flee(ghost))

    def shaping(self, ghost, all_ghosts) -> float:
        phi = self.potential(ghost, all_ghosts)
        gid = ghost.gid
        if gid not in self._prev:
            self._prev[gid] = phi
            return 0.0
        r = self.gamma * phi - self._prev[gid]
        self._prev[gid] = phi
        return r

    def reset(self):
        self._prev.clear()