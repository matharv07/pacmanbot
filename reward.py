"""
Potential-based reward shaping for the MAPPO ghost pursuit pipeline.

Every shaping term is formulated as  r(t) = γ Φ(s_{t+1}) - Φ(s_t), so that
the optimal policy is invariant to the shaping (Ng et al., 1999).
"""

import math
import numpy as np

class RewardShaper:
    """Tracks per-ghost potentials and returns the shaping delta each step."""

    def __init__(self, alpha=1.0, beta=2.0, gamma_ex=0.01, delta=0.05, gamma=0.99):
        """
        Parameters
        ----------
        alpha    : hunt shaping weight (distance to Pacman or belief peak)
        beta     : encirclement shaping weight (circular variance / pincer formation)
        gamma_ex : exploration shaping weight
        delta    : belief-entropy shaping weight
        gamma    : RL discount factor for Ng et al. invariant shaping
        """
        self.alpha    = alpha
        self.beta     = beta
        self.gamma_ex = gamma_ex
        self.delta    = delta
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

    def _phi_hunt(self, ghost, target) -> float:
        if getattr(ghost, 'pacman_powered', False) or target is None:
            return 0.0
        # Use cached Dijkstra distance if available (from CBBA auction)
        if hasattr(ghost, 'cbba_agent') and target in ghost.cbba_agent._dist_cache:
            d = ghost.cbba_agent._dist_cache[target]
            if math.isinf(d) or math.isnan(d):
                d = abs(ghost.y - target[0]) + abs(ghost.x - target[1])
        else:
            d = abs(ghost.y - target[0]) + abs(ghost.x - target[1])
        if math.isinf(d) or math.isnan(d):
            d = 999.0
        diag = math.hypot(ghost.world.width, ghost.world.height)
        return -self.alpha * (d / diag)

    def _phi_flee(self, ghost, target) -> float:
        if not getattr(ghost, 'pacman_powered', False) or target is None:
            return 0.0
        if hasattr(ghost, 'cbba_agent') and target in ghost.cbba_agent._dist_cache:
            d = ghost.cbba_agent._dist_cache[target]
            if math.isinf(d) or math.isnan(d):
                d = abs(ghost.y - target[0]) + abs(ghost.x - target[1])
        else:
            d = abs(ghost.y - target[0]) + abs(ghost.x - target[1])
        if math.isinf(d) or math.isnan(d):
            d = 999.0
        diag = math.hypot(ghost.world.width, ghost.world.height)
        return self.alpha * (d / diag)

    def _phi_surround(self, ghost, all_ghosts, target) -> float:
        """Rewards multi-angle pincer/encirclement around Pacman using circular variance."""
        if getattr(ghost, 'pacman_powered', False) or target is None:
            return 0.0
        pr, pc = target
        angles = []
        for g in all_ghosts.values():
            if getattr(g, 'dead', False):
                continue
            dy, dx = g.y - pr, g.x - pc
            if dy == 0 and dx == 0:
                continue
            dist = math.hypot(dy, dx)
            if dist <= 15.0:
                angles.append(math.atan2(dy, dx))
        if len(angles) < 2:
            return 0.0
        N = len(angles)
        R = math.hypot(sum(math.cos(a) for a in angles) / N,
                       sum(math.sin(a) for a in angles) / N)
        return self.beta * (1.0 - R)

    def _phi_explore(self, ghost) -> float:
        if not hasattr(ghost.world, 'prm_nodes') or not hasattr(ghost, 'prm_last_seen'):
            return 0.0
        total_nodes = max(len(ghost.world.prm_nodes), 1)
        known = getattr(ghost, 'prm_known_count', None)
        if known is None:
            known = sum(1 for v in ghost.prm_last_seen.values() if v != -1)
        return self.gamma_ex * (known / total_nodes)

    def _phi_belief(self, ghost) -> float:
        if not hasattr(ghost.belief_map, '_b_flat'):
            return 0.0
        b = ghost.belief_map._b_flat
        p = b[b > 0]
        if p.size == 0:
            return 0.0
        entropy = -float(np.sum(p * np.log(p + 1e-12)))
        return -self.delta * entropy

    def _phi_dispersion(self, ghost, all_ghosts) -> float:
        """Repulsive potential to prevent ghosts from clumping up during search."""
        if len(all_ghosts) < 2:
            return 0.0
        min_dist = 999.0
        for gid, g in all_ghosts.items():
            if gid == ghost.gid or getattr(g, 'dead', False):
                continue
            dist = math.hypot(ghost.y - g.y, ghost.x - g.x)
            if dist < min_dist:
                min_dist = dist
        repulsion_radius = max(2.0, min(ghost.world.height, ghost.world.width) * 0.15)
        if min_dist < repulsion_radius:
            return -self.gamma_ex * ((repulsion_radius - min_dist) / repulsion_radius)
        return 0.0

    def potential(self, ghost, all_ghosts) -> float:
        target = self._pac_target(ghost)
        return (self._phi_hunt(ghost, target) + 
                self._phi_surround(ghost, all_ghosts, target) + 
                self._phi_explore(ghost) + 
                self._phi_belief(ghost) +
                self._phi_dispersion(ghost, all_ghosts) +
                self._phi_flee(ghost, target))

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
