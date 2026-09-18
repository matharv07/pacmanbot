"""
Potential-based reward shaping for the MAPPO ghost pursuit pipeline.

Every shaping term is formulated as  r(t) = γ Φ(s_{t+1}) - Φ(s_t), so that
the optimal policy is invariant to the shaping (Ng et al., 1999).

Tuned for AGGRESSIVE coordinated swarm pursuit:
- Hunt potential uses steep close-range gradient to reward closing distance
- Surround potential rewards multi-angle encirclement (pincer formation)
- Mesh connectivity potential keeps the radio mesh intact for coordination
- Cornering potential rewards trapping Pacman in dead-ends
- Dispersion potential prevents useless clumping far from target
"""

import math
import numpy as np

class RewardShaper:
    """Tracks per-ghost potentials and returns the shaping delta each step."""

    def __init__(self, alpha=6.0, beta=7.0, gamma_ex=0.008, delta_peak=2.5,
                 delta_spread=2.5, delta_ent=1.0, beta_mesh=3.5,
                 alpha_corner=4.0, gamma=0.99):
        """
        Parameters
        ----------
        alpha        : hunt shaping weight — steep close-range gradient
        beta         : encirclement shaping weight — coordinated pincer reward
        gamma_ex     : exploration shaping weight (low: exploration is secondary to pursuit)
        delta_peak   : belief peak certainty weight
        delta_spread : belief spatial standard deviation penalty weight
        delta_ent    : belief normalized entropy weight
        beta_mesh    : extended mesh connectivity potential weight
        alpha_corner : dead-end cornering / trapping potential weight
        gamma        : RL discount factor for Ng et al. invariant shaping
        """
        self.alpha        = alpha
        self.beta         = beta
        self.gamma_ex     = gamma_ex
        self.delta_peak   = delta_peak
        self.delta_spread = delta_spread
        self.delta_ent    = delta_ent
        self.beta_mesh    = beta_mesh
        self.alpha_corner = alpha_corner
        self.gamma        = gamma
        self._prev: dict[int, float] = {}

    @staticmethod
    def _pac_target(ghost):
        t = ghost.known_pacman
        if t is not None:
            return t
        t = ghost.last_lost_pacman
        if t is not None and (getattr(ghost, 'frame', 0) - getattr(ghost, 'pacman_last_seen', 0) <= 25):
            return t
        if hasattr(ghost.belief_map, 'top_cells'):
            top = ghost.belief_map.top_cells(n=1)
            if top:
                return top[0]
        return None

    def _phi_hunt(self, ghost, target) -> float:
        if getattr(ghost, 'pacman_powered', False) or target is None:
            return 0.0
        target_ck = (round(float(target[0]), 2), round(float(target[1]), 2))
        d = None
        if hasattr(ghost, 'cbba_agent') and ghost.cbba_agent is not None:
            d = ghost.cbba_agent._dist_cache.get(target_ck, ghost.cbba_agent._dist_cache.get(target))
        if d is None or math.isinf(d) or math.isnan(d):
            w = getattr(ghost, 'world', None)
            if w is not None and hasattr(w, 'apsp') and hasattr(w, 'prm_node_idx'):
                p1 = (round(float(ghost.y), 1), round(float(ghost.x), 1))
                p2 = (round(float(target[0]), 1), round(float(target[1]), 1))
                idx1 = w.prm_node_idx.get(p1)
                idx2 = w.prm_node_idx.get(p2)
                if idx1 is not None and idx2 is not None:
                    d = float(w.apsp[idx1, idx2])
        if d is None or math.isinf(d) or math.isnan(d):
            d = abs(ghost.y - target[0]) + abs(ghost.x - target[1])
        if math.isinf(d) or math.isnan(d):
            d = 999.0
        #corridor lead interception: flanking ghosts cutting off Pacman's lead path
        p_dir = getattr(ghost, '_player_dir', (0, 0))
        p_speed = math.hypot(p_dir[0], p_dir[1])
        if p_speed > 0.05:
            lead_y = target[0] + p_dir[0] * 3.0
            lead_x = target[1] + p_dir[1] * 3.0
            d_lead = abs(ghost.y - lead_y) + abs(ghost.x - lead_x)
            d = min(d, d_lead + 0.5)
        #three-scale hunt potential for aggressive pursuit: - kill zone (sigma=2.5): extremely steep reward within striking distance - near chase (sigma=5.0): strong pull during active pursuit - far guide (sigma=14.0): gentle gradient across the whole map
        kill_zone  = 0.4 * math.exp(-d / 2.5)
        near_chase = 0.35 * math.exp(-d / 5.0)
        far_guide  = 0.25 * math.exp(-d / 14.0)
        return self.alpha * (kill_zone + near_chase + far_guide)

    def _phi_flee(self, ghost, target) -> float:
        if not getattr(ghost, 'pacman_powered', False) or target is None:
            return 0.0
        target_ck = (round(float(target[0]), 2), round(float(target[1]), 2))
        d = None
        if hasattr(ghost, 'cbba_agent') and ghost.cbba_agent is not None:
            d = ghost.cbba_agent._dist_cache.get(target_ck, ghost.cbba_agent._dist_cache.get(target))
        if d is None or math.isinf(d) or math.isnan(d):
            w = getattr(ghost, 'world', None)
            if w is not None and hasattr(w, 'apsp') and hasattr(w, 'prm_node_idx'):
                p1 = (round(float(ghost.y), 1), round(float(ghost.x), 1))
                p2 = (round(float(target[0]), 1), round(float(target[1]), 1))
                idx1 = w.prm_node_idx.get(p1)
                idx2 = w.prm_node_idx.get(p2)
                if idx1 is not None and idx2 is not None:
                    d = float(w.apsp[idx1, idx2])
        if d is None or math.isinf(d) or math.isnan(d):
            d = abs(ghost.y - target[0]) + abs(ghost.x - target[1])
        if math.isinf(d) or math.isnan(d):
            d = 999.0
        #danger potential: strongly negative when close to powered Pacman
        return -self.alpha * 2.0 * math.exp(-d / 6.0)

    def _phi_surround(self, ghost, all_ghosts, target) -> float:
        """Rewards multi-angle pincer/encirclement around Pacman using circular variance and distance compression."""
        if getattr(ghost, 'pacman_powered', False) or target is None:
            return 0.0
        pr, pc = target
        angles = []
        dists = []
        for g in all_ghosts.values():
            if getattr(g, 'dead', False):
                continue
            dy, dx = g.y - pr, g.x - pc
            if dy == 0 and dx == 0:
                continue
            dist = math.hypot(dy, dx)
            if dist <= 12.0:
                angles.append(math.atan2(dy, dx))
                dists.append(dist)
        if len(angles) < 2:
            return 0.0
        N = len(angles)
        R = math.hypot(sum(math.cos(a) for a in angles) / N,
                       sum(math.sin(a) for a in angles) / N)
        encirclement = 1.0 - R
        #distance compression: surges as the perimeter tightens around Pacman -- steeper proximity curve to reward closing in as a group
        avg_prox = sum(math.exp(-d / 5.0) for d in dists) / N
        #bonus for having 3+ ghosts converging (proper swarm)
        swarm_bonus = 1.0 + 0.3 * max(0, N - 2)
        return self.beta * encirclement * avg_prox * swarm_bonus

    def _phi_explore(self, ghost) -> float:
        if not hasattr(ghost.world, 'prm_nodes') or not hasattr(ghost, 'prm_last_seen'):
            return 0.0
        total_nodes = max(len(ghost.world.prm_nodes), 1)
        known = getattr(ghost, 'prm_known_count', None)
        if known is None:
            known = sum(1 for v in ghost.prm_last_seen.values() if v != -1)
        return self.gamma_ex * (known / total_nodes)

    def _phi_belief(self, ghost) -> float:
        if not hasattr(ghost.belief_map, '_b_flat') or not hasattr(ghost.belief_map, '_open_arr'):
            return 0.0
        b = ghost.belief_map._b_flat
        coords = ghost.belief_map._open_arr
        if len(b) == 0 or len(coords) != len(b):
            return 0.0
        p = b[b > 0]
        if p.size == 0:
            return 0.0
        peak = float(np.max(b))
        mu = np.dot(b, coords)
        diff = coords - mu
        var = float(np.dot(b, np.sum(diff**2, axis=1)))
        sigma = math.sqrt(max(0.0, var))
        diag = math.hypot(ghost.world.width, ghost.world.height) if ghost.world else 50.0
        norm_spread = min(1.0, sigma / max(diag, 1.0))
        # 3. Normalized entropy
        n_nodes = max(len(b), 2)
        entropy = -float(np.sum(p * np.log(p + 1e-12)))
        norm_entropy = min(1.0, entropy / math.log(n_nodes))
        return (self.delta_peak * peak 
                - self.delta_spread * norm_spread 
                - self.delta_ent * norm_entropy)

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

    def _phi_mesh(self, ghost, all_ghosts) -> float:
        """
        Extended mesh connectivity potential.
        Evaluates whether the ghost is part of the multi-hop connected radio mesh
        (radio radius = 12.0) and applies an elastic tension gradient before links sever.
        """
        alive = [g for g in all_ghosts.values() if not getattr(g, 'dead', False)]
        if len(alive) < 2:
            return 0.0

        visited = {ghost.gid}
        queue = [ghost]
        min_dist = math.inf
        for g in alive:
            if g.gid != ghost.gid:
                d = math.hypot(ghost.y - g.y, ghost.x - g.x)
                if d < min_dist:
                    min_dist = d

        while queue:
            curr = queue.pop(0)
            for g in alive:
                if g.gid not in visited:
                    if math.hypot(curr.y - g.y, curr.x - g.x) <= 12.0:
                        visited.add(g.gid)
                        queue.append(g)

        frac_connected = len(visited) / len(alive)
        tension = 0.0
        if min_dist > 8.0:
            tension = min(1.0, ((min_dist - 8.0) / 4.0) ** 2)
        return self.beta_mesh * (frac_connected - 0.5 * tension)

    def _phi_corner(self, ghost, all_ghosts, target) -> float:
        """Rewards closing in on and trapping Pacman in a dead-end or restricted corridor."""
        if getattr(ghost, 'pacman_powered', False) or target is None:
            return 0.0
        pr, pc = target
        dist_pac = math.hypot(ghost.y - pr, ghost.x - pc)
        if dist_pac > 6.0 or getattr(ghost, 'world', None) is None or not hasattr(ghost.world, 'is_passable'):
            return 0.0
        p_radius = 0.35
        cardinals = [(0.7, 0.0), (-0.7, 0.0), (0.0, 0.7), (0.0, -0.7)]
        open_exits = 0
        for dr, dc in cardinals:
            if ghost.world.is_passable(pc + dc, pr + dr, radius=p_radius):
                open_exits += 1
        #strong cornering reward: Pacman is trapped in a dead-end or corridor
        if open_exits <= 1:
            return self.alpha_corner * math.exp(-dist_pac / 3.0)
        elif open_exits == 2 and dist_pac < 4.0:
            #corridor: count how many ghosts are covering the two exits
            ghosts_covering = 0
            for g in all_ghosts.values():
                if getattr(g, 'dead', False) or g.gid == ghost.gid:
                    continue
                gd = math.hypot(g.y - pr, g.x - pc)
                if gd < 5.0:
                    ghosts_covering += 1
            cover_mult = 1.0 + 0.5 * min(ghosts_covering, 2)
            return 0.6 * self.alpha_corner * math.exp(-dist_pac / 3.0) * cover_mult
        return 0.0

    def potential(self, ghost, all_ghosts) -> float:
        target = self._pac_target(ghost)
        p = (self._phi_hunt(ghost, target) + 
             self._phi_surround(ghost, all_ghosts, target) + 
             self._phi_explore(ghost) + 
             self._phi_belief(ghost) + 
             self._phi_corner(ghost, all_ghosts, target) + 
             self._phi_mesh(ghost, all_ghosts) +
             self._phi_dispersion(ghost, all_ghosts) +
             self._phi_flee(ghost, target))
        return max(-6.0, min(6.0, p))

    def shaping(self, ghost, all_ghosts) -> float:
        phi = self.potential(ghost, all_ghosts)
        gid = ghost.gid
        if gid not in self._prev:
            self._prev[gid] = phi
            return 0.0
        r = (self.gamma * phi - self._prev[gid]) * 0.3
        self._prev[gid] = phi
        return max(-0.5, min(0.5, r))

    def reset(self):
        self._prev.clear()