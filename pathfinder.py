import heapq
import math
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as _sp_dijkstra
import os as _os

FLEE_SPEED_RATIO = 1.0 / max(1e-6, float(_os.environ.get("GHOST_SPEED", "0.50")))
def _euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def _connect_temp_nodes_batch(world, nodes_list, radius=0.3):
    results = []
    prm_arr = world.prm_nodes_arr
    if prm_arr is None or len(prm_arr) == 0:
        return [(np.array([]), np.array([], dtype=int)) for _ in nodes_list]
    if not hasattr(world, '_conn_cache'):
        world._conn_cache = {}
    misses = []
    results_map = {}
    for i, origin in enumerate(nodes_list):
        origin_tup = (round(float(origin[0]), 2), round(float(origin[1]), 2), round(float(radius), 2))
        if origin_tup in world._conn_cache:
            results_map[i] = world._conn_cache[origin_tup]
        else:
            misses.append((i, origin, origin_tup))
    if misses:
        all_p1s = []
        all_p2s = []
        all_dists = []
        all_indices = []
        miss_offsets = []
        current_offset = 0
        for i, origin, origin_tup in misses:
            dy = prm_arr[:, 0] - origin[0]
            dx = prm_arr[:, 1] - origin[1]
            dist = np.hypot(dx, dy)
            valid_mask = dist <= 10.0
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) > 0:
                valid_nodes = prm_arr[valid_indices]
                valid_targets = np.column_stack((valid_nodes[:, 1], valid_nodes[:, 0]))
                origin_xy = (origin[1], origin[0])
                p1_arr = np.full((len(valid_targets), 2), origin_xy, dtype=np.float32)
                all_p1s.append(p1_arr)
                all_p2s.append(valid_targets)
                all_dists.append(dist[valid_indices])
                all_indices.append(valid_indices)
                miss_offsets.append((i, origin_tup, current_offset, current_offset + len(valid_targets)))
                current_offset += len(valid_targets)
            else:
                miss_offsets.append((i, origin_tup, current_offset, current_offset))
        if current_offset > 0:
            p1s = np.vstack(all_p1s)
            p2s = np.vstack(all_p2s)
            dists = np.concatenate(all_dists)
            indices = np.concatenate(all_indices)
            if hasattr(world, 'batch_line_of_sight_pairs'):
                los = world.batch_line_of_sight_pairs(p1s, p2s, radius=radius, step_size=0.5)
            elif hasattr(world, 'batch_line_of_sight'):   #fallback if pairs isn't available
                los = np.zeros(len(p1s), dtype=bool)
                for start, p1 in zip(range(0, len(p1s), current_offset), all_p1s):
                    pass
            else:
                los = np.ones(len(p1s), dtype=bool)
        for i, origin_tup, start, end in miss_offsets:
            if start == end:
                res = (np.array([]), np.array([], dtype=int))
            else:
                l_mask = los[start:end]
                clear_dists = dists[start:end][l_mask]
                clear_indices = indices[start:end][l_mask]
                res = (clear_dists, clear_indices)
            world._conn_cache[origin_tup] = res
            results_map[i] = res
    if len(world._conn_cache) > 5000:
        world._conn_cache.clear()
    for i in range(len(nodes_list)):
        results.append(results_map[i])
    return results

def _reconstruct(came_from, node):
    path = [node]
    while node in came_from:
        node = came_from[node]
        path.append(node)
    path.reverse()
    return path

def astar(world, start, goal, radius=0.3):
    if start == goal:
        return [start]
    if not hasattr(world, 'apsp') or not hasattr(world, 'prm_node_idx'):
        return []
    start_conns, goal_conns = _connect_temp_nodes_batch(world, [start, goal], radius=radius)
    best_dist = math.inf
    if hasattr(world, 'line_of_sight'):
        if world.line_of_sight((start[1], start[0]), (goal[1], goal[0]), radius=radius, step_size=0.5):
            best_dist = _euclidean(start, goal)
    best_i, best_j = None, None
    sd, si = start_conns
    gd, gi = goal_conns
    if len(si) > 0 and len(gi) > 0:
        apsp_sub = world.apsp[np.ix_(si, gi)]
        total_dists = sd[:, None] + apsp_sub + gd[None, :]
        min_idx = np.argmin(total_dists)
        min_d = total_dists.flat[min_idx]
        if min_d < best_dist:
            best_dist = min_d
            best_i = si[min_idx // len(gi)]
            best_j = gi[min_idx % len(gi)]
    if best_dist == math.inf:
        return []
    path = [start]
    if best_i is not None and best_j is not None:
        curr = best_j
        prm_path = []
        while curr != best_i and curr >= 0:
            prm_path.append(world.prm_nodes[curr])
            curr = int(world.apsp_pred[best_i, curr])
        prm_path.append(world.prm_nodes[best_i])
        prm_path.reverse()
        path.extend(prm_path)
    path.append(goal)
    return path

def dijkstra_multi(world, start, targets, radius=0.3):
    if not targets or not hasattr(world, 'apsp') or not hasattr(world, 'prm_node_idx'):
        return {}
    target_set = list(set(targets))
    all_nodes = [start] + target_set
    all_conns = _connect_temp_nodes_batch(world, all_nodes, radius=radius)
    sd, si = all_conns[0]
    results = {}
    #precalculate batch LOS for targets within 15 units
    direct_los = {t: False for t in target_set}
    if hasattr(world, 'batch_line_of_sight') and target_set:
        target_arr = np.array(target_set)
        start_xy = (start[1], start[0])
        dx = target_arr[:, 1] - start_xy[0]
        dy = target_arr[:, 0] - start_xy[1]
        dist = np.hypot(dx, dy)
        close_mask = dist <= 15.0
        if np.any(close_mask):
            close_targets = target_arr[close_mask]
            target_xy = np.column_stack((close_targets[:, 1], close_targets[:, 0]))
            los_res = world.batch_line_of_sight(start_xy, target_xy, radius=radius, step_size=0.5)
            close_indices = np.where(close_mask)[0]
            for i, is_los in zip(close_indices, los_res):
                direct_los[target_set[i]] = is_los        
    start_to_all_prm = None
    if len(si) > 0:
        start_to_all_prm = np.min(sd[:, None] + world.apsp[si, :], axis=0)
    for idx, t in enumerate(target_set):
        gd, gi = all_conns[idx + 1]
        best_dist = math.inf
        if direct_los[t]:
            best_dist = _euclidean(start, t)
        if start_to_all_prm is not None and len(gi) > 0:
            min_d = np.min(start_to_all_prm[gi] + gd)
            if min_d < best_dist:
                best_dist = min_d
        if start == t:
            best_dist = 0.0
        results[t] = (best_dist, [start, t] if best_dist != math.inf else [])
    return results

def next_step(world, start, goal):
    path = astar(world, start, goal)
    if len(path) >= 2:
        return path[1]
    return None

def astar_belief(belief_map, start: tuple, goal: tuple) -> list:
    """
    A* pathfinding on the ghost's personal belief topology (discovered map only).
    Uses belief_map._open_cells, _neighbours, and respects discovered wall dead zones.
    Returns a list of (y, x) waypoints from start to goal.
    """
    if start == goal:
        return [start]
    if not hasattr(belief_map, '_open_cells') or not belief_map._open_cells:
        return []
    start_idx = belief_map._closest_node(start)
    goal_idx = belief_map._closest_node(goal)
    if start_idx < 0 or goal_idx < 0:
        return []
    start_node = belief_map._open_cells[start_idx]
    goal_node = belief_map._open_cells[goal_idx]
    disabled_walls = getattr(belief_map, '_disabled_wall_nodes', set())
    if start_node in disabled_walls or goal_node in disabled_walls:
        if goal_node in disabled_walls:
            return []
    if start_node == goal_node:
        return [start, goal]
    open_set = []
    counter = 0
    start_h = _euclidean(start_node, goal_node)
    heapq.heappush(open_set, (start_h, counter, start_node))
    g_score = {start_node: 0.0}
    came_from = {}
    visited = set()
    found = False
    while open_set:
        f, _, current = heapq.heappop(open_set)
        if current == goal_node:
            found = True
            break
        if current in visited:
            continue
        visited.add(current)
        curr_g = g_score[current]
        neighbours = belief_map._neighbours.get(current, [])
        for nbr in neighbours:
            if nbr in disabled_walls or nbr in visited:
                continue
            step_cost = _euclidean(current, nbr)
            tentative_g = curr_g + step_cost
            if tentative_g < g_score.get(nbr, math.inf):
                came_from[nbr] = current
                g_score[nbr] = tentative_g
                h = _euclidean(nbr, goal_node)
                counter += 1
                heapq.heappush(open_set, (tentative_g + h, counter, nbr))
    if not found:
        return []
    path = _reconstruct(came_from, goal_node)
    full_path = []
    if _euclidean(start, path[0]) > 0.1:
        full_path.append(start)
    full_path.extend(path)
    if _euclidean(path[-1], goal) > 0.1:
        full_path.append(goal)
    return full_path

def find_topological_flee_target(world, ghost_pos: tuple, pac_pos: tuple, radius: float = 0.35) -> tuple[float, float] | None:
    """
    Computes a topologically safe escape node on the maze graph that:
    1. Maximizes maze graph distance from Pac-Man.
    2. Ensures a positive lead margin (ghost arrives before Pac-Man).
    3. Penalizes dead-end corridors (corridor degree 1).
    4. Rewards branching intersections/loops (corridor degree >= 3).
    """
    if world is None or not hasattr(world, 'apsp') or not hasattr(world, 'prm_nodes') or not world.prm_nodes:
        return None
    gy, gx = float(ghost_pos[0]), float(ghost_pos[1])
    py, px = float(pac_pos[0]), float(pac_pos[1])
    conns = _connect_temp_nodes_batch(world, [(gy, gx), (py, px)], radius=radius)
    gd, gi = conns[0]
    pd, pi = conns[1]
    if len(gi) == 0:
        return None
    d_ghost = np.min(gd[:, None] + world.apsp[gi, :], axis=0)
    if len(pi) > 0:
        d_pac = np.min(pd[:, None] + world.apsp[pi, :], axis=0)
    else:
        prm_arr = getattr(world, 'prm_nodes_arr', None)
        if prm_arr is not None and len(prm_arr) > 0:
            d_pac = np.hypot(prm_arr[:, 0] - py, prm_arr[:, 1] - px)
        else:
            return None
    degrees = np.array([len(world.prm_graph.get(n, [])) for n in world.prm_nodes], dtype=np.float32)
    deg_bonus = np.where(degrees >= 3, 8.0, np.where(degrees == 2, 0.0, -25.0))
    with np.errstate(invalid='ignore'):
        lead_margin = np.where((d_pac < math.inf) & (d_ghost < math.inf), d_pac - d_ghost, -np.inf)
        scores = d_pac * 2.0 + lead_margin * 1.5 - d_ghost * 0.4 + deg_bonus
        invalid = (lead_margin <= 0) | (d_ghost == math.inf) | (d_pac == math.inf)
        scores[invalid] = -np.inf
    best_idx = int(np.argmax(scores))
    if scores[best_idx] > -np.inf:
        node = world.prm_nodes[best_idx]
        return (float(node[0]), float(node[1]))
    with np.errstate(invalid='ignore'):
        fallback_scores = d_pac * 2.0 - d_ghost * 0.5 + deg_bonus
        fallback_scores[(d_ghost == math.inf) | (d_pac == math.inf)] = -np.inf
    if np.any(fallback_scores > -np.inf):
        best_fb = int(np.argmax(fallback_scores))
        node = world.prm_nodes[best_fb]
        return (float(node[0]), float(node[1]))
    return None

def _belief_base(belief_map):
    """Builds (once per topology) the full adjacency over the belief grid plus the
    per-edge endpoint arrays needed to mask it cheaply as walls are discovered."""
    if not hasattr(belief_map, '_open_cells') or not hasattr(belief_map, '_nbr_idx'):
        return None
    if hasattr(belief_map, '_ensure_initialised'):
        belief_map._ensure_initialised()
    n = len(belief_map._open_cells)
    if n == 0:
        return None
    base = getattr(belief_map, '_plan_base', None)
    if base is not None and base['n'] == n and base['nbr_id'] == id(belief_map._nbr_idx):
        return base
    nbr_idx = belief_map._nbr_idx
    rows = np.repeat(np.arange(n, dtype=np.int32), nbr_idx.shape[1])
    cols = nbr_idx.ravel()
    data = belief_map._nbr_dist.ravel().astype(np.float64)
    keep = (cols >= 0) & (data > 0)
    graph = csr_matrix((data[keep], (rows[keep], cols[keep])), shape=(n, n))
    graph.sum_duplicates()
    erows = np.repeat(np.arange(n, dtype=np.int32), np.diff(graph.indptr))
    base = {'n': n, 'nbr_id': id(nbr_idx), 'graph': graph, 'erows': erows, 'ecols': graph.indices.copy(),
            'masked': None, 'mask_stamp': -1, 'rows': {}, 'node_idx': {}}
    belief_map._plan_base = base
    return base

def _belief_csr(belief_map):
    """Adjacency over the ghost's DISCOVERED topology.

    The belief graph starts optimistic — every cell assumed open — and nodes are masked out
    as lidar confirms walls, so planning never uses a wall the ghost has not seen. The base
    graph is built once; discovery only re-applies a vectorised edge mask (O(E), ~0.1 ms).
    """
    base = _belief_base(belief_map)
    if base is None:
        return None
    walk = getattr(belief_map, '_walkable_mask', None)
    stamp = len(getattr(belief_map, '_disabled_wall_nodes', ()))
    if base['masked'] is not None and base['mask_stamp'] == stamp:
        return base['masked']
    g = base['graph'].copy()
    if walk is not None and len(walk) == base['n']:
        ok = walk[base['erows']] & walk[base['ecols']]
        g.data[~ok] = 0.0
        g.eliminate_zeros()   #scipy treats explicit zeros as weight-0 edges, so drop them
    base['masked'] = g
    base['mask_stamp'] = stamp
    return g

def _node_idx(belief_map, pos):
    """Nearest belief node for a coordinate, memoised — positions repeat heavily."""
    base = _belief_base(belief_map)
    key = (round(float(pos[0]), 2), round(float(pos[1]), 2))
    cache = base['node_idx'] if base is not None else None
    if cache is not None:
        idx = cache.get(key)
        if idx is not None:
            return idx
    idx = belief_map._closest_node(pos)
    if cache is not None and len(cache) < 20000:
        cache[key] = idx
    return idx

def _dist_row(belief_map, src_idx, with_pred=False, allow_stale=False):
    """Geodesic distances from one belief node to every node, cached per source.

    Navigation and bidding ask for a row computed under the current wall mask. CBBA bundle
    ordering passes allow_stale=True and accepts a row from a few discoveries ago — an
    ordering heuristic does not need to know about a wall found two frames back, and this
    is what keeps the per-pair leg cost at a dict lookup during early exploration.
    """
    g = _belief_csr(belief_map)
    if g is None:
        return None, None
    base = belief_map._plan_base
    rows, stamp = base['rows'], base['mask_stamp']
    hit = rows.get(src_idx)
    if hit is not None and (allow_stale or hit[2] == stamp) and (not with_pred or hit[1] is not None):
        return hit[0], hit[1]
    if with_pred:
        d, pred = _sp_dijkstra(g, directed=False, indices=src_idx, return_predecessors=True)
    else:
        d, pred = _sp_dijkstra(g, directed=False, indices=src_idx), None
    if len(rows) >= 384:
        rows.clear()
    rows[src_idx] = (d.astype(np.float32), pred, stamp)
    return rows[src_idx][0], pred

def _manhattan_dists(start, target_set):
    return {t: (abs(start[0] - t[0]) + abs(start[1] - t[1]), [start, t]) for t in target_set}

def dijkstra_multi_belief(belief_map, start, targets):
    """Multi-target shortest path over the ghost's own discovered map.

    Drop-in replacement for dijkstra_multi() — returns {target: (dist, path)} — but uses
    only information the ghost has sensed or been told, never the ground-truth world graph.
    """
    if not targets:
        return {}
    target_set = list(set(targets))
    if _belief_csr(belief_map) is None:
        return _manhattan_dists(start, target_set)
    start_idx = _node_idx(belief_map, start)
    if start_idx < 0:
        return _manhattan_dists(start, target_set)
    dist, _ = _dist_row(belief_map, start_idx)
    cells = belief_map._open_cells
    results = {}
    for t in target_set:
        ti = _node_idx(belief_map, t)
        if ti < 0 or not math.isfinite(dist[ti]):
            results[t] = (math.inf, [])
            continue
        #add the residual hop from the graph node to the exact target coordinate
        results[t] = (float(dist[ti]) + _euclidean(cells[ti], (float(t[0]), float(t[1]))), [start, t])
    if start in results:
        results[start] = (0.0, [start])
    return results

def leg_cost_belief(belief_map, a, b):
    """Geodesic cost between two arbitrary points on the discovered map (CBBA bundle legs)."""
    if _belief_csr(belief_map) is None:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    ia, ib = _node_idx(belief_map, a), _node_idx(belief_map, b)
    if ia < 0 or ib < 0:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    if ia == ib:
        return 0.0
    base = getattr(belief_map, '_plan_base', None)
    if base is not None:
        rows = base.get('rows', {})
        if ib in rows and ia not in rows:
            d = rows[ib][0][ia]
            return float(d) if math.isfinite(d) else abs(a[0] - b[0]) + abs(a[1] - b[1])
    dist, _ = _dist_row(belief_map, ia, allow_stale=True)
    d = dist[ib]
    return float(d) if math.isfinite(d) else abs(a[0] - b[0]) + abs(a[1] - b[1])

def path_belief(belief_map, start, goal):
    """Shortest waypoint path over the discovered map via Dijkstra predecessors.
    Returns [start, ..., goal] or [] if unreachable. Falls back to astar_belief."""
    if _belief_csr(belief_map) is None:
        return astar_belief(belief_map, start, goal)
    si, gi = _node_idx(belief_map, start), _node_idx(belief_map, goal)
    if si < 0 or gi < 0:
        return []
    if si == gi:
        return [start, goal]
    dist, pred = _dist_row(belief_map, si, with_pred=True)
    if not math.isfinite(dist[gi]):
        return []
    cells = belief_map._open_cells
    nodes = []
    cur = gi
    while cur != si and cur >= 0:
        nodes.append(cells[cur])
        cur = int(pred[cur])
    nodes.reverse()
    return [start] + [(float(n[0]), float(n[1])) for n in nodes] + [goal]

def find_topological_flee_target_belief(belief_map, ghost_pos: tuple, pac_pos: tuple):
    """Belief-space analogue of find_topological_flee_target.

    Picks the reachable node that maximises geodesic distance from Pacman while staying
    cheap for the ghost to reach, using only the discovered topology.
    """
    if _belief_csr(belief_map) is None:
        return None
    g_idx, p_idx = _node_idx(belief_map, ghost_pos), _node_idx(belief_map, pac_pos)
    if g_idx < 0 or p_idx < 0:
        return None
    d_ghost, _ = _dist_row(belief_map, g_idx)
    d_pac, _ = _dist_row(belief_map, p_idx)
    reachable = np.isfinite(d_ghost) & np.isfinite(d_pac)
    if not np.any(reachable):
        return None
    graph = _belief_csr(belief_map)
    deg_bonus = np.zeros(d_pac.shape)
    if graph is not None and graph.shape[0] == len(deg_bonus):
        deg = np.diff(graph.indptr)
        deg_bonus = np.where(deg >= 3, 4.0, np.where(deg <= 1, -12.0, 0.0))
    score = np.full(d_pac.shape, -np.inf)
    score[reachable] = d_pac[reachable] - 0.6 * d_ghost[reachable] + deg_bonus[reachable]
    lead = np.zeros(d_pac.shape, dtype=bool)
    lead[reachable] = d_pac[reachable] > FLEE_SPEED_RATIO * d_ghost[reachable] + 1.0
    if np.any(lead):
        safe = np.where(lead, score, -np.inf)
        best = int(np.argmax(safe))
    else:
        best = int(np.argmax(score))
    if not np.isfinite(score[best]):
        return None
    node = belief_map._open_cells[best]
    return (float(node[0]), float(node[1]))

def ghost_dists(ghost, start, targets):
    """Distances from `start` to `targets` over a ghost's own discovered topology.

    Kept module-level (rather than only as a Ghost method) so the allocator and CBBA stay
    decoupled from the Ghost class, and so agent stubs without a belief map still work.
    """
    bm = getattr(ghost, 'belief_map', None)
    if bm is None:
        return _manhattan_dists(start, list(set(targets)))
    return dijkstra_multi_belief(bm, start, list(targets))
