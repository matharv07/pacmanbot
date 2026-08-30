import heapq
import math
import numpy as np

def _euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def _connect_temp_nodes_batch(world, nodes_list):
    results = []
    prm_arr = world.prm_nodes_arr
    if prm_arr is None or len(prm_arr) == 0:
        return [(np.array([]), np.array([], dtype=int)) for _ in nodes_list]
    if not hasattr(world, '_conn_cache'):
        world._conn_cache = {}
    misses = []
    results_map = {}
    for i, origin in enumerate(nodes_list):
        origin_tup = (round(origin[0], 2), round(origin[1], 2))
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
                p1_arr = np.full((len(valid_targets), 2), origin, dtype=np.float32)
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
                los = world.batch_line_of_sight_pairs(p1s, p2s, radius=0.4, step_size=0.5)
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
    if len(world._conn_cache) > 20000:
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

def astar(world, start, goal):
    if start == goal:
        return [start]
    if not hasattr(world, 'apsp') or not hasattr(world, 'prm_node_idx'):
        return []
    start_conns, goal_conns = _connect_temp_nodes_batch(world, [start, goal])
    best_dist = math.inf
    if hasattr(world, 'line_of_sight'):
        if world.line_of_sight((start[1], start[0]), (goal[1], goal[0]), radius=0.4, step_size=0.5):
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

def dijkstra_multi(world, start, targets):
    if not targets or not hasattr(world, 'apsp') or not hasattr(world, 'prm_node_idx'):
        return {}
    target_set = list(set(targets))
    all_nodes = [start] + target_set
    all_conns = _connect_temp_nodes_batch(world, all_nodes)
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
            los_res = world.batch_line_of_sight(start_xy, target_xy, radius=0.4, step_size=0.5)
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