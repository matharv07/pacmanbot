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
        
    for origin in nodes_list:
        origin_tup = (round(origin[0], 2), round(origin[1], 2))
        if origin_tup in world._conn_cache:
            results.append(world._conn_cache[origin_tup])
            continue
            
        dy = prm_arr[:, 0] - origin[0]
        dx = prm_arr[:, 1] - origin[1]
        dist = np.hypot(dx, dy)
        valid_mask = dist <= 10.0
        
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) > 0:
            valid_nodes = prm_arr[valid_indices]
            valid_targets = np.column_stack((valid_nodes[:, 1], valid_nodes[:, 0]))
            
            if hasattr(world, 'batch_line_of_sight'):
                los = world.batch_line_of_sight(origin, valid_targets, radius=0.4)
            else:
                los = np.array([world.line_of_sight(origin, (t[0], t[1]), radius=0.4) for t in valid_targets])
                
            clear_indices = valid_indices[los]
            clear_dists = dist[clear_indices]
            res = (clear_dists, clear_indices)
        else:
            res = (np.array([]), np.array([], dtype=int))
            
        world._conn_cache[origin_tup] = res
        results.append(res)
        
    if len(world._conn_cache) > 20000:
        world._conn_cache.clear()
        
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
        if world.line_of_sight((start[1], start[0]), (goal[1], goal[0]), radius=0.4):
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
    
    for idx, t in enumerate(target_set):
        gd, gi = all_conns[idx + 1]
        best_dist = math.inf
        
        if hasattr(world, 'line_of_sight'):
            if world.line_of_sight((start[1], start[0]), (t[1], t[0]), radius=0.4):
                best_dist = _euclidean(start, t)
                
        if len(si) > 0 and len(gi) > 0:
            apsp_sub = world.apsp[np.ix_(si, gi)]
            total_dists = sd[:, None] + apsp_sub + gd[None, :]
            min_idx = np.argmin(total_dists)
            min_d = total_dists.flat[min_idx]
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