import heapq
import math
import numpy as np

def _euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def _connect_temp_node(world, node):
    """Vectorized PRM connection: batch distance filter + batch LOS check."""
    if not hasattr(world, 'prm_nodes') or not world.prm_nodes:
        return []
    nodes_arr = np.array(world.prm_nodes, dtype=np.float32)
    node_arr = np.array(node, dtype=np.float32)
    dists = np.sqrt(np.sum((nodes_arr - node_arr) ** 2, axis=1))
    # Pre-filter by distance threshold
    within = dists < 7.0
    if not np.any(within):
        return []
    valid_idx = np.nonzero(within)[0]
    valid_dists = dists[valid_idx]
    # Sort by distance, take top 15
    order = np.argsort(valid_dists)[:15]
    candidate_idx = valid_idx[order]
    candidate_dists = valid_dists[order]
    candidate_nodes = nodes_arr[candidate_idx]
    # Batch LOS check: nodes are (y, x), LOS expects (x, y)
    targets = np.column_stack((candidate_nodes[:, 1], candidate_nodes[:, 0]))
    origin = (node[1], node[0])  # (x, y)
    if hasattr(world, 'batch_line_of_sight'):
        los = world.batch_line_of_sight(origin, targets, radius=0.4)
    else:
        los = np.array([world.line_of_sight(origin, (t[0], t[1]), radius=0.4) for t in targets])
    connections = []
    for i, is_vis in enumerate(los):
        if is_vis:
            n = world.prm_nodes[candidate_idx[i]]
            connections.append((float(candidate_dists[i]), n))
    return connections

def _connect_temp_nodes_batch(world, nodes_list):
    """Batch connect multiple temp nodes at once — avoids redundant array creation."""
    if not hasattr(world, 'prm_nodes') or not world.prm_nodes:
        return [[] for _ in nodes_list]
    prm_arr = np.array(world.prm_nodes, dtype=np.float32)
    results = []
    for node in nodes_list:
        node_arr = np.array(node, dtype=np.float32)
        dists = np.sqrt(np.sum((prm_arr - node_arr) ** 2, axis=1))
        within = dists < 7.0
        if not np.any(within):
            results.append([])
            continue
        valid_idx = np.nonzero(within)[0]
        valid_dists = dists[valid_idx]
        order = np.argsort(valid_dists)[:15]
        candidate_idx = valid_idx[order]
        candidate_dists = valid_dists[order]
        candidate_nodes = prm_arr[candidate_idx]
        targets = np.column_stack((candidate_nodes[:, 1], candidate_nodes[:, 0]))
        origin = (node[1], node[0])
        if hasattr(world, 'batch_line_of_sight'):
            los = world.batch_line_of_sight(origin, targets, radius=0.4)
        else:
            los = np.array([world.line_of_sight(origin, (t[0], t[1]), radius=0.4) for t in targets])
        connections = []
        for i, is_vis in enumerate(los):
            if is_vis:
                n = world.prm_nodes[candidate_idx[i]]
                connections.append((float(candidate_dists[i]), n))
        results.append(connections)
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
    if not hasattr(world, 'prm_graph'):
        return []
        
    start_conns, goal_conns = _connect_temp_nodes_batch(world, [start, goal])
    
    # Ad-hoc graph access function
    def get_neighbors(n):
        if n == start:
            return [(d, tgt) for d, tgt in start_conns]
        elif n == goal:
            return []
        else:
            nbrs = []
            for tgt in world.prm_graph.get(n, []):
                nbrs.append((_euclidean(n, tgt), tgt))
            # Also check if it connects to goal
            for d, tgt in goal_conns:
                if tgt == n:
                    nbrs.append((d, goal))
            return nbrs

    g_score = {start: 0.0}
    came_from = {}
    open_heap = [(0.0 + _euclidean(start, goal), 0.0, start)]
    closed = set()
    
    while open_heap:
        f, g, node = heapq.heappop(open_heap)
        if node in closed:
            continue
        closed.add(node)
        if node == goal:
            return _reconstruct(came_from, node)
            
        for step_cost, neighbour in get_neighbors(node):
            if neighbour in closed:
                continue
            tentative_g = g + step_cost
            if tentative_g < g_score.get(neighbour, math.inf):
                g_score[neighbour] = tentative_g
                came_from[neighbour] = node
                f_new = tentative_g + _euclidean(neighbour, goal)
                heapq.heappush(open_heap, (f_new, tentative_g, neighbour))
                
    return []

def dijkstra_multi(world, start, targets):
    if not targets or not hasattr(world, 'prm_graph'):
        return {}
        
    target_set = set(targets)
    
    # Batch all temp node connections at once
    all_nodes = [start] + list(target_set)
    all_conns = _connect_temp_nodes_batch(world, all_nodes)
    start_conns = all_conns[0]
    target_conns_map = {}
    for i, t in enumerate(target_set):
        target_conns_map[t] = all_conns[1 + i]
        
    # Map PRM nodes to the targets they connect to
    prm_to_targets = {}
    for t, conns in target_conns_map.items():
        for d, prm_node in conns:
            if prm_node not in prm_to_targets:
                prm_to_targets[prm_node] = []
            prm_to_targets[prm_node].append((d, t))
            
    def get_neighbors(n):
        if n == start:
            return [(d, tgt) for d, tgt in start_conns]
        if n in target_set:
            return []
            
        nbrs = []
        for tgt in world.prm_graph.get(n, []):
            nbrs.append((_euclidean(n, tgt), tgt))
            
        # Connect to targets if in range
        if n in prm_to_targets:
            for d, t in prm_to_targets[n]:
                nbrs.append((d, t))
                
        return nbrs

    results = {t: (math.inf, []) for t in target_set}
    remaining = set(target_set)
    
    if start in target_set:
        results[start] = (0.0, [start])
        remaining.discard(start)
        
    if not remaining:
        return results
        
    dist = {start: 0.0}
    came_from = {}
    open_heap = [(0.0, start)]
    closed = set()
    
    while open_heap and remaining:
        d, node = heapq.heappop(open_heap)
        if node in closed:
            continue
        closed.add(node)
        
        if node in remaining:
            path = _reconstruct(came_from, node)
            results[node] = (d, path)
            remaining.discard(node)
            if not remaining:
                break
                
        for step_cost, neighbour in get_neighbors(node):
            if neighbour in closed:
                continue
            tentative = d + step_cost
            if tentative < dist.get(neighbour, math.inf):
                dist[neighbour] = tentative
                came_from[neighbour] = node
                heapq.heappush(open_heap, (tentative, neighbour))
                
    return results

def next_step(world, start, goal):
    path = astar(world, start, goal)
    if len(path) >= 2:
        return path[1]
    return None