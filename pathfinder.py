import heapq
import math

def _euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def _connect_temp_node(world, node):
    connections = []
    # Find up to 10 closest nodes in PRM to connect to
    if not hasattr(world, 'prm_nodes'): return []
    dists = []
    for n in world.prm_nodes:
        dists.append((_euclidean(node, n), n))
    dists.sort()
    for d, n in dists[:15]:
        if d < 7.0 and world.line_of_sight(node, n, radius=0.4):
            connections.append((d, n))
    return connections

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
        
    start_conns = _connect_temp_node(world, start)
    goal_conns = _connect_temp_node(world, goal)
    
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
        
    start_conns = _connect_temp_node(world, start)
    target_set = set(targets)
    
    # Precompute goal connections for fast lookup
    target_conns_map = {}
    for t in target_set:
        target_conns_map[t] = _connect_temp_node(world, t)
        
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