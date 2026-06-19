from __future__ import annotations
import math
from typing import Optional
import ast
from allocator import TaskType, Task, generate_tasks
from pathfinder import dijkstra_multi, astar

AUCTION_EVERY   = 6      #full auction every 0.6s 
LT              = 3
LAMBDA          = 0.99   #time decay factor

def _task_key(task: Task) -> tuple:
    return (int(task.task_type), task.target_pos, getattr(task, 'owner', -1))

class CBBA_Agent:
    def __init__(self, gid: int, lt: int = LT, lamda: float = LAMBDA):
        self.gid = gid
        self.lt = lt
        self.lamda = lamda
        self.bundle: list = []  #tasks in agent's bundle
        self.path: list = []    #agent's ordered tasks for execution
        self.y: dict = {}       #winning bids
        self.z: dict = {}       #task winners
        self.s: dict = {}       #last sync frames
        self._task_map: dict = {}
        self._last_auction: int = -1
        self._dist_cache: dict = {}   #(pos) -> distance, cached per-auction

    def step(self, ghost, frame: int) -> Optional[Task]:
        changed = False
        for key in list(self.z.keys()):
            winner = self.z[key]
            if winner is not None and winner != self.gid:
                if ghost.known_agents.get(winner) == "UNKNOWN":
                    self.y[key] = 0.0
                    self.z[key] = None
                    changed = True
        if changed:
            self._cascade_release()
        if frame - self._last_auction >= AUCTION_EVERY or not self.bundle:
            self._last_auction = frame
            tasks, dists = generate_tasks(ghost, frame)                
            self._task_map = {_task_key(t): t for t in tasks}
            self._phase1(ghost, tasks, dists)
        return self.get_active_task()

    def get_active_task(self) -> Optional[Task]:
        for key in self.path:
            task = self._task_map.get(key)
            if task is not None:
                return task
        return None

    def get_known_task_for(self, other_gid: int) -> Optional[Task]:
        best_task = None
        best_score = -math.inf
        for key, winner in self.z.items():
            if winner == other_gid:
                task = self._task_map.get(key)
                if task is None:
                    #reconstruct task from consensus if we didn't evaluate it locally
                    task_type, target_pos, owner = key
                    score = self.y.get(key, 0.0)
                    task = Task(task_type=task_type, target_pos=target_pos, score=score, owner=owner)
                if task and task.score > best_score:
                    best_score = task.score
                    best_task = task
        return best_task

    def get_consensus_payload(self) -> dict:    #forwards consensus instead of raw tasks
        return {"y": dict(self.y), "z": dict(self.z), "s": dict(self.s)}

    def receive_consensus(self, sender_gid: int, y_k: dict, z_k: dict, s_k: dict, frame: int) -> bool:
        #snapshot before updating to pass into _table1
        s_i_snapshot = dict(self.s)
        self.s[sender_gid] = max(self.s.get(sender_gid, -1), frame)
        for agent_id, ts in s_k.items():
            if isinstance(agent_id, str):
                agent_id = int(agent_id)
            self.s[agent_id] = max(self.s.get(agent_id, -1), ts)
        changed = False
        all_keys = set(self.y) | set(y_k)
        for key in all_keys:
            z_k_j = z_k.get(key)
            z_i_j = self.z.get(key)
            y_k_j = y_k.get(key, 0.0)
            y_i_j = self.y.get(key, 0.0)
            action = self._table1(sender_gid, z_k_j, z_i_j, y_k_j, y_i_j, s_k, s_i_snapshot)
            if action == "update":
                if y_k_j != y_i_j or z_k_j != z_i_j:
                    self.y[key] = y_k_j
                    self.z[key] = z_k_j
                    changed = True
            elif action == "reset":
                if y_i_j != 0.0 or z_i_j is not None:
                    self.y[key] = 0.0
                    self.z[key] = None
                    changed = True
        if changed:
            self._cascade_release()
        return changed

    def _phase1(self, ghost, tasks: list, dists: dict):
        valid_keys = {_task_key(t) for t in tasks}
        self._task_map.update({_task_key(t): t for t in tasks})
        self._dist_cache = {pos: d for pos, (d, _) in dists.items()}
        self._astar_cache = {}
        #pruning bundle & path: keeping only tasks we still own in the new task set
        new_bundle = []
        for k in self.bundle:
            if k in valid_keys and self.z.get(k) == self.gid and self._task_map.get(k) is not None:
                new_bundle.append(k)
            else:
                break 
        kept = set(new_bundle)
        self.bundle = new_bundle
        self.path   = [k for k in self.path if k in kept]
        #greedily adding tasks until bundle full or no valid candidate remains
        while len(self.bundle) < self.lt:
            best_key = None
            best_gain = 0.0
            best_n = 0
            for task in tasks:
                key = _task_key(task)
                if key in self.bundle:
                    continue
                gain, n = self._marginal_gain(key, ghost)
                if gain <= self.y.get(key, 0.0):
                    continue
                if gain > best_gain:
                    best_gain = gain
                    best_key = key
                    best_n = n
            if best_key is None:
                break
            self.bundle.append(best_key)
            self.path.insert(best_n, best_key)
            self.y[best_key] = best_gain
            self.z[best_key] = self.gid

    def _marginal_gain(self, key: tuple, ghost) -> tuple:
        task = self._task_map.get(key)
        if task is None:
            return 0.0, 0
        s_old = self._path_score(self.path, ghost)
        best_gain = -math.inf
        best_n = 0
        for n in range(len(self.path) + 1):
            new_path = self.path[:n] + [key] + self.path[n:]
            gain = self._path_score(new_path, ghost) - s_old
            if gain > best_gain:
                best_gain = gain
                best_n = n
        return max(best_gain, 0.0), best_n

    def _path_score(self, path: list, ghost) -> float:
        if not path:
            return 0.0
        cumulative = 0.0
        total = 0.0
        prev_pos = (ghost.row, ghost.col)
        for key in path:
            task = self._task_map.get(key)
            if task is None:
                continue
            tgt = task.target_pos
            if prev_pos == (ghost.row, ghost.col):
                d = self._dist_cache.get(tgt, math.inf)
            else:
                pair = (prev_pos, tgt)
                if pair not in getattr(self, '_astar_cache', {}):
                    path_to_tgt = astar(ghost.grid, prev_pos, tgt)
                    d = float(len(path_to_tgt) - 1) if path_to_tgt else math.inf
                    if not hasattr(self, '_astar_cache'):
                        self._astar_cache = {}
                    self._astar_cache[pair] = d
                d = self._astar_cache[pair]
            cumulative += d
            total += task.score * (self.lamda ** cumulative)
            prev_pos = tgt
        return total

    def _table1(self, k: int, z_kj, z_ij, y_kj: float, y_ij: float, s_k: dict, s_i: dict) -> str:
        i = self.gid
        def sk(a): return s_k.get(a, s_k.get(str(a), -1))  #k's timestamp for agent a
        def si(a): return s_i.get(a, s_i.get(str(a), -1))  #i's timestamp for agent a
        if z_kj == k:           #sender claims self as winner
            if z_ij == i: return "update" if y_kj > y_ij else "leave"
            elif z_ij == k: return "update" if sk(k) > si(k) else "leave"
            elif z_ij is None: return "update"
            else: return "update" if y_kj > y_ij else "leave"  #z_ij == m
        elif z_kj == i:         #sender claims receiver as winner
            if z_ij == i: return "leave"
            elif z_ij == k: return "reset"
            elif z_ij is None: return "update"
            else: return "update" if sk(i) > si(i) else "leave"  #z_ij == m
        elif z_kj is None:      #sender says unassigned
            if z_ij == i: return "leave"
            elif z_ij == k: return "update" if sk(k) > si(k) else "leave"
            elif z_ij is None: return "leave"
            else: return "update" if sk(z_ij) > si(z_ij) else "leave"  #z_ij == m
        else:                   #sender claims third agent m as winner
            m = z_kj
            if z_ij == i: return "update" if sk(m) > si(m) else "leave"
            elif z_ij == k: return "update" if sk(m) > si(m) else "leave"
            elif z_ij is None: return "update"
            elif z_ij == m: return "update" if sk(m) > si(m) else "leave"  #same m
            else: return "update" if y_kj > y_ij else "leave"   #different m'

    def _cascade_release(self):
        n_bar = None
        for n, key in enumerate(self.bundle):
            if self.z.get(key) != self.gid:
                n_bar = n
                break
        if n_bar is None:
            return
        kept = set(self.bundle[:n_bar])
        for key in self.bundle[n_bar:]:
            self.y[key] = 0.0
            self.z[key] = None
        self.path   = [k for k in self.path if k in kept]  #filter in one pass, preserves order
        self.bundle = self.bundle[:n_bar]