from __future__ import annotations
import math
from typing import Optional
from allocator import TaskType, Task, generate_tasks

AUCTION_EVERY   = 5      #full auction every 0.5s
LT              = 3
LAMBDA          = 0.95   #time decay factor

def _manhattan(a: tuple, b: tuple) -> float:
    return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))

def _task_key(task: Task) -> tuple:
    return (int(task.task_type), task.target_pos)

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

    def step(self, ghost, frame: int) -> Optional[Task]:
        if frame - self._last_auction >= AUCTION_EVERY or not self.bundle:
            self._last_auction = frame
            tasks = generate_tasks(ghost, frame)
            self._task_map = {_task_key(t): t for t in tasks}
            self._phase1(ghost, tasks)
        return self.get_active_task()

    def get_active_task(self) -> Optional[Task]:
        for key in self.path:
            task = self._task_map.get(key)
            if task is not None:
                return task
        return None

    def get_consensus_payload(self) -> dict:    #forwards consensus instead of raw tasks
        return {"y": {str(k): v for k, v in self.y.items()}, "z": {str(k): v for k, v in self.z.items()}, "s": dict(self.s)}

    def receive_consensus(self, sender_gid: int, y_k: dict, z_k: dict, s_k: dict, frame: int) -> bool:
        y_k = {eval(k): v for k, v in y_k.items()}
        z_k = {eval(k): v for k, v in z_k.items()}
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
            action = self._table1(sender_gid, z_k_j, z_i_j, y_k_j, y_i_j, s_k, self.s)
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

    def _phase1(self, ghost, tasks: list):
        valid_keys = {_task_key(t) for t in tasks}
        self._task_map.update({_task_key(t): t for t in tasks})
        #pruning bundle & path: keeping only tasks we still own in the new task set
        self.bundle = [k for k in self.bundle if k in valid_keys and self.z.get(k) == self.gid]
        self.path = [k for k in self.path if k in self.bundle]
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
            d = _manhattan(prev_pos, task.target_pos)
            cumulative += d
            total += task.score * (self.lamda ** cumulative)
            prev_pos = task.target_pos
        return total

    def _table1(self, k: int, z_kj, z_ij, y_kj: float, y_ij: float, s_k: dict, s_i: dict) -> str:
        #returns update / reset / leave
        i = self.gid
        def s_kj(agent): return s_k.get(agent, s_k.get(str(agent), -1))
        def s_ij(agent): return s_i.get(agent, -1)
        if z_kj == k:
            if z_ij == k:
                return "update" if s_kj(k) > s_ij(k) else "leave"
            elif z_ij == i:
                return "update"
            else:
                return "update" if y_kj > y_ij else "leave"
        elif z_kj == i:
            if z_ij == k:
                return "reset"
            elif z_ij == i:
                return "leave"
            else:
                return "leave"
        elif z_kj is None:
            return "update" if y_kj > y_ij else "leave"
        else:
            if z_ij == k:
                return "update" if y_kj > y_ij else "leave"
            elif z_ij == i:
                return "reset" if s_kj(i) > s_ij(i) else "leave"
            elif z_ij == z_kj:
                return "update" if y_kj > y_ij else "leave"
            else:
                return "update" if y_kj > y_ij else "leave"

    def _cascade_release(self):
        n_bar = None
        for n, key in enumerate(self.bundle):
            if self.z.get(key) != self.gid:
                n_bar = n
                break
        if n_bar is None:
            return
        for key in self.bundle[n_bar:]:
            self.y[key] = 0.0
            self.z[key] = None
            if key in self.path:
                self.path.remove(key)
        self.bundle = self.bundle[:n_bar]