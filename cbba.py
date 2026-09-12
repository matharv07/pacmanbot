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
    pos = (round(float(task.target_pos[0]), 1), round(float(task.target_pos[1]), 1))
    return (int(task.task_type), pos)

class CBBA_Agent:
    def __init__(self, gid: int, lt: int = LT, lamda: float = LAMBDA):
        self.gid = gid
        self.lt = lt
        self.lamda = lamda
        self.bundle: list = []             #tasks in agent's bundle
        self.path: list = []               #agent's ordered tasks for execution
        self.y: dict = {}                  #winning bids
        self.z: dict = {}                  #task winners
        self.s: dict = {}                  #last sync frames
        self._task_map: dict = {}
        self._task_created_frame: dict = {}
        self._last_auction: int = -gid     #stagger initial auction frames
        self._dist_cache: dict = {}        #(pos) -> distance, cached per-auction
        self._astar_cache: dict = {}       #persists across auctions
        self._unreachable_cache: dict = {} #(pos) -> timeout_frame

    def reset_caches(self):
        self._dist_cache.clear()
        self._astar_cache.clear()

    def mark_unreachable(self, target_pos: tuple, frame: int):
        self._unreachable_cache[target_pos] = frame + 150  #5 seconds penalty

    def step(self, ghost, frame: int) -> Optional[Task]:
        changed = False
        for pos, timeout in list(self._unreachable_cache.items()):
            if frame > timeout:
                del self._unreachable_cache[pos]
        for key in list(self.y.keys()):
            if frame - self._task_created_frame.get(key, frame) > 60:
                self.y.pop(key, None)
                self.z.pop(key, None)
                self._task_map.pop(key, None)
                self._task_created_frame.pop(key, None)
                changed = True
        for key in list(self.z.keys()):
            winner = self.z[key]
            if winner is not None and winner != self.gid:
                orphan_task = False
                if hasattr(ghost, 'is_agent_dead') and ghost.is_agent_dead(winner):
                    orphan_task = True
                elif hasattr(ghost, 'dead_agents') and winner in ghost.dead_agents:
                    orphan_task = True
                elif ghost.known_agents.get(winner) == "UNKNOWN":
                    silence = frame - ghost.last_heartbeat.get(winner, -1)
                    if silence > 100:
                        orphan_task = True
                    elif silence > 40:
                        self.y[key] *= 0.95
                if orphan_task:
                    self.y[key] = 0.0
                    self.z[key] = None
                    changed = True
                    self._last_auction = -1  #trigger auction re-evaluation to adopt orphaned task immediately
        if getattr(ghost, 'pacman_powered', False):
            hunt_keys = [k for k in list(self.bundle) if k[0] == TaskType.HUNT]
            if hunt_keys:
                for hk in hunt_keys:
                    self.bundle.remove(hk)
                    if hk in self.path:
                        self.path.remove(hk)
                    self.y[hk] = 0.0
                    self.z[hk] = None
                changed = True
        if changed:
            self._cascade_release()
        if (frame + self.gid) % AUCTION_EVERY == 0 and frame != self._last_auction:
            self._last_auction = frame
            tasks, dists = generate_tasks(ghost, frame)
            for t in tasks:
                k = _task_key(t)
                self._task_map[k] = t
                if k not in self._task_created_frame:
                    self._task_created_frame[k] = frame
            self._phase1(ghost, tasks, dists)
        return self.get_active_task()

    def get_active_task(self) -> Optional[Task]:
        for key in self.path:
            task = self._task_map.get(key)
            if task is not None:
                return task
        return None

    def remove_task(self, task):
        if task is None:
            return
        if hasattr(task, 'task_type') and hasattr(task, 'target_pos'):
            t_type = int(task.task_type)
            ty, tx = float(task.target_pos[0]), float(task.target_pos[1])
            self.path = [k for k in self.path if not (k[0] == t_type and abs(float(k[1][0]) - ty) < 0.15 and abs(float(k[1][1]) - tx) < 0.15)]
            self.bundle = [k for k in self.bundle if not (k[0] == t_type and abs(float(k[1][0]) - ty) < 0.15 and abs(float(k[1][1]) - tx) < 0.15)]
        elif isinstance(task, (tuple, list)) and len(task) >= 2:
            t_type = int(task[0])
            ty, tx = float(task[1][0]), float(task[1][1])
            self.path = [k for k in self.path if not (k[0] == t_type and abs(float(k[1][0]) - ty) < 0.15 and abs(float(k[1][1]) - tx) < 0.15)]
            self.bundle = [k for k in self.bundle if not (k[0] == t_type and abs(float(k[1][0]) - ty) < 0.15 and abs(float(k[1][1]) - tx) < 0.15)]

    def get_known_task_for(self, other_gid: int) -> Optional[Task]:
        best_task = None
        best_score = -math.inf
        for key, winner in self.z.items():
            if winner == other_gid:
                task = self._task_map.get(key)
                if task is None:
                    task_type, target_pos = key[0], key[1]
                    score = self.y.get(key, 0.0)
                    task = Task(task_type=task_type, target_pos=target_pos, score=score, owner=winner if winner is not None else -1)
                if task and task.score > best_score:
                    best_score = task.score
                    best_task = task
        return best_task

    def get_consensus_payload(self) -> dict:  #forwards consensus instead of raw tasks
        meta = {}
        for k, t in self._task_map.items():
            if t is not None:
                meta[k] = (getattr(t, 'assigned_to', -1), getattr(t, 'owner', -1), getattr(t, 'target_speed', 1.0))
        return {"y": dict(self.y), "z": dict(self.z), "s": dict(self.s), "meta": meta}

    def receive_consensus(self, sender_gid: int, y_k: dict, z_k: dict, s_k: dict, frame: int, task_meta: dict = None) -> bool:  #snapshot before updating to pass into _table1
        s_i_snapshot = dict(self.s)
        self.s[sender_gid] = max(self.s.get(sender_gid, -1), frame)
        for agent_id, ts in s_k.items():
            if isinstance(agent_id, str):
                agent_id = int(agent_id)
            self.s[agent_id] = max(self.s.get(agent_id, -1), ts)
        changed = False
        all_keys = set(self.y) | set(y_k)
        for key in all_keys:
            t_meta = task_meta.get(key) if task_meta else None
            assigned_to = t_meta[0] if t_meta else -1
            owner = t_meta[1] if t_meta else -1
            speed = t_meta[2] if t_meta else 1.0
            if key not in self._task_map:
                t_type, t_pos = key[0], key[1]
                score = max(y_k.get(key, 0.0), self.y.get(key, 0.0), 1.0)
                self._task_map[key] = Task(task_type=TaskType(t_type), target_pos=t_pos, score=score, assigned_to=assigned_to, owner=owner, target_speed=speed, created_frame=frame)
                self._task_created_frame[key] = frame
            else:
                existing = self._task_map[key]
                if existing.assigned_to == -1 and assigned_to != -1:
                    existing.assigned_to = assigned_to
                if existing.owner == -1 and owner != -1:
                    existing.owner = owner
            z_k_j = z_k.get(key)
            z_i_j = self.z.get(key)
            y_k_j = y_k.get(key, 0.0)
            y_i_j = self.y.get(key, 0.0)
            if z_k_j == z_i_j and y_k_j == y_i_j:
                continue
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
        candidate_keys = set()
        candidate_tasks = []
        for t in tasks:
            k = _task_key(t)
            candidate_keys.add(k)
            candidate_tasks.append(t)
            self._task_map[k] = t
            if k not in self._task_created_frame:
                self._task_created_frame[k] = ghost.frame
        #include active orphaned or outbiddable tasks from self._task_map
        for k, t in list(self._task_map.items()):
            if k not in candidate_keys:
                winner = self.z.get(k)
                if winner is None or winner != self.gid:
                    if ghost.frame - self._task_created_frame.get(k, ghost.frame) <= 60:
                        candidate_tasks.append(t)
                        candidate_keys.add(k)
        self._dist_cache = {(round(float(pos[0]), 2), round(float(pos[1]), 2)): d for pos, (d, _) in dists.items()}
        missing = [t.target_pos for t in candidate_tasks if (round(float(t.target_pos[0]), 2), round(float(t.target_pos[1]), 2)) not in self._dist_cache]
        if missing:
            if getattr(ghost, 'world', None) is not None and hasattr(ghost.world, 'apsp'):
                new_dists = dijkstra_multi(ghost.world, (ghost.y, ghost.x), missing)
                for pos, (d, _) in new_dists.items():
                    self._dist_cache[(round(float(pos[0]), 2), round(float(pos[1]), 2))] = d
            for t in candidate_tasks:
                ck = (round(float(t.target_pos[0]), 2), round(float(t.target_pos[1]), 2))
                if ck not in self._dist_cache:
                    self._dist_cache[ck] = abs(ghost.y - t.target_pos[0]) + abs(ghost.x - t.target_pos[1])
        #pruning bundle & path: keeping only tasks we still own that are valid
        new_bundle = []
        for k in self.bundle:
            if (k in candidate_keys or k in self._task_map) and self.z.get(k) == self.gid and self._task_map.get(k) is not None:
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
            for task in candidate_tasks:
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
        if self._unreachable_cache.get(task.target_pos, -1) > ghost.frame:
            return 0.0, 0
        #distance horizon gate: distant ghosts should not abandon quadrant for remote peer hunt/cutoff tasks unless explicitly designated (assigned_to == ghost.gid) or self-owned
        if task.task_type == TaskType.HUNT and task.assigned_to != ghost.gid:
            if task.owner != ghost.gid and task.owner != -1:
                tgt = task.target_pos
                cache_key = (round(float(tgt[0]), 2), round(float(tgt[1]), 2))
                d_tgt = self._dist_cache.get(cache_key)
                if d_tgt is None:
                    d_tgt = math.hypot(ghost.y - tgt[0], ghost.x - tgt[1])
                horizon = 22.0
                if hasattr(ghost, 'world') and ghost.world is not None:
                    dim = max(getattr(ghost.world, 'height', 20), getattr(ghost.world, 'width', 20))
                    horizon = max(22.0, dim * 0.65)
                if d_tgt > horizon:
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
        if best_gain > 0.0 and task.assigned_to != -1 and task.assigned_to == ghost.gid:
            best_gain *= 1.3
        return max(best_gain, 0.0), best_n

    def _path_score(self, path: list, ghost) -> float:
        if not path:
            return 0.0
        cumulative = 0.0
        total = 0.0
        prev_pos = (ghost.y, ghost.x)
        for key in path:
            task = self._task_map.get(key)
            if task is None:
                continue
            tgt = task.target_pos
            if prev_pos == (ghost.y, ghost.x):
                cache_key = (round(float(tgt[0]), 2), round(float(tgt[1]), 2))
                d = self._dist_cache.get(cache_key)
                if d is None or d == math.inf:
                    d = math.hypot(ghost.y - tgt[0], ghost.x - tgt[1])
            else:
                r1, c1 = int(round(prev_pos[0])), int(round(prev_pos[1]))
                r2, c2 = int(round(tgt[0])), int(round(tgt[1]))
                if r1 == r2 and c1 == c2:
                    d = 0.0
                elif abs(r1 - r2) + abs(c1 - c2) == 1:
                    d = 1.0
                else:
                    w = getattr(ghost, 'world', None)
                    if w is not None and hasattr(w, 'apsp') and hasattr(w, 'prm_node_idx'):
                        p1 = (round(float(r1), 1), round(float(c1), 1))
                        p2 = (round(float(r2), 1), round(float(c2), 1))
                        idx1 = w.prm_node_idx.get(p1)
                        idx2 = w.prm_node_idx.get(p2)
                        if idx1 is not None and idx2 is not None:
                            d = float(w.apsp[idx1, idx2])
                        else:
                            cache_key = (r1, c1, r2, c2)
                            if not hasattr(w, '_pair_dist_cache'):
                                w._pair_dist_cache = {}
                            d = w._pair_dist_cache.get(cache_key)
                            if d is None:
                                d = float(abs(r1 - r2) + abs(c1 - c2))
                                w._pair_dist_cache[cache_key] = d
                    else:
                        d = float(abs(r1 - r2) + abs(c1 - c2))
            cumulative += d
            total += task.score * (self.lamda ** cumulative)
            prev_pos = tgt
        return total

    def _table1(self, k: int, z_kj, z_ij, y_kj: float, y_ij: float, s_k: dict, s_i: dict) -> str:
        i = self.gid
        if z_kj == k:           #sender claims self as winner
            if z_ij == i: return "update" if y_kj > y_ij else "leave"
            elif z_ij == k: 
                sk_k = s_k.get(k, s_k.get(str(k), -1))
                si_k = s_i.get(k, s_i.get(str(k), -1))
                return "update" if sk_k > si_k else "leave"
            elif z_ij is None: return "update"
            else: return "update" if y_kj > y_ij else "leave"  #z_ij == m
        elif z_kj == i:         #sender claims receiver as winner
            if z_ij == i: return "leave"
            elif z_ij == k: return "reset"
            elif z_ij is None: return "update"
            else: 
                sk_i = s_k.get(i, s_k.get(str(i), -1))
                si_i = s_i.get(i, s_i.get(str(i), -1))
                return "update" if sk_i > si_i else "leave"  #z_ij == m
        elif z_kj is None:      #sender says unassigned
            if z_ij == i: return "leave"
            elif z_ij == k: 
                sk_k = s_k.get(k, s_k.get(str(k), -1))
                si_k = s_i.get(k, s_i.get(str(k), -1))
                return "update" if sk_k > si_k else "leave"
            elif z_ij is None: return "leave"
            else: 
                sk_zij = s_k.get(z_ij, s_k.get(str(z_ij), -1))
                si_zij = s_i.get(z_ij, s_i.get(str(z_ij), -1))
                return "update" if sk_zij > si_zij else "leave"  #z_ij == m
        else:                   #sender claims third agent m as winner
            m = z_kj
            if z_ij == i or z_ij == k or z_ij == m:
                sk_m = s_k.get(m, s_k.get(str(m), -1))
                si_m = s_i.get(m, s_i.get(str(m), -1))
                return "update" if sk_m > si_m else "leave"
            elif z_ij is None: return "update"
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
        for key in self.bundle[n_bar + 1:]:
            self.y[key] = 0.0
            self.z[key] = None
        self.path   = [k for k in self.path if k in kept]  #filter in one pass, preserves order
        self.bundle = self.bundle[:n_bar]