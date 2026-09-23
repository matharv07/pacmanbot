"""
Curriculum Learning Scheduler.

Stages (re-laid 2026-09-20 after runs 13/14 showed the 3- and 4-ghost boards are the HARDEST for the swarm
and taught nothing measurable in 325 updates):
  Stage 0: 13x17 grid, 2 ghosts  - solo skills: survive the powered phase, deny/convert pellets, track Pacman --- a kill needs cooperation (Pacman is 2x faster), so this stage is gated on time, not kill rate
  Stage 1: 21x27 grid, 5 ghosts  - first swarm stage: enough ghosts for encirclement / mesh / flank terms to fire
  Stage 2: 27x33 grid, 6 ghosts  - full swarm pressure
  Stage 3: 33x41 grid, 7 ghosts  - full game (final)

Advancement is triggered when the rolling kill rate over ADVANCE_WINDOW updates reaches the
stage target, or plateaus above 0.92x the target. Mean return is recorded for logging only.
"""

from __future__ import annotations
from dataclasses import dataclass
import collections

@dataclass(frozen=True)
class Stage:
    world_height: float
    world_width: float
    obs_resolution: float
    n_ghosts: int
    n_power: int
    advance_return: float          #logging/reference only; gates use kill rate (see should_advance)
    min_updates: int
    target_kill_rate: float

    @property
    def rows(self) -> int:
        return int(self.world_height * self.obs_resolution)
        
    @property
    def cols(self) -> int:
        return int(self.world_width * self.obs_resolution)

STAGES = [Stage(world_height=13, world_width=17, obs_resolution=1.0, n_ghosts=2, n_power=3,  advance_return=0.0, min_updates=60,  target_kill_rate=0.0),
          Stage(world_height=21, world_width=27, obs_resolution=1.0, n_ghosts=5, n_power=12, advance_return=1.5, min_updates=100, target_kill_rate=0.78),
          Stage(world_height=27, world_width=33, obs_resolution=1.0, n_ghosts=6, n_power=20, advance_return=1.5, min_updates=120, target_kill_rate=0.82),
          Stage(world_height=33, world_width=41, obs_resolution=1.0, n_ghosts=7, n_power=28, advance_return=float('inf'), min_updates=50000, target_kill_rate=0.90)]

ADVANCE_WINDOW = 40    #rolling window of updates for advancement checks

class CurriculumScheduler:
    def __init__(self, start_stage: int = 0):
        self.stage_idx = min(len(STAGES) - 1, start_stage)
        self._return_history: collections.deque = collections.deque(maxlen=ADVANCE_WINDOW)
        self._kill_history: collections.deque = collections.deque(maxlen=ADVANCE_WINDOW)
        self._updates_in_stage: int = 0

    @property
    def stage(self) -> Stage:
        return STAGES[self.stage_idx]

    @property
    def is_final(self) -> bool:
        return self.stage_idx >= len(STAGES) - 1

    def record_return(self, mean_return: float | None, kill_rate: float | None = None):
        if mean_return is not None and kill_rate is not None:
            self._return_history.append(mean_return)
            self._kill_history.append(kill_rate)
        self._updates_in_stage += 1

    def should_advance(self) -> bool:
        if self.is_final:
            return False
        #strictly enforce min_updates before allowing ANY stage advancement
        if self._updates_in_stage < self.stage.min_updates:
            return False
        if len(self._kill_history) < ADVANCE_WINDOW:
            return False
        avg_kill = sum(self._kill_history) / len(self._kill_history)
        avg_ret = sum(self._return_history) / len(self._return_history)
        if avg_kill >= self.stage.target_kill_rate:
            print(f"Curriculum advancing: kill {avg_kill:.1%} >= target {self.stage.target_kill_rate:.1%} (avg ret: {avg_ret:.2f})")
            return True
        #plateau: kill rate has flattened above the competency bar (0.92x target)
        if self._updates_in_stage >= self.stage.min_updates + 2 * ADVANCE_WINDOW:
            half = ADVANCE_WINDOW // 2
            hist = list(self._kill_history)
            kill_first = sum(hist[:half]) / half
            kill_second = sum(hist[half:]) / half
            competency_kill = self.stage.target_kill_rate * 0.92
            if (kill_second - kill_first) < 0.02 and avg_kill >= competency_kill:
                print(f"Curriculum advancing due to plateau: kill progress {kill_second - kill_first:+.3f} < 0.02 (kill: {avg_kill:.1%} >= {competency_kill:.1%}, avg ret: {avg_ret:.2f})")
                return True
        return False

    def advance(self):
        if self.is_final:
            return
        self.stage_idx += 1
        self._return_history.clear()
        self._kill_history.clear()
        self._updates_in_stage = 0

    def state_dict(self) -> dict:
        return {"stage_idx": self.stage_idx, "updates_in_stage": self._updates_in_stage, "return_history": list(self._return_history), "kill_history": list(self._kill_history)}

    def load_state_dict(self, d: dict):
        self.stage_idx = min(len(STAGES) - 1, d.get("stage_idx", 0))
        self._updates_in_stage = d.get("updates_in_stage", 0)
        ret_hist = d.get("return_history", [])
        kill_hist = d.get("kill_history", [])
        if len(ret_hist) != len(kill_hist):
            min_len = min(len(ret_hist), len(kill_hist))
            ret_hist = ret_hist[-min_len:] if min_len > 0 else []
            kill_hist = kill_hist[-min_len:] if min_len > 0 else []
        self._return_history = collections.deque(ret_hist, maxlen=ADVANCE_WINDOW)
        self._kill_history = collections.deque(kill_hist, maxlen=ADVANCE_WINDOW)

    def __repr__(self):
        s = self.stage
        avg_k = (sum(self._kill_history) / len(self._kill_history)) if self._kill_history else 0.0
        return (f"CurriculumScheduler(stage={self.stage_idx}, grid={s.rows}×{s.cols}, ghosts={s.n_ghosts}, updates={self._updates_in_stage}, win_rate={avg_k:.1%})")