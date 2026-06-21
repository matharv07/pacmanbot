"""
Curriculum Learning Scheduler for stepwise grid-size scaling.

Defines training stages that gradually increase grid complexity:
  Stage 0: 7x9    grid, 2 ghosts - learn basic pursuit
  Stage 1: 13x17  grid, 3 ghosts - learn corridor navigation
  Stage 2: 21x27  grid, 5 ghosts - learn belief-based hunting
  Stage 3: 33x41  grid, 7 ghosts - full game (final fine-tuning)

Advancement is triggered when the rolling mean return plateaus above
a per-stage threshold for a sustained window of updates.
"""

from __future__ import annotations
from dataclasses import dataclass
import collections

@dataclass(frozen=True)
class Stage:
    rows: int
    cols: int
    n_ghosts: int
    n_power: int           
    advance_return: float
    min_updates: int

STAGES = [Stage(rows=7,  cols=9,  n_ghosts=2, n_power=2,  advance_return=42.0, min_updates=150),
    Stage(rows=13, cols=17, n_ghosts=3, n_power=6,  advance_return=40.0, min_updates=350),
    Stage(rows=21, cols=27, n_ghosts=5, n_power=14, advance_return=28.0, min_updates=350),
    Stage(rows=33, cols=41, n_ghosts=7, n_power=28, advance_return=float('inf'), min_updates=0)]

ADVANCE_WINDOW = 150   #minimum updates required per stage

class CurriculumScheduler:
    def __init__(self, start_stage: int = 0):
        self.stage_idx = start_stage
        self._return_history: collections.deque = collections.deque(maxlen=ADVANCE_WINDOW)
        self._updates_in_stage: int = 0
    @property
    def stage(self) -> Stage:
        return STAGES[self.stage_idx]
    @property
    def is_final(self) -> bool:
        return self.stage_idx >= len(STAGES) - 1

    def record_return(self, mean_return: float | None):
        if mean_return is not None:
            self._return_history.append(mean_return)
        self._updates_in_stage += 1

    def should_advance(self) -> bool:
        if self.is_final:
            return False
        if self._updates_in_stage < self.stage.min_updates:
            return False
        if len(self._return_history) < ADVANCE_WINDOW // 2:
            return False
        avg = sum(self._return_history) / len(self._return_history)
        return avg >= self.stage.advance_return

    def advance(self):
        if self.is_final:
            return
        self.stage_idx += 1
        self._return_history.clear()
        self._updates_in_stage = 0

    def state_dict(self) -> dict:
        return { "stage_idx": self.stage_idx, "updates_in_stage": self._updates_in_stage, "return_history": list(self._return_history) }

    def load_state_dict(self, d: dict):
        self.stage_idx = d.get("stage_idx", 0)
        self._updates_in_stage = d.get("updates_in_stage", 0)
        self._return_history = collections.deque(d.get("return_history", []), maxlen=ADVANCE_WINDOW)

    def __repr__(self):
        s = self.stage
        return (f"CurriculumScheduler(stage={self.stage_idx}, "f"grid={s.rows}×{s.cols}, ghosts={s.n_ghosts}, "f"updates={self._updates_in_stage})")