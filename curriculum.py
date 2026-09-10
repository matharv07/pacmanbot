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
    world_height: float
    world_width: float
    obs_resolution: float
    n_ghosts: int
    n_power: int
    advance_return: float
    min_updates: int
    target_kill_rate: float = 0.35

    @property
    def rows(self) -> int:
        return int(self.world_height * self.obs_resolution)
        
    @property
    def cols(self) -> int:
        return int(self.world_width * self.obs_resolution)

STAGES = [
    Stage(world_height=7,  world_width=9,  obs_resolution=1.0, n_ghosts=3, n_power=2,  advance_return=-7.0, min_updates=150, target_kill_rate=0.34),
    Stage(world_height=13, world_width=17, obs_resolution=1.0, n_ghosts=4, n_power=6,  advance_return=-6.0, min_updates=150, target_kill_rate=0.38),
    Stage(world_height=21, world_width=27, obs_resolution=1.0, n_ghosts=5, n_power=14, advance_return=-2.0, min_updates=200, target_kill_rate=0.45),
    Stage(world_height=27, world_width=33, obs_resolution=1.0, n_ghosts=6, n_power=24, advance_return=0.0,  min_updates=250, target_kill_rate=0.50),
    Stage(world_height=33, world_width=41, obs_resolution=1.0, n_ghosts=7, n_power=28, advance_return=float('inf'), min_updates=0, target_kill_rate=0.60)]

ADVANCE_WINDOW = 50    #rolling window of updates achieving return/kill threshold required to clear a stage

class CurriculumScheduler:
    def __init__(self, start_stage: int = 0):
        self.stage_idx = start_stage
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
        if mean_return is not None:
            self._return_history.append(mean_return)
        if kill_rate is not None:
            self._kill_history.append(kill_rate)
        self._updates_in_stage += 1

    def should_advance(self) -> bool:
        if self.is_final:
            return False
        if self._updates_in_stage < self.stage.min_updates:
            return False
        if len(self._return_history) < ADVANCE_WINDOW:
            return False
        avg_ret = sum(self._return_history) / len(self._return_history)
        avg_kill = (sum(self._kill_history) / len(self._kill_history)) if self._kill_history else 0.0

        # Dominant performance gate: exceeds stage target by 15% relative (or >= 50% for small stages)
        dominant_gate = max(0.50, min(0.95, self.stage.target_kill_rate * 1.15))
        if avg_kill >= dominant_gate and avg_ret >= (self.stage.advance_return - 2.0):
            return True

        # Solid target: meets both calibrated advance_return and target_kill_rate
        if avg_ret >= self.stage.advance_return and avg_kill >= self.stage.target_kill_rate:
            return True

        # Plateau detection: if training has stalled in this stage after min_updates + ADVANCE_WINDOW
        if self._updates_in_stage >= self.stage.min_updates + ADVANCE_WINDOW:
            half = ADVANCE_WINDOW // 2
            hist = list(self._return_history)
            avg_first = sum(hist[:half]) / half
            avg_second = sum(hist[half:]) / half
            competency_kill = self.stage.target_kill_rate * 0.90
            # If returns have flattened out (< 0.5 improvement) and policy maintains competent baseline (>= 90% of target kill rate)
            if (avg_second - avg_first) < 0.5 and (avg_kill >= competency_kill or avg_ret >= self.stage.advance_return):
                print(f"Curriculum advancing due to plateau: improvement {avg_second - avg_first:.2f} < 0.5 (Current avg ret: {avg_second:.2f}, kill: {avg_kill:.1%})")
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
        self.stage_idx = d.get("stage_idx", 0)
        self._updates_in_stage = d.get("updates_in_stage", 0)
        self._return_history = collections.deque(d.get("return_history", []), maxlen=ADVANCE_WINDOW)
        self._kill_history = collections.deque(d.get("kill_history", []), maxlen=ADVANCE_WINDOW)

    def __repr__(self):
        s = self.stage
        avg_k = (sum(self._kill_history) / len(self._kill_history)) if self._kill_history else 0.0
        return (f"CurriculumScheduler(stage={self.stage_idx}, grid={s.rows}×{s.cols}, ghosts={s.n_ghosts}, updates={self._updates_in_stage}, win_rate={avg_k:.1%})")