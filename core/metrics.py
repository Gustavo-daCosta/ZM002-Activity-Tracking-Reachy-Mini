"""Per-stage inference timings and loop FPS, for comparing pose models."""

import time
from collections import defaultdict
from typing import Callable, Dict

import numpy as np


class StageStats:
    def __init__(self, window_s: float = 1.0, clock: Callable[[], float] = time.perf_counter):
        self.window_s = window_s
        self._clock = clock
        self._start = clock()
        self.frames = 0
        self.frames_with_target = 0
        self._samples = defaultdict(list)
        self._window = defaultdict(list)
        self._window_start = self._start
        self._window_frames = 0
        self._panel = {"fps": 0.0, "ms": {}}

    def add_frame(self, timings: Dict[str, float], has_target: bool) -> None:
        self.frames += 1
        self.frames_with_target += int(has_target)
        self._window_frames += 1
        if timings:
            timings = {**timings, "total": sum(timings.values())}
        for stage, ms in timings.items():
            self._samples[stage].append(ms)
            self._window[stage].append(ms)

        now = self._clock()
        elapsed = now - self._window_start
        if elapsed >= self.window_s:
            self._panel = {
                "fps": self._window_frames / elapsed,
                "ms": {stage: float(np.mean(values)) for stage, values in self._window.items()},
            }
            self._window = defaultdict(list)
            self._window_start = now
            self._window_frames = 0

    def window_summary(self) -> dict:
        return self._panel

    def final_summary(self) -> dict:
        elapsed = self._clock() - self._start
        return {
            "frames": self.frames,
            "fps": self.frames / elapsed if elapsed > 0 else 0.0,
            "person_pct": 100.0 * self.frames_with_target / self.frames if self.frames else 0.0,
            "stages": {
                stage: {"mean": float(np.mean(values)), "p95": float(np.percentile(values, 95))}
                for stage, values in self._samples.items()
            },
        }


def format_summary(model: str, summary: dict) -> str:
    lines = [
        f"=== {model} ===",
        f"frames: {summary['frames']}  loop FPS: {summary['fps']:.1f}  person detected: {summary['person_pct']:.0f}%",
    ]
    for stage, stats in summary["stages"].items():
        lines.append(f"  {stage:<9} mean {stats['mean']:6.1f} ms   p95 {stats['p95']:6.1f} ms")
    return "\n".join(lines)
