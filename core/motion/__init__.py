"""Motion recognition from sequences of COCO keypoints (first movement: hand wave)."""

from dataclasses import dataclass, field
from typing import Optional, Protocol

WAVE, OTHER, PAUSE = 1, 0, -1


@dataclass
class Detection:
    is_wave: bool
    score: float  # 0..1
    details: dict = field(default_factory=dict)


class MotionDetector(Protocol):
    name: str

    def detect(self, features: Optional[dict]) -> Optional[Detection]: ...
