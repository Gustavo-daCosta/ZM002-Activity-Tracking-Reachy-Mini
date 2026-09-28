"""Motion recognition from sequences of COCO-17 keypoints."""

from dataclasses import dataclass, field

WAVE, OTHER, PAUSE = 1, 0, -1


@dataclass
class Detection:
    """Result of one wave detector.

    Attributes:
        is_wave: Whether the window was classified as a wave.
        score: Confidence in [0, 1].
        details: Detector-specific values shown on the overlay.
    """

    is_wave: bool
    score: float
    details: dict = field(default_factory=dict)
