"""
Bi-temporal change detection specialist.

STUB — this phase of the roadmap. Returns a structured mock result so the
controller/UI pipeline can be built and tested end-to-end now. Replace
_run_mock's internals with a real model call once you train/adopt one
(e.g. a CDVQA-style model, or a Siamese change-detection backbone).
"""

from __future__ import annotations
from PIL import Image
import random

from specialists.base import BaseSpecialist, SpecialistResult


class ChangeDetectionSpecialist(BaseSpecialist):
    name = "change_detection"

    def __init__(self, mock: bool = True):
        self.mock = mock
        if not mock:
            raise NotImplementedError(
                "Real change-detection model not wired up yet. Use mock=True for now."
            )

    _MOCK_CHANGE_DESCRIPTIONS = [
        "Detected new construction in the northeast quadrant between the two "
        "timestamps; approximately 12% of the scene shows land-cover change.",
        "Vegetation loss observed along the southern edge, consistent with "
        "deforestation or seasonal clearing.",
        "No significant structural change detected; minor spectral variation "
        "likely due to seasonal/lighting differences.",
    ]

    def run(self, image_a: Image.Image, image_b: Image.Image, query: str = "") -> SpecialistResult:
        answer = random.choice(self._MOCK_CHANGE_DESCRIPTIONS)
        return SpecialistResult(
            answer=answer,
            confidence=round(random.uniform(0.5, 0.8), 2),
            evidence={
                "note": "MOCK OUTPUT — no real change-detection model wired up yet",
                "change_map": None,  # would hold a path/array to a diff heatmap
            },
            raw=None,
            model_name="mock-change-detection-v0",
            is_mock=True,
        )
