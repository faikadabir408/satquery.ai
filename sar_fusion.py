"""
Cross-modal optical-SAR fusion specialist.

STUB — later-phase roadmap item, needs the most additional data engineering
(co-registered optical/SAR pairs, e.g. from SEN1-2 or similar). Mocked here
so the orchestration layer can be developed and tested now.
"""

from __future__ import annotations
from PIL import Image
import random

from specialists.base import BaseSpecialist, SpecialistResult


class SARFusionSpecialist(BaseSpecialist):
    name = "sar_fusion"

    def __init__(self, mock: bool = True):
        self.mock = mock
        if not mock:
            raise NotImplementedError(
                "Real optical-SAR fusion model not wired up yet. Use mock=True for now."
            )

    _MOCK_FUSION_ANSWERS = [
        "Combining optical and SAR data, the region shows consistent structural "
        "signatures suggesting built-up land, despite partial cloud cover in the optical layer.",
        "SAR backscatter indicates surface roughness consistent with agricultural terrain; "
        "this aligns with the optical image's crop-field appearance.",
        "No reliable optical signal was available (cloud-obscured); SAR data alone "
        "suggests a water body based on low backscatter return.",
    ]

    def run(self, optical_image: Image.Image, sar_image: Image.Image, query: str = "") -> SpecialistResult:
        answer = random.choice(self._MOCK_FUSION_ANSWERS)
        return SpecialistResult(
            answer=answer,
            confidence=round(random.uniform(0.5, 0.75), 2),
            evidence={"note": "MOCK OUTPUT — no real fusion model wired up yet"},
            raw=None,
            model_name="mock-sar-fusion-v0",
            is_mock=True,
        )
