"""
Base interface every specialist implements, plus the shared result shape.
Keeping this uniform means the orchestrator/app code never needs to know
which specialist it's calling.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SpecialistResult:
    answer: str
    confidence: float                    # 0.0 - 1.0
    evidence: dict = field(default_factory=dict)   # e.g. bbox, heatmap path, region text
    raw: Optional[Any] = None            # raw model output, for debugging
    model_name: str = "unknown"
    is_mock: bool = False


class BaseSpecialist:
    name: str = "base"

    def run(self, **kwargs) -> SpecialistResult:
        raise NotImplementedError
