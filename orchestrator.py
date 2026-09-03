"""
Orchestrator
------------
The single entry point the app (or an API) calls. Wires together:
  validator -> router -> specialist -> formatted output

Produces a structured dict with the answer, confidence, evidence, and a
full execution trace — this is the "agentic" transparency layer.
"""

from __future__ import annotations
from typing import Optional
from PIL import Image

from controller.validator import validate_single, validate_pair, InputMode
from controller.router import RuleBasedRouter, ExecutionTrace, TaskType
from specialists.vqa_caption import VQACaptionSpecialist
from specialists.change_detection import ChangeDetectionSpecialist
from specialists.sar_fusion import SARFusionSpecialist


class SatQueryOrchestrator:
    def __init__(self, mock: bool = True):
        self.router = RuleBasedRouter()
        self.vqa_caption = VQACaptionSpecialist(mock=mock)
        self.change_detection = ChangeDetectionSpecialist(mock=mock)
        self.sar_fusion = SARFusionSpecialist(mock=mock)

    def answer_single(self, image: Image.Image, query: str, declared_modality: Optional[str] = None) -> dict:
        trace = ExecutionTrace()
        trace.log("validator.start", input="single_image")
        validation = validate_single(image, declared_modality)
        trace.log("validator.result", valid=validation.valid, mode=validation.mode.value,
                   metadata=validation.metadata, reasons=validation.reasons)

        if not validation.valid:
            return self._error_response(validation.reasons, trace)

        decision = self.router.route(query, validation, trace)

        if decision.task == TaskType.UNSUPPORTED:
            return self._error_response(["Could not route this query to a specialist."], trace)

        task_map = {
            TaskType.VQA: "vqa",
            TaskType.CAPTION: "caption",
            TaskType.GROUNDING: "grounding",
        }
        task = task_map.get(decision.task, "vqa")

        trace.log("specialist.dispatch", specialist=decision.specialist, task=task)
        result = self.vqa_caption.run(image=image, task=task, query=query)
        trace.log("specialist.result", model=result.model_name, confidence=result.confidence,
                   is_mock=result.is_mock)

        return self._success_response(result, decision, trace)

    def answer_bitemporal(self, image_a: Image.Image, image_b: Image.Image, query: str) -> dict:
        trace = ExecutionTrace()
        trace.log("validator.start", input="bitemporal_pair")
        validation = validate_pair(image_a, image_b, pair_kind="bitemporal")
        trace.log("validator.result", valid=validation.valid, mode=validation.mode.value,
                   reasons=validation.reasons)

        if not validation.valid:
            return self._error_response(validation.reasons, trace)

        decision = self.router.route(query, validation, trace)
        trace.log("specialist.dispatch", specialist="change_detection")
        result = self.change_detection.run(image_a, image_b, query=query)
        trace.log("specialist.result", model=result.model_name, confidence=result.confidence,
                   is_mock=result.is_mock)

        return self._success_response(result, decision, trace)

    def answer_optical_sar(self, optical_image: Image.Image, sar_image: Image.Image, query: str) -> dict:
        trace = ExecutionTrace()
        trace.log("validator.start", input="optical_sar_pair")
        validation = validate_pair(optical_image, sar_image, pair_kind="optical_sar")
        trace.log("validator.result", valid=validation.valid, mode=validation.mode.value,
                   reasons=validation.reasons)

        if not validation.valid:
            return self._error_response(validation.reasons, trace)

        decision = self.router.route(query, validation, trace)
        trace.log("specialist.dispatch", specialist="sar_fusion")
        result = self.sar_fusion.run(optical_image, sar_image, query=query)
        trace.log("specialist.result", model=result.model_name, confidence=result.confidence,
                   is_mock=result.is_mock)

        return self._success_response(result, decision, trace)

    # -- helpers ------------------------------------------------------

    @staticmethod
    def _success_response(result, decision, trace: ExecutionTrace) -> dict:
        return {
            "status": "ok",
            "answer": result.answer,
            "confidence": result.confidence,
            "task": decision.task.value,
            "specialist": decision.specialist,
            "is_mock": result.is_mock,
            "model_name": result.model_name,
            "evidence": result.evidence,
            "execution_trace": trace.as_dict(),
        }

    @staticmethod
    def _error_response(reasons: list[str], trace: ExecutionTrace) -> dict:
        return {
            "status": "error",
            "answer": None,
            "reasons": reasons,
            "execution_trace": trace.as_dict(),
        }
