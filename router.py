"""
Controller / Router
--------------------
Decides which specialist handles a given (validated input, natural-language
query) pair, and builds an execution trace as it goes.

Two routing strategies are supported:
  1. RuleBasedRouter   - fast, deterministic, keyword/heuristic driven.
                         Good enough for a prototype and has zero dependencies.
  2. LLMRouter          - calls an LLM with a small function-calling-style
                         prompt to classify intent. Swap in once you want
                         more robust natural-language understanding.

Both implement the same `.route()` interface so the rest of the system
doesn't care which one is active.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time
import uuid

from controller.validator import InputMode, ValidationResult


class TaskType(str, Enum):
    VQA = "vqa"
    CAPTION = "caption"
    GROUNDING = "grounding"
    CHANGE_DETECTION = "change_detection"
    SAR_FUSION = "sar_fusion"
    UNSUPPORTED = "unsupported"


@dataclass
class RoutingDecision:
    task: TaskType
    specialist: str
    confidence: float
    trace: list[dict] = field(default_factory=list)


@dataclass
class ExecutionTrace:
    """Structured record of what the system did, for transparency/debugging."""
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    steps: list[dict] = field(default_factory=list)

    def log(self, step: str, **details):
        self.steps.append({"step": step, "t": round(time.time(), 3), **details})

    def as_dict(self):
        return {"trace_id": self.trace_id, "steps": self.steps}


# --- keyword tables used by the rule-based router --------------------------

_CAPTION_HINTS = ("describe", "caption", "what is in this image", "summarize the scene", "what does this image show")
_GROUNDING_HINTS = ("where is", "locate", "point to", "bounding box", "highlight the", "find the")
_CHANGE_HINTS = ("change", "difference", "before and after", "over time", "compare these two", "what changed")
_SAR_HINTS = ("sar", "radar", "cloud cover", "regardless of clouds", "optical and sar", "fuse")


class RuleBasedRouter:
    """Deterministic keyword router. No model dependency — safe default."""

    def route(self, query: str, validation: ValidationResult, trace: ExecutionTrace) -> RoutingDecision:
        q = query.lower().strip()
        trace.log("router.received_query", query=query, input_mode=validation.mode.value)

        if not validation.valid:
            trace.log("router.rejected", reasons=validation.reasons)
            return RoutingDecision(task=TaskType.UNSUPPORTED, specialist="none", confidence=1.0)

        # Mode dictates the eligible task family first.
        if validation.mode == InputMode.OPTICAL_SAR:
            decision = RoutingDecision(task=TaskType.SAR_FUSION, specialist="sar_fusion", confidence=0.9)

        elif validation.mode == InputMode.BITEMPORAL:
            decision = RoutingDecision(task=TaskType.CHANGE_DETECTION, specialist="change_detection", confidence=0.9)

        elif validation.mode == InputMode.SINGLE:
            if any(h in q for h in _GROUNDING_HINTS):
                decision = RoutingDecision(task=TaskType.GROUNDING, specialist="vqa_caption", confidence=0.75)
            elif any(h in q for h in _CAPTION_HINTS):
                decision = RoutingDecision(task=TaskType.CAPTION, specialist="vqa_caption", confidence=0.8)
            else:
                # default: treat as an open VQA question
                decision = RoutingDecision(task=TaskType.VQA, specialist="vqa_caption", confidence=0.6)
        else:
            decision = RoutingDecision(task=TaskType.UNSUPPORTED, specialist="none", confidence=1.0)

        trace.log("router.decision", task=decision.task.value, specialist=decision.specialist,
                   confidence=decision.confidence)
        return decision


class LLMRouter:
    """
    Placeholder for an LLM-driven router. Wire this up to any chat/completions
    API later — the prompt below is a starting point. Kept out of the
    prototype's critical path until you're ready to add an API key.
    """

    SYSTEM_PROMPT = """You are the routing controller for a remote-sensing vision-language system.
Given a user's natural-language query and the input mode (single / bitemporal / optical_sar),
respond with ONLY one of: vqa, caption, grounding, change_detection, sar_fusion, unsupported."""

    def __init__(self, llm_call_fn=None):
        # llm_call_fn: Callable[[str, str], str] -> (system_prompt, user_prompt) -> response text
        self.llm_call_fn = llm_call_fn

    def route(self, query: str, validation: ValidationResult, trace: ExecutionTrace) -> RoutingDecision:
        if self.llm_call_fn is None:
            raise NotImplementedError(
                "LLMRouter has no llm_call_fn configured. Pass one in, or use RuleBasedRouter for now."
            )
        user_prompt = f"Query: {query}\nInput mode: {validation.mode.value}"
        raw = self.llm_call_fn(self.SYSTEM_PROMPT, user_prompt).strip().lower()
        try:
            task = TaskType(raw)
        except ValueError:
            task = TaskType.UNSUPPORTED
        trace.log("router.llm_decision", raw_response=raw, resolved_task=task.value)
        return RoutingDecision(task=task, specialist=task.value if task != TaskType.UNSUPPORTED else "none",
                                confidence=0.7)
