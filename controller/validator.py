"""
Input Validator
----------------
Runs cheap, fast checks on the incoming image(s) before anything expensive
(model inference) happens. Decides:
  - is this actually image data we can process?
  - single image, bi-temporal pair, or optical+SAR pair?
  - basic quality flags (too small, corrupt, etc.)

This stays rule-based / heuristic on purpose — no model needed here.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from PIL import Image
import numpy as np


class InputMode(str, Enum):
    SINGLE = "single"
    BITEMPORAL = "bitemporal"
    OPTICAL_SAR = "optical_sar"
    INVALID = "invalid"


@dataclass
class ValidationResult:
    mode: InputMode
    valid: bool
    reasons: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


MIN_DIM = 32          # reject absurdly tiny images
MAX_DIM = 8192         # sanity cap
SAR_GRAYSCALE_STD_HINT = 3.0  # heuristic threshold, tuned later w/ real SAR data


def _basic_image_checks(img: Image.Image) -> list[str]:
    reasons = []
    w, h = img.size
    if w < MIN_DIM or h < MIN_DIM:
        reasons.append(f"image too small ({w}x{h}), minimum is {MIN_DIM}px")
    if w > MAX_DIM or h > MAX_DIM:
        reasons.append(f"image too large ({w}x{h}), maximum is {MAX_DIM}px")
    return reasons


def _looks_like_sar(img: Image.Image) -> bool:
    """
    Very rough heuristic: SAR imagery is typically single-channel /
    near-grayscale with a distinct speckle texture. This is a placeholder —
    replace with a real modality classifier once you have labeled SAR/optical
    examples (or just rely on explicit metadata/filenames from the user).
    """
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    channel_std = arr.std(axis=(0, 1))
    # if R, G, B channels are nearly identical -> effectively grayscale
    channel_spread = channel_std.max() - channel_std.min()
    return channel_spread < SAR_GRAYSCALE_STD_HINT


def validate_single(image: Image.Image, declared_modality: Optional[str] = None) -> ValidationResult:
    reasons = _basic_image_checks(image)
    if reasons:
        return ValidationResult(mode=InputMode.INVALID, valid=False, reasons=reasons)

    modality = declared_modality or ("sar" if _looks_like_sar(image) else "optical")
    return ValidationResult(
        mode=InputMode.SINGLE,
        valid=True,
        metadata={"modality": modality, "size": image.size},
    )


def validate_pair(image_a: Image.Image, image_b: Image.Image,
                   pair_kind: str = "bitemporal") -> ValidationResult:
    """
    pair_kind: 'bitemporal' (same modality, different time) or
               'optical_sar' (same time/area, different modality)
    """
    reasons = _basic_image_checks(image_a) + _basic_image_checks(image_b)

    if image_a.size != image_b.size:
        reasons.append(
            f"image pair size mismatch: {image_a.size} vs {image_b.size} "
            "(co-registration required upstream)"
        )

    if reasons:
        return ValidationResult(mode=InputMode.INVALID, valid=False, reasons=reasons)

    mode = InputMode.OPTICAL_SAR if pair_kind == "optical_sar" else InputMode.BITEMPORAL
    return ValidationResult(
        mode=mode,
        valid=True,
        metadata={"size": image_a.size, "pair_kind": pair_kind},
    )
