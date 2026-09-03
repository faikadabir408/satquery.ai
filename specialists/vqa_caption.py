"""
VQA / Captioning / Grounding specialist.

Two modes:
  - mock=True  (default here): returns deterministic, structured fake
    answers so the whole pipeline is runnable and demoable with zero
    model weights / GPU. Useful for testing the controller + UI right now.
  - mock=False: loads Florence-2-base (or a fine-tuned checkpoint) via
    transformers and runs real inference. Use this on Colab once you've
    fine-tuned (see training/finetune_florence2_lora.ipynb).

Swapping is a single flag — nothing else in the system needs to change.
"""

from __future__ import annotations
from PIL import Image
import random

from specialists.base import BaseSpecialist, SpecialistResult


class VQACaptionSpecialist(BaseSpecialist):
    name = "vqa_caption"

    def __init__(self, mock: bool = True, model_path: str = "microsoft/Florence-2-base",
                 device: str = "cuda"):
        self.mock = mock
        self.model_path = model_path
        self.device = device
        self._model = None
        self._processor = None

        if not mock:
            self._load_real_model()

    # -- real model path -----------------------------------------------

    def _load_real_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path, trust_remote_code=True, torch_dtype=torch.float16
        ).to(self.device)
        self._processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)

    def _run_real(self, image: Image.Image, task: str, query: str = "") -> SpecialistResult:
        import torch

        # Florence-2 uses task-prompt tokens; map our task types to theirs.
        task_prompt_map = {
            "caption": "<DETAILED_CAPTION>",
            "vqa": "<VQA>",              # placeholder tag; fine-tune defines the real one
            "grounding": "<CAPTION_TO_PHRASE_GROUNDING>",
        }
        prompt = task_prompt_map.get(task, "<CAPTION>")
        full_prompt = prompt if task != "vqa" else f"{prompt} {query}"

        inputs = self._processor(text=full_prompt, images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=256,
                num_beams=3,
            )
        raw_text = self._processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self._processor.post_process_generation(
            raw_text, task=prompt, image_size=(image.width, image.height)
        )
        answer = parsed.get(prompt, str(parsed))

        return SpecialistResult(
            answer=str(answer),
            confidence=0.75,  # placeholder until real confidence estimation is added
            evidence={"task_prompt": prompt},
            raw=parsed,
            model_name=self.model_path,
            is_mock=False,
        )

    # -- mock path --------------------------------------------------------

    _MOCK_CAPTIONS = [
        "A satellite view showing a mix of agricultural fields and a small settlement, "
        "with a road running diagonally through the frame.",
        "An aerial scene of forested land bordering a river, with visible sediment "
        "patterns near the water's edge.",
        "An urban area with dense building clusters, a central road network, and "
        "sparse green space.",
    ]

    _MOCK_VQA_ANSWERS = [
        "Yes, there appears to be water present in the lower-left region of the image.",
        "The dominant land cover class is agricultural / cropland.",
        "No visible cloud cover is affecting this scene.",
        "There are approximately 3 to 5 distinguishable building clusters.",
    ]

    def _run_mock(self, image: Image.Image, task: str, query: str = "") -> SpecialistResult:
        w, h = image.size
        if task == "caption":
            answer = random.choice(self._MOCK_CAPTIONS)
        elif task == "grounding":
            answer = f"Region matching '{query}' located near the image center."
        else:  # vqa
            answer = random.choice(self._MOCK_VQA_ANSWERS)

        return SpecialistResult(
            answer=answer,
            confidence=round(random.uniform(0.55, 0.85), 2),
            evidence={
                "note": "MOCK OUTPUT — replace by setting mock=False after fine-tuning",
                "bbox_example": [int(w * 0.3), int(h * 0.3), int(w * 0.7), int(h * 0.7)] if task == "grounding" else None,
            },
            raw=None,
            model_name="mock-vqa-caption-v0",
            is_mock=True,
        )

    # -- unified entry point ------------------------------------------

    def run(self, image: Image.Image, task: str = "vqa", query: str = "") -> SpecialistResult:
        if self.mock:
            return self._run_mock(image, task, query)
        return self._run_real(image, task, query)
