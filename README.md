# SatQuery AI (Prototype)

An agentic, query-driven vision-language system for natural-language analysis
of remote sensing imagery — single-image VQA/captioning, bi-temporal change
detection, and optical-SAR fusion.

**Current status: early prototype.** The orchestration layer (validator →
router → specialist → structured output) is fully working. Single-image
VQA/captioning is wired for real fine-tuned inference; change detection and
SAR fusion are structurally complete but return mock outputs until those
specialists are built (see Roadmap).

## Quickstart (mock mode — runs anywhere, no GPU needed)

```bash
cd satquery-ai
pip install -r requirements.txt
python app/gradio_app.py
```

Open the local URL Gradio prints. Three tabs:
- **Single Image** — upload one image, ask a question or ask for a caption
- **Bi-Temporal** — upload two images, ask what changed
- **Optical + SAR** — upload an optical and a SAR image, ask a fused question

Every response includes an **execution trace** (JSON) showing the validator
result, routing decision, and which specialist ran — this is the
transparency/evidence layer the system is built around.

## Architecture

```
Image(s) + Query
      │
      ▼
 controller/validator.py   → checks image validity, detects single / bi-temporal / optical-SAR
      │
      ▼
 controller/router.py      → classifies query intent, picks a specialist (rule-based; LLM-based router stubbed in)
      │
      ▼
 specialists/*.py          → vqa_caption (real-capable), change_detection (mock), sar_fusion (mock)
      │
      ▼
 controller/orchestrator.py → ties it together, builds the structured response + execution trace
      │
      ▼
 app/gradio_app.py          → UI
```

Each specialist implements the same interface (`specialists/base.py`), so
swapping a mock for a real model — or adding a new specialist entirely —
doesn't require touching the router or UI.

## Repo layout

```
satquery-ai/
├── controller/
│   ├── validator.py       # input validation + modality/mode detection
│   ├── router.py           # intent classification + routing (rule-based + LLM stub)
│   └── orchestrator.py     # wires validator → router → specialist → response
├── specialists/
│   ├── base.py              # shared interface
│   ├── vqa_caption.py       # mock + real (Florence-2) VQA/caption/grounding
│   ├── change_detection.py  # mock (Phase 4)
│   └── sar_fusion.py         # mock (Phase 4, needs more data engineering)
├── training/
│   └── finetune_florence2_lora.ipynb  # Colab-ready LoRA fine-tuning skeleton
├── data/
│   └── prepare_rsvqa.py      # RSVQA-LR → training jsonl converter
├── app/
│   └── gradio_app.py          # demo UI
└── requirements.txt
```

## Roadmap

**Phase 1 — Orchestration skeleton (done)**
Validator, router, mocked specialists, Gradio UI, structured execution
traces. This is what's in the repo right now.

**Phase 2 — Real VQA/captioning**
- Download RSVQA-LR (small, ~772 images) + a captioning set (UCM-Captions or NWPU-Captions)
- Run `data/prepare_rsvqa.py` to format it
- Fine-tune Florence-2-base via LoRA using `training/finetune_florence2_lora.ipynb` on Colab
- Set `mock=False` in `specialists/vqa_caption.py`, point at your checkpoint

**Phase 3 — Evidence grounding**
- Wire up Florence-2's native grounding output (`<CAPTION_TO_PHRASE_GROUNDING>`) to draw bounding boxes on the image in the UI
- Add a real confidence signal (e.g. generation entropy or self-consistency sampling) instead of the placeholder value

**Phase 4 — Expand specialists**
- Bi-temporal change detection (CDVQA-style dataset/model)
- Optical-SAR fusion (needs co-registered pairs — SEN1-2 or similar; this is the most compute/data-heavy piece, treat as a stretch goal)

## Notes on compute

Built with free-tier Colab/Kaggle (T4, 16GB VRAM) in mind:
- Florence-2-base (0.23B params) fine-tunes comfortably on a T4
- BigEarthNet (590GB) and full VRSBench are impractical on free tier — start
  with RSVQA-LR and small captioning sets, expand later if you get access to
  more compute
- LoRA (via `peft`) keeps memory and checkpoint size manageable
