"""
SatQuery AI - Prototype UI

Run with:  python app/gradio_app.py
(run from the repo root so the `controller` / `specialists` imports resolve)

Starts in mock mode by default — works with no GPU, no downloaded model
weights. Flip MOCK_MODE to False once you have a fine-tuned Florence-2
checkpoint (see training/finetune_florence2_lora.ipynb).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import gradio as gr
from controller.orchestrator import SatQueryOrchestrator

MOCK_MODE = True
orchestrator = SatQueryOrchestrator(mock=MOCK_MODE)


def run_single_image(image, query):
    if image is None:
        return "Please upload an image.", "", "{}"
    if not query or not query.strip():
        return "Please enter a query.", "", "{}"

    result = orchestrator.answer_single(image, query)
    return _format_output(result)


def run_bitemporal(image_a, image_b, query):
    if image_a is None or image_b is None:
        return "Please upload both images.", "", "{}"
    if not query or not query.strip():
        return "Please enter a query.", "", "{}"

    result = orchestrator.answer_bitemporal(image_a, image_b, query)
    return _format_output(result)


def run_optical_sar(optical_image, sar_image, query):
    if optical_image is None or sar_image is None:
        return "Please upload both an optical and a SAR image.", "", "{}"
    if not query or not query.strip():
        return "Please enter a query.", "", "{}"

    result = orchestrator.answer_optical_sar(optical_image, sar_image, query)
    return _format_output(result)


def _format_output(result: dict):
    if result["status"] == "error":
        answer = "⚠️ " + "; ".join(result["reasons"])
        meta = ""
    else:
        mock_tag = " (MOCK)" if result.get("is_mock") else ""
        answer = result["answer"]
        meta = (
            f"Task: {result['task']} | Specialist: {result['specialist']}{mock_tag} | "
            f"Confidence: {result['confidence']}"
        )
    trace_json = json.dumps(result["execution_trace"], indent=2)
    return answer, meta, trace_json


with gr.Blocks(title="SatQuery AI") as demo:
    gr.Markdown("# 🛰️ SatQuery AI — Prototype")
    gr.Markdown(
        "Agentic vision-language assistant for remote sensing imagery. "
        f"**Mode: {'MOCK (no model weights loaded)' if MOCK_MODE else 'LIVE MODEL'}**"
    )

    with gr.Tab("Single Image (VQA / Caption / Grounding)"):
        with gr.Row():
            img_in = gr.Image(type="pil", label="Upload remote sensing image")
            with gr.Column():
                query_in = gr.Textbox(label="Query", placeholder="e.g. 'Describe this image' or 'Is there water present?'")
                run_btn = gr.Button("Run", variant="primary")
        answer_out = gr.Textbox(label="Answer", lines=3)
        meta_out = gr.Textbox(label="Execution Summary")
        trace_out = gr.Code(label="Execution Trace (JSON)", language="json")
        run_btn.click(run_single_image, inputs=[img_in, query_in], outputs=[answer_out, meta_out, trace_out])

    with gr.Tab("Bi-Temporal (Change Detection)"):
        with gr.Row():
            img_a = gr.Image(type="pil", label="Image A (earlier)")
            img_b = gr.Image(type="pil", label="Image B (later)")
        query_bt = gr.Textbox(label="Query", placeholder="e.g. 'What changed between these two images?'")
        run_bt_btn = gr.Button("Run", variant="primary")
        answer_bt = gr.Textbox(label="Answer", lines=3)
        meta_bt = gr.Textbox(label="Execution Summary")
        trace_bt = gr.Code(label="Execution Trace (JSON)", language="json")
        run_bt_btn.click(run_bitemporal, inputs=[img_a, img_b, query_bt], outputs=[answer_bt, meta_bt, trace_bt])

    with gr.Tab("Optical + SAR Fusion"):
        with gr.Row():
            img_opt = gr.Image(type="pil", label="Optical image")
            img_sar = gr.Image(type="pil", label="SAR image")
        query_sar = gr.Textbox(label="Query", placeholder="e.g. 'What land cover is present here?'")
        run_sar_btn = gr.Button("Run", variant="primary")
        answer_sar = gr.Textbox(label="Answer", lines=3)
        meta_sar = gr.Textbox(label="Execution Summary")
        trace_sar = gr.Code(label="Execution Trace (JSON)", language="json")
        run_sar_btn.click(run_optical_sar, inputs=[img_opt, img_sar, query_sar], outputs=[answer_sar, meta_sar, trace_sar])

    gr.Markdown(
        "---\n"
        "**Prototype status:** Single-image VQA/captioning path is wired for real fine-tuned "
        "inference (flip `MOCK_MODE = False` after training). Change detection and SAR fusion "
        "are structurally complete but return mock outputs until those specialists are built."
    )

if __name__ == "__main__":
    demo.launch()
