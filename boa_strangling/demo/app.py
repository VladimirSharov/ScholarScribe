"""Standalone demo: type a title + abstract, get predicted subject tags.

Shows two models side by side - "best" (checkpoint-145404, the actual final
checkpoint) vs "comparison" (checkpoint-117000, an earlier checkpoint from
the same run that a training-time validation metric briefly favored). See
THESIS_LOG.md in the main repo (2026-07-21 entries) for why 145404 is
genuinely ahead once thresholds are compared fairly, and why 117000 alone
can look artificially competitive with its own (looser) calibration.

Fully self-contained: everything needed lives under package/ next to this
file (fp16 weights, tokenizer, label encoder, precomputed thresholds) - no
access to the original training datasets required.

Usage:
    pip install -r requirements.txt
    python app.py
"""
import json
import os
import pickle

import numpy as np
import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification

BASE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(BASE, "package")

with open(os.path.join(PKG, "label_encoder.pkl"), "rb") as f:
    MLB = pickle.load(f)
with open(os.path.join(PKG, "inference_meta.json"), encoding="utf-8") as f:
    META = json.load(f)
with open(os.path.join(PKG, "examples.json"), encoding="utf-8") as f:
    EXAMPLES = json.load(f)

BUCKET_IDS = np.array(META["bucket_ids"])
FREQ_BIN_NAMES = META["freq_bin_names"]
THRESHOLD_MAPS = META["threshold_maps"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TOKENIZER = XLMRobertaTokenizer.from_pretrained(os.path.join(PKG, "tokenizer"))


def _load_model(name):
    # Weights are stored fp16 to keep the download small; upcast to fp32 on
    # CPU since fp16 matmul support/perf on CPU-only PyTorch is unreliable.
    model = XLMRobertaForSequenceClassification.from_pretrained(
        os.path.join(PKG, "weights", name), torch_dtype=torch.float16
    )
    if DEVICE.type != "cuda":
        model = model.float()
    model.to(DEVICE).eval()
    return model


print(f"Loading models on {DEVICE}...")
MODELS = {}
for _name in ("best", "comparison"):
    _weights_dir = os.path.join(PKG, "weights", _name)
    if os.path.isdir(_weights_dir) and os.path.exists(os.path.join(_weights_dir, "model.safetensors")):
        MODELS[_name] = _load_model(_name)
        print(f"  loaded '{_name}'")
    else:
        print(f"  '{_name}' not found under {_weights_dir} - skipping (download it later to enable)")
print("Ready.")


def apply_thresholds(probs, threshold_map):
    preds = np.zeros_like(probs, dtype=bool)
    for bucket_name, thresh in threshold_map.items():
        bucket_id = FREQ_BIN_NAMES.index(bucket_name)
        cols = np.where(BUCKET_IDS == bucket_id)[0]
        preds[cols] = probs[cols] > thresh
    return preds


def run_one(text, model_key, threshold_key, top_k=12):
    model = MODELS[model_key]
    threshold_map = THRESHOLD_MAPS[threshold_key]
    encoding = TOKENIZER(
        text, truncation=True, padding="max_length", max_length=512, return_tensors="pt"
    ).to(DEVICE)
    with torch.no_grad():
        logits = model(**encoding).logits
    probs = torch.sigmoid(logits.float()).cpu().numpy()[0]
    preds = apply_thresholds(probs, threshold_map)
    predicted_tags = sorted(MLB.classes_[i] for i in np.where(preds)[0])
    top_idx = np.argsort(probs)[-top_k:][::-1]
    top = [(MLB.classes_[i], float(probs[i])) for i in top_idx]
    return predicted_tags, top


def format_tags(tags):
    return ", ".join(tags) if tags else "(no tag crossed the threshold)"


def format_top(top):
    return "\n".join(f"{score:>6.3f}  {tag}" for tag, score in top)


def fill_example(choice):
    if not choice or choice == "(type your own)":
        return ""
    return next(e["text"] for e in EXAMPLES if e["id"] == choice)


def _example_note(choice):
    if not choice or choice == "(type your own)":
        return ""
    note = next((e.get("note") for e in EXAMPLES if e["id"] == choice), None)
    return f"*{note}*" if note else ""


_NOT_DOWNLOADED = "(checkpoint not downloaded yet - see boa_strangling/demo/README.md)"


def predict(text, threshold_mode, example_choice):
    if example_choice and example_choice != "(type your own)":
        text = next(e["text"] for e in EXAMPLES if e["id"] == example_choice)

    if not text or not text.strip():
        return "", "", "", "", "Type or pick some text first."

    text = text[:3000]
    shared = threshold_mode.startswith("Shared")
    best_key = "shared_fixed" if shared else "best_own"
    cmp_key = "shared_fixed" if shared else "comparison_own"

    if "best" in MODELS:
        best_tags, best_top = run_one(text, "best", best_key)
        best_tags, best_top = format_tags(best_tags), format_top(best_top)
    else:
        best_tags, best_top = _NOT_DOWNLOADED, ""

    if "comparison" in MODELS:
        cmp_tags, cmp_top = run_one(text, "comparison", cmp_key)
        cmp_tags, cmp_top = format_tags(cmp_tags), format_top(cmp_top)
    else:
        cmp_tags, cmp_top = _NOT_DOWNLOADED, ""

    true_tags_note = ""
    if example_choice and example_choice != "(type your own)":
        true = next(e["true_tags"] for e in EXAMPLES if e["id"] == example_choice)
        true_tags_note = "**Ground-truth tags:** " + ", ".join(true)
        if not example_choice.startswith("synthetic:"):
            true_tags_note += f"  \n**Source ID:** `{example_choice}`"
        note = _example_note(example_choice)
        if note:
            true_tags_note = note + "\n\n" + true_tags_note

    return best_tags, best_top, cmp_tags, cmp_top, true_tags_note


def build_ui():
    import gradio as gr

    with gr.Blocks(title="ScholarScribe tag prediction demo") as demo:
        gr.Markdown(
            "# ScholarScribe - thesis tag prediction demo\n"
            "Type a title + abstract (or pick a real test-set example below) and compare the "
            "final checkpoint (`checkpoint-145404`, **best**) against an earlier checkpoint from "
            "the same training run (`checkpoint-117000`, **comparison**) that a validation metric "
            "briefly favored during training, but which later analysis showed is behind on every "
            "metric once thresholds are compared fairly. Full analysis: `THESIS_LOG.md`, "
            "2026-07-21 entries, in the main repo."
        )
        with gr.Row():
            example_dropdown = gr.Dropdown(
                choices=[("(type your own)", "(type your own)")]
                + [(e["name"], e["id"]) for e in EXAMPLES],
                value="(type your own)",
                label="Example text (optional)",
                info="Each example is picked to show something specific - a rare-checkpoint "
                     "win, a long-tail-heavy thesis, a clean failure case, a micro- vs. "
                     "macro-F1 illustration, or a synthetic (self-authored) text.",
            )
            threshold_mode = gr.Radio(
                choices=["Shared thresholds (fair comparison)", "Each model's own calibration"],
                value="Shared thresholds (fair comparison)",
                label="Threshold mode",
                info="Shared = both models scored with the same decision rule (isolates real "
                     "model quality). Own = each model's own val-fit thresholds (closer to how "
                     "each would actually be deployed alone, but can make 'comparison' look "
                     "artificially competitive - see THESIS_LOG.md).",
            )
        text_box = gr.Textbox(label="Title + abstract", lines=8, max_lines=20, placeholder="Paste or type text here (max 3000 characters)...")
        predict_btn = gr.Button("Predict", variant="primary")
        true_tags_box = gr.Markdown()
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Best (checkpoint-145404)")
                best_tags_box = gr.Textbox(label="Predicted tags", interactive=False)
                best_top_box = gr.Textbox(label="Top-12 scores", interactive=False, lines=12)
            with gr.Column():
                gr.Markdown("### Comparison (checkpoint-117000)")
                cmp_tags_box = gr.Textbox(label="Predicted tags", interactive=False)
                cmp_top_box = gr.Textbox(label="Top-12 scores", interactive=False, lines=12)

        example_dropdown.change(fill_example, example_dropdown, text_box)
        predict_btn.click(
            predict,
            inputs=[text_box, threshold_mode, example_dropdown],
            outputs=[best_tags_box, best_top_box, cmp_tags_box, cmp_top_box, true_tags_box],
        )
    return demo


if __name__ == "__main__":
    build_ui().launch()
