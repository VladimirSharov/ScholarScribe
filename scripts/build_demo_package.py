"""Build a minimal, portable inference package for boa_strangling/demo/app.py.

Ships only what inference needs: fp16 model weights (small enough to
transfer to a laptop), the tokenizer, the label encoder, and precomputed
per-frequency-bucket thresholds/bucket assignments - so the demo runs
standalone, with no access to the ~200MB training datasets or the
~3.2GB-per-checkpoint training state (optimizer.pt etc.) that predict.py
normally needs to fit thresholds from scratch.

Threshold values are read from the honest checkpoint-comparison outputs
already produced under RUN_DIR/predictions/ (see THESIS_LOG.md, 2026-07-21
entries) rather than recomputed, so the demo's numbers match what's already
been analyzed and written up.

Usage: python3 boa_strangling/scripts/build_demo_package.py
"""
import json
import os
import pickle
import shutil
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from train import load_data, compute_tag_bucket_ids, FREQ_BIN_NAMES  # noqa: E402
from transformers import XLMRobertaForSequenceClassification  # noqa: E402

RUN_DIR = "boa_strangling/results/xlm_roberta_multilabel_20260719_151127"
OUT_DIR = "boa_strangling/demo/package"
CMP3 = f"{RUN_DIR}/predictions/checkpoint_comparison_3way"
CMPFIXED = f"{RUN_DIR}/predictions/checkpoint_comparison_fixed_thresholds_145k"

BEST_CKPT = "checkpoint-145404"
COMPARISON_CKPT = "checkpoint-117000"


def thresholds_used(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["thresholds_used"]


def export_fp16(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    print(f"Loading {src_dir} ...")
    model = XLMRobertaForSequenceClassification.from_pretrained(src_dir, torch_dtype=torch.float16)
    model.save_pretrained(dst_dir, safe_serialization=True)
    print(f"Saved fp16 weights to {dst_dir}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(f"{RUN_DIR}/label_encoder.pkl", "rb") as f:
        mlb = pickle.load(f)
    shutil.copy(f"{RUN_DIR}/label_encoder.pkl", f"{OUT_DIR}/label_encoder.pkl")

    print("Loading train.json to compute label frequency buckets (one-time, not shipped)...")
    with open(f"{RUN_DIR}/config.json") as f:
        config = json.load(f)
    train_data, _, _ = load_data(config["data_dir"])
    train_labels = mlb.transform([item["tags"] for item in train_data])
    bucket_ids = compute_tag_bucket_ids(train_labels)
    del train_data, train_labels
    print(f"Bucket assignment computed for {len(bucket_ids)} labels.")

    best_own = thresholds_used(f"{CMP3}/predict_{BEST_CKPT}_seed42_n100.json")
    comparison_own = thresholds_used(f"{CMP3}/predict_{COMPARISON_CKPT}_seed42_n100.json")
    shared = thresholds_used(f"{CMPFIXED}/full_test_eval_{COMPARISON_CKPT}_fixed145k.json")

    meta = {
        "freq_bin_names": FREQ_BIN_NAMES,
        "bucket_ids": bucket_ids.tolist(),
        "threshold_maps": {
            "best_own": best_own,
            "comparison_own": comparison_own,
            "shared_fixed": shared,
        },
        "source": {
            "run_dir": RUN_DIR,
            "best_checkpoint": BEST_CKPT,
            "comparison_checkpoint": COMPARISON_CKPT,
            "note": "shared_fixed thresholds fit on checkpoint-145000's val predictions; "
                    "see THESIS_LOG.md 2026-07-21 (third entry) for why 'best_own' made "
                    "checkpoint-117000 look more competitive than it actually is.",
        },
    }
    with open(f"{OUT_DIR}/inference_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("Wrote inference_meta.json")

    export_fp16(f"{RUN_DIR}/checkpoints/{BEST_CKPT}", f"{OUT_DIR}/weights/best")
    export_fp16(f"{RUN_DIR}/checkpoints/{COMPARISON_CKPT}", f"{OUT_DIR}/weights/comparison")

    tok_dir = f"{OUT_DIR}/tokenizer"
    os.makedirs(tok_dir, exist_ok=True)
    for fname in ["sentencepiece.bpe.model", "special_tokens_map.json", "tokenizer_config.json"]:
        shutil.copy(f"{RUN_DIR}/model/{fname}", f"{tok_dir}/{fname}")
    print("Copied tokenizer files")

    with open(f"{CMP3}/qualitative_review.json", encoding="utf-8") as f:
        qual = json.load(f)

    examples = []
    seen = set()
    candidates = qual["selected_10"][:4] + list(qual["interesting_per_checkpoint"].values())
    for row in candidates:
        if row["id"] in seen:
            continue
        seen.add(row["id"])
        examples.append({
            "id": row["id"],
            "text": row["text_3000"][:3000],
            "true_tags": row["true_tags"],
        })
        if len(examples) >= 6:
            break
    with open(f"{OUT_DIR}/examples.json", "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(examples)} example texts to examples.json")

    print("\nPackage contents:")
    total = 0
    for root, _, files in os.walk(OUT_DIR):
        for fn in files:
            p = os.path.join(root, fn)
            size = os.path.getsize(p)
            total += size
            print(f"  {p}  ({size / 1e6:.1f} MB)")
    print(f"\nTotal package size: {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
