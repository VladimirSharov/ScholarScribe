"""Run inference with a trained checkpoint from boa_strangling/results/<run>/.

Usage:
    # sample N random rows from that run's own test set
    python3 boa_strangling/scripts/predict.py --run_dir boa_strangling/results/xlm_roberta_multilabel_20260717_205258 --num_samples 20

    # ad hoc text
    python3 boa_strangling/scripts/predict.py --run_dir <run_dir> --text "Some title. Some abstract..."

    # a file of {"id":..., "text":...} objects, one per line (JSONL)
    python3 boa_strangling/scripts/predict.py --run_dir <run_dir> --input_file my_texts.jsonl

Per-bucket thresholds are refit on that run's own val.json (same frequency bands
as train.py) rather than reusing train-time thresholds, so predict.py works
standalone against just the saved model/ + label_encoder.pkl.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from train import (  # noqa: E402
    FREQ_BIN_NAMES,
    THRESHOLD_GRID,
    compute_tag_bucket_ids,
    find_best_threshold_per_group,
    apply_group_thresholds,
    compute_prf_stats,
    stratified_recall,
    load_data,
)
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification


def fit_thresholds(run_dir, mlb, model_dir=None, threshold_model_dir=None):
    """Fit per-bucket thresholds on val, then return a (possibly different)
    model loaded from model_dir for scoring.

    threshold_model_dir defaults to model_dir - the normal case, where each
    checkpoint calibrates its own decision rule. Passing a different, fixed
    threshold_model_dir lets several checkpoints be scored under one shared
    decision rule, isolating raw output-quality differences between
    checkpoints from differences in how each one happens to calibrate."""
    model_dir = model_dir or os.path.join(run_dir, "model")
    threshold_model_dir = threshold_model_dir or model_dir
    with open(os.path.join(run_dir, "config.json")) as f:
        config = json.load(f)
    data_dir = config["data_dir"]
    train_data, val_data, _ = load_data(data_dir)
    train_labels = mlb.transform([item["tags"] for item in train_data])
    val_labels = mlb.transform([item["tags"] for item in val_data])
    bucket_ids = compute_tag_bucket_ids(train_labels)

    tokenizer_source = os.path.join(run_dir, "model") if not os.path.exists(os.path.join(model_dir, "sentencepiece.bpe.model")) else model_dir
    tokenizer = XLMRobertaTokenizer.from_pretrained(tokenizer_source)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if threshold_model_dir == model_dir:
        model = XLMRobertaForSequenceClassification.from_pretrained(model_dir)
        model.eval()
        model.to(device)
        val_probs = run_inference(model, tokenizer, [item["text"] for item in val_data], device)
    else:
        threshold_model = XLMRobertaForSequenceClassification.from_pretrained(threshold_model_dir)
        threshold_model.eval()
        threshold_model.to(device)
        val_probs = run_inference(threshold_model, tokenizer, [item["text"] for item in val_data], device)
        del threshold_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        model = XLMRobertaForSequenceClassification.from_pretrained(model_dir)
        model.eval()
        model.to(device)

    threshold_map = find_best_threshold_per_group(val_probs, val_labels, bucket_ids, THRESHOLD_GRID)
    return model, tokenizer, device, bucket_ids, threshold_map


def run_inference(model, tokenizer, texts, device, batch_size=32, max_length=512):
    all_probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoding = tokenizer(
                batch, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt"
            ).to(device)
            logits = model(**encoding).logits
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, help="boa_strangling/results/<run> directory")
    parser.add_argument("--model_dir", help="override which weights to load (e.g. <run_dir>/checkpoints/checkpoint-38000); defaults to <run_dir>/model")
    parser.add_argument("--text", help="ad hoc single text to tag")
    parser.add_argument("--input_file", help="JSONL of {id, text} (or {title, abstract}) objects")
    parser.add_argument("--num_samples", type=int, default=20, help="random rows from the run's own test set")
    parser.add_argument("--seed", type=int, help="seed the test-set sampling so two calls (e.g. comparing checkpoints) draw the same rows")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--output", help="where to write the JSON report (default: <run_dir>/predictions/predict_<ts>.json)")
    parser.add_argument("--full_test_eval", action="store_true",
                         help="score the entire test set (not a sample) and report aggregate metrics "
                              "(f1_micro/macro, per-bucket recall/f1) using thresholds fit on val and "
                              "frozen before touching test labels - same schema as train.py's test_results.json")
    parser.add_argument("--threshold_model_dir",
                         help="fit thresholds from this checkpoint's val predictions instead of --model_dir's; "
                              "use to score multiple checkpoints under one shared, fixed decision rule")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    with open(os.path.join(args.run_dir, "label_encoder.pkl"), "rb") as f:
        mlb = pickle.load(f)

    print("Fitting per-bucket thresholds on the run's own val.json..."
          + (f" (using {args.threshold_model_dir}'s calibration)" if args.threshold_model_dir else ""))
    model, tokenizer, device, bucket_ids, threshold_map = fit_thresholds(
        args.run_dir, mlb, args.model_dir, args.threshold_model_dir)
    thresholds_used = {FREQ_BIN_NAMES[b]: float(t) for b, t in threshold_map.items()}
    print(f"Per-bucket thresholds (frozen, fit on val): {thresholds_used}")

    if args.full_test_eval:
        with open(os.path.join(args.run_dir, "config.json")) as f:
            config = json.load(f)
        _, _, test_data = load_data(config["data_dir"])
        test_labels = mlb.transform([item["tags"] for item in test_data])

        print(f"Scoring full test set ({len(test_data)} rows) with frozen thresholds...")
        test_probs = run_inference(model, tokenizer, [item["text"] for item in test_data], device)
        preds_binary = apply_group_thresholds(test_probs, bucket_ids, threshold_map)

        labels_bool = test_labels.astype(bool)
        preds_bool = preds_binary.astype(bool)

        results = compute_prf_stats(labels_bool, preds_bool)
        results.update({
            "avg_predictions_per_sample": preds_binary.sum(axis=1).mean(),
            "avg_true_per_sample": test_labels.sum(axis=1).mean(),
            "total_predictions": int(preds_binary.sum()),
            "total_true": int(test_labels.sum()),
        })
        for bucket_id, thresh in threshold_map.items():
            results[f"threshold_freq_{FREQ_BIN_NAMES[bucket_id]}"] = thresh
        results.update(stratified_recall(labels_bool, preds_bool, bucket_ids, FREQ_BIN_NAMES))
        results = {f"eval_{k}": v for k, v in results.items()}

        print("Full test-set results (frozen val-fit thresholds):")
        for key, value in results.items():
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

        out_path = args.output or os.path.join(
            args.run_dir, "predictions",
            f"full_test_eval_{os.path.basename(args.model_dir) if args.model_dir else 'model'}.json"
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"run_dir": args.run_dir, "model_dir": args.model_dir, "thresholds_used": thresholds_used,
                       "test_results": results}, f, indent=2, ensure_ascii=False)
        print(f"\nSaved full test-set results to: {out_path}")
        return

    items = []
    true_tags_by_index = None

    if args.text:
        items = [{"id": None, "text": args.text}]
    elif args.input_file:
        with open(args.input_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    text = obj.get("text") or f"{obj.get('title', '')} {obj.get('abstract', '')}".strip()
                    items.append({"id": obj.get("id"), "text": text})
    else:
        with open(os.path.join(args.run_dir, "config.json")) as f:
            config = json.load(f)
        _, _, test_data = load_data(config["data_dir"])
        idxs = np.random.choice(len(test_data), min(args.num_samples, len(test_data)), replace=False)
        items = [{"id": test_data[i]["id"], "text": test_data[i]["text"], "true_tags": test_data[i]["tags"]} for i in idxs]

    texts = [it["text"] for it in items]
    probs = run_inference(model, tokenizer, texts, device)
    preds_binary = apply_group_thresholds(probs, bucket_ids, threshold_map)

    results = []
    for i, it in enumerate(items):
        predicted_labels = preds_binary[i]
        predicted_tags = list(mlb.inverse_transform(predicted_labels.reshape(1, -1))[0]) if predicted_labels.sum() > 0 else []
        top_idx = np.argsort(probs[i])[-args.top_k:][::-1]
        top_scores = [(mlb.classes_[j], float(probs[i][j])) for j in top_idx]

        entry = {
            "id": it.get("id"),
            "text": it["text"][:300] + "..." if len(it["text"]) > 300 else it["text"],
            "predicted_tags": predicted_tags,
            "top_predictions_with_scores": top_scores,
            "num_predicted": len(predicted_tags),
        }
        if "true_tags" in it:
            true_tags = it["true_tags"]
            inter = set(true_tags) & set(predicted_tags)
            entry.update({
                "true_tags": list(true_tags),
                "num_true": len(true_tags),
                "intersection": list(inter),
                "precision": len(inter) / len(predicted_tags) if predicted_tags else 0.0,
                "recall": len(inter) / len(true_tags) if true_tags else 0.0,
            })
        results.append(entry)

    out_path = args.output or os.path.join(
        args.run_dir, "predictions", f"predict_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"run_dir": args.run_dir, "thresholds_used": thresholds_used, "results": results}, f, indent=2, ensure_ascii=False)

    for r in results:
        print(f"\n[{r.get('id')}] {r['text'][:120]}...")
        print(f"  predicted: {r['predicted_tags']}")
        if "true_tags" in r:
            print(f"  true:      {r['true_tags']}")
            print(f"  precision={r['precision']:.2f} recall={r['recall']:.2f}")

    print(f"\nSaved {len(results)} predictions to: {out_path}")


if __name__ == "__main__":
    main()
