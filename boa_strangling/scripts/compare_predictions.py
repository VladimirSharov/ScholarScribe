"""Compare predictions from two runs, model-agnostic (any two sources that
produce records shaped like {id, true_tags, predicted_tags, ...} - the schema
shared by train.py's sample_predictions_report.json and predict.py's output).
This is what lets an xlm-roberta run be compared against a llama-based run
later, as long as the llama side is dumped in the same shape.

Usage:
    python3 boa_strangling/scripts/compare_predictions.py \\
        --a boa_strangling/results/xlm_roberta_multilabel_20260716_062451 \\
        --b boa_strangling/results/xlm_roberta_multilabel_20260717_205258

    # or point directly at report JSON files (predict.py output, or a
    # hand-built llama report):
    python3 boa_strangling/scripts/compare_predictions.py --a runA/report.json --b runB/report.json
"""
import argparse
import json
import os


def load_records(path):
    """Accept a results run_dir (locates predictions/sample_predictions_report.json,
    falling back to the newest predict_*.json), or a direct path to a report JSON
    (either a bare list, or {"results": [...]} as produced by predict.py)."""
    if os.path.isdir(path):
        default = os.path.join(path, "predictions", "sample_predictions_report.json")
        if os.path.exists(default):
            path = default
        else:
            pred_dir = os.path.join(path, "predictions")
            candidates = sorted(f for f in os.listdir(pred_dir) if f.startswith("predict_"))
            if not candidates:
                raise FileNotFoundError(f"No sample_predictions_report.json or predict_*.json under {pred_dir}")
            path = os.path.join(pred_dir, candidates[-1])

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    records = data["results"] if isinstance(data, dict) and "results" in data else data
    return path, records


def prf(true_tags, pred_tags):
    true_set, pred_set = set(true_tags), set(pred_tags)
    inter = true_set & pred_set
    precision = len(inter) / len(pred_set) if pred_set else 0.0
    recall = len(inter) / len(true_set) if true_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def align(records_a, records_b):
    """Align by id when both sides have (non-null) ids; else fall back to
    positional zip with a warning, since index order isn't guaranteed to
    match across independently-generated reports."""
    ids_a = [r.get("id") for r in records_a]
    ids_b = [r.get("id") for r in records_b]
    if all(ids_a) and all(ids_b):
        by_id_b = {r["id"]: r for r in records_b}
        pairs = [(r, by_id_b[r["id"]]) for r in records_a if r["id"] in by_id_b]
        missing = len(records_a) - len(pairs)
        if missing:
            print(f"Note: {missing}/{len(records_a)} records in A had no id match in B; comparing the remaining {len(pairs)}.")
        return pairs
    print("Warning: one or both reports lack ids for every record; aligning positionally (index order), which may not correspond to the same samples.")
    return list(zip(records_a, records_b))


def summarize(label, records):
    precisions, recalls, f1s = [], [], []
    for r in records:
        if "true_tags" not in r:
            continue
        p, rec, f1 = prf(r["true_tags"], r.get("predicted_tags", []))
        precisions.append(p)
        recalls.append(rec)
        f1s.append(f1)
    n = len(precisions)
    if n == 0:
        print(f"{label}: no records with true_tags, skipping metric summary")
        return
    print(f"{label}: n={n}  precision={sum(precisions)/n:.4f}  recall={sum(recalls)/n:.4f}  f1={sum(f1s)/n:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="results run_dir or report JSON path")
    parser.add_argument("--b", required=True, help="results run_dir or report JSON path")
    parser.add_argument("--label_a", default="A")
    parser.add_argument("--label_b", default="B")
    parser.add_argument("--output", help="where to write the full per-sample diff JSON")
    parser.add_argument("--show", type=int, default=5, help="how many disagreement examples to print")
    args = parser.parse_args()

    path_a, records_a = load_records(args.a)
    path_b, records_b = load_records(args.b)
    print(f"{args.label_a}: {path_a} ({len(records_a)} records)")
    print(f"{args.label_b}: {path_b} ({len(records_b)} records)\n")

    summarize(args.label_a, records_a)
    summarize(args.label_b, records_b)

    pairs = align(records_a, records_b)

    diffs = []
    both_correct = a_only_correct = b_only_correct = neither_correct = 0
    jaccards = []

    for ra, rb in pairs:
        true_tags = set(ra.get("true_tags", []))
        pred_a = set(ra.get("predicted_tags", []))
        pred_b = set(rb.get("predicted_tags", []))

        union = pred_a | pred_b
        jaccard = len(pred_a & pred_b) / len(union) if union else 1.0
        jaccards.append(jaccard)

        a_hit = bool(true_tags & pred_a)
        b_hit = bool(true_tags & pred_b)
        if a_hit and b_hit:
            both_correct += 1
        elif a_hit:
            a_only_correct += 1
        elif b_hit:
            b_only_correct += 1
        else:
            neither_correct += 1

        diffs.append({
            "id": ra.get("id"),
            "text": ra.get("text"),
            "true_tags": list(true_tags),
            f"predicted_{args.label_a}": list(pred_a),
            f"predicted_{args.label_b}": list(pred_b),
            "only_in_" + args.label_a: list(pred_a - pred_b),
            "only_in_" + args.label_b: list(pred_b - pred_a),
            "shared": list(pred_a & pred_b),
            "jaccard": jaccard,
        })

    n = len(pairs)
    if n:
        print(f"\nAligned samples: {n}")
        print(f"Avg tag-set jaccard overlap ({args.label_a} vs {args.label_b}): {sum(jaccards)/n:.4f}")
        print(f"Samples where both got >=1 true tag right: {both_correct}/{n}")
        print(f"Samples where only {args.label_a} got a true tag right: {a_only_correct}/{n}")
        print(f"Samples where only {args.label_b} got a true tag right: {b_only_correct}/{n}")
        print(f"Samples where neither got any true tag right: {neither_correct}/{n}")

        shown = 0
        for d in diffs:
            if d["only_in_" + args.label_a] or d["only_in_" + args.label_b]:
                if shown >= args.show:
                    break
                print(f"\n[{d['id']}] {str(d['text'])[:120]}")
                print(f"  true:        {d['true_tags']}")
                print(f"  only {args.label_a}:  {d['only_in_' + args.label_a]}")
                print(f"  only {args.label_b}:  {d['only_in_' + args.label_b]}")
                shown += 1

    out_path = args.output or "boa_strangling/results/comparison_report.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "a": path_a, "b": path_b,
            "n_aligned": n,
            "avg_jaccard": sum(jaccards) / n if n else None,
            "both_correct": both_correct, "a_only_correct": a_only_correct,
            "b_only_correct": b_only_correct, "neither_correct": neither_correct,
            "diffs": diffs,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nFull comparison saved to: {out_path}")


if __name__ == "__main__":
    main()
