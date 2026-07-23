import json
from pathlib import Path
from datetime import date

INPUT_DIR = Path("split")
OUTPUT_DIR = Path(f"boa_strangling/data/data_multiuni_{date.today().isoformat()}")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_record(d):
    text = f"<LANG={d['abstract_language']}> {d['title']} {d['abstract']}"
    return {
        "id": d["id"],
        "text": text,
        "tags": d["tags"],
        "lang": d["abstract_language"],
        "university": d.get("university"),
        "source": d.get("source"),
        "faculty": d.get("faculty"),
    }


def main():
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise SystemExit(f"ERROR: Output dir '{OUTPUT_DIR}' already exists and is not empty.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    splits = {}
    for name in ("train", "val", "test"):
        raw = load_jsonl(INPUT_DIR / f"{name}.jsonl")
        splits[name] = [build_record(d) for d in raw]
        json.dump(splits[name], open(OUTPUT_DIR / f"{name}.json", "w", encoding="utf-8"),
                   ensure_ascii=False, indent=2)

    all_tags = set()
    for recs in splits.values():
        for r in recs:
            all_tags.update(r["tags"])

    metadata = {
        "produced_by": "prepare_multiuni_split.py",
        "date": date.today().isoformat(),
        "source": str(INPUT_DIR.resolve()),
        "output_dir": str(OUTPUT_DIR.resolve()),
        "filters": "none (all tags from split/ kept, per explicit decision)",
        "statistics": {
            "train_count": len(splits["train"]),
            "val_count": len(splits["val"]),
            "test_count": len(splits["test"]),
            "total": sum(len(v) for v in splits.values()),
            "unique_tags": len(all_tags),
        },
    }
    json.dump(metadata, open(OUTPUT_DIR / "metadata.json", "w", encoding="utf-8"),
               ensure_ascii=False, indent=2)

    print("Output written to:", OUTPUT_DIR)
    print(metadata["statistics"])


if __name__ == "__main__":
    main()
