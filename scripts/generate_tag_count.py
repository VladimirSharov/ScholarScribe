# boa_strangling/scripts/generate_tag_counts.py

import json
from collections import Counter
from pathlib import Path

INPUT = Path("boa_strangling/data/full_dataset.json")
OUTPUT_DIR = Path("boa_strangling/stats/tags")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_tag_counts(data, tag_field):
    counter = Counter()
    for entry in data:
        tags = entry.get(tag_field, [])
        counter.update(tags)
    return dict(counter)

def main():
    with open(INPUT, encoding="utf-8") as f:
        data = json.load(f)

    subject_counts = extract_tag_counts(data, "subject_tags")
    additional_counts = extract_tag_counts(data, "additional_tags")

    with open(OUTPUT_DIR / "subject_tag_counts.json", "w", encoding="utf-8") as f:
        json.dump(subject_counts, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_DIR / "additional_tag_counts.json", "w", encoding="utf-8") as f:
        json.dump(additional_counts, f, ensure_ascii=False, indent=2)

    print("✅ Tag dictionaries written to:")
    print(f" - {OUTPUT_DIR / 'subject_tag_counts.json'}")
    print(f" - {OUTPUT_DIR / 'additional_tag_counts.json'}")

if __name__ == "__main__":
    main()
