import json
import random
import re
from collections import Counter
from pathlib import Path
from datetime import date

SEED = 42
MIN_TAG_FREQ = 5
MIN_ABSTRACT_WORDS = 40
VALID_LANGUAGES = {"fin", "eng"}

INPUT_PATH = Path("boa_strangling/data/full_dataset.json")
OUTPUT_DIR = Path("boa_strangling/data/data_split_fi_eng_min5_abs40/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    with open(INPUT_PATH, encoding="utf-8") as f:
        return json.load(f)

def normalize_text(text):
    return re.sub(r"\s+", " ", text).strip()

def word_count(text):
    return len(text.split())

def filter_language(data):
    return [d for d in data if d.get("abstract_language") in VALID_LANGUAGES]

def build_tag_vocab(data, min_freq):
    counter = Counter()
    for d in data:
        tags = d.get("subject_tags", []) + d.get("additional_tags", [])
        counter.update(tags)
    return {tag for tag, freq in counter.items() if freq >= min_freq}

def filter_data(data, valid_tags):
    filtered = []
    for d in data:
        abstract = normalize_text(d.get("abstract", [""])[0])
        if word_count(abstract) < MIN_ABSTRACT_WORDS:
            continue

        tags = list(set([
            tag for tag in d.get("subject_tags", []) + d.get("additional_tags", [])
            if tag in valid_tags
        ]))

        if not tags:
            continue

        filtered.append({
            "identifier": d.get("identifier"),
            "title": d.get("thesis_title", [""])[0],
            "abstract": abstract,
            "language": d.get("abstract_language"),
            "combined_tags": tags
        })
    return filtered

def split_data(data, train_frac=0.8, val_frac=0.1):
    random.seed(SEED)
    random.shuffle(data)
    n = len(data)
    return data[:int(n*train_frac)], data[int(n*train_frac):int(n*(train_frac+val_frac))], data[int(n*(train_frac+val_frac)):]

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def main():
    data = load_data()
    stats = {
        "original_count": len(data)
    }

    data = filter_language(data)
    stats["after_language_filter"] = len(data)

    valid_tags = build_tag_vocab(data, MIN_TAG_FREQ)

    filtered = filter_data(data, valid_tags)
    stats["after_tag_filter"] = len(filtered)

    train, val, test = split_data(filtered)
    stats["final_count"] = len(filtered)
    stats["unique_valid_tags"] = len(valid_tags)

    save_json(train, OUTPUT_DIR / "train.json")
    save_json(val, OUTPUT_DIR / "val.json")
    save_json(test, OUTPUT_DIR / "test.json")
    save_json(sorted(list(valid_tags)), OUTPUT_DIR / "tag_vocab.json")

    metadata = {
        "produced_by": "prep_split_dataset_filtered.py",
        "date": str(date.today()),
        "source": str(INPUT_PATH),
        "output_dir": str(OUTPUT_DIR),
        "filters": {
            "language": list(VALID_LANGUAGES),
            "min_tag_freq": MIN_TAG_FREQ,
            "min_abstract_words": MIN_ABSTRACT_WORDS
        },
        "split_ratio": {
            "train": 0.8,
            "val": 0.1,
            "test": 0.1
        },
        "statistics": stats
    }

    save_json(metadata, OUTPUT_DIR / "metadata.json")
    print("✅ Dataset written to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
