import json
from pathlib import Path
from collections import Counter

def load_tags(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tag_set = set()
    for entry in data:
        tag_set.update(entry.get("combined_tags", []))
    return tag_set

def main():
    base = Path("boa_strangling/data/data_split_fi_eng_min5_abs40")

    train_tags = load_tags(base / "train.json")
    val_tags = load_tags(base / "val.json")
    test_tags = load_tags(base / "test.json")

    print("=== Label Coverage Report ===")
    print(f"Train tags: {len(train_tags)}")
    print(f"Val tags: {len(val_tags)}")
    print(f"Test tags: {len(test_tags)}")

    val_missing = val_tags - train_tags
    test_missing = test_tags - train_tags

    print(f"\n[Val Split]")
    print(f"  Tags unseen in train: {len(val_missing)}")
    print(f"  % of val tags missing in train: {100 * len(val_missing) / len(val_tags):.2f}%")

    print(f"\n[Test Split]")
    print(f"  Tags unseen in train: {len(test_missing)}")
    print(f"  % of test tags missing in train: {100 * len(test_missing) / len(test_tags):.2f}%")

    print("\nUnseen val tags (sample):", list(val_missing)[:10])
    print("Unseen test tags (sample):", list(test_missing)[:10])

if __name__ == "__main__":
    main()
