import json
import re
from pathlib import Path
from collections import Counter

# Paths
OLD_DATASET_PATH = Path("data_split_v5/full_dataset.json")
LOG_PATH = Path("language_analysis_log.txt")
NEW_DATASET_PATH = Path("data_split_v6/full_dataset.json")

# Load previous dataset
with open(OLD_DATASET_PATH, encoding='utf-8') as f:
    data = json.load(f)

# Prepare log map
log_lines = LOG_PATH.read_text(encoding="utf-8").splitlines()
log_map = {}
stats = Counter()

for line in log_lines:
    if line.startswith("[ADD]") or line.startswith("[DISAGREE]") or line.startswith("[SKIP]"):
        match = re.search(r'\[(\w+)] (https?://[^\s]+): (.+)', line)
        if match:
            kind, url, detail = match.groups()
            log_map[url] = {"type": kind, "detail": detail}
            stats[kind] += 1

# Update dataset
updated = []
confusion = Counter()

for entry in data:
    url = entry.get("identifier")
    log_entry = log_map.get(url)

    if log_entry:
        kind = log_entry["type"]
        detail = log_entry["detail"]

        if kind == "SKIP":
            # Nothing to update
            pass

        elif kind == "ADD":
            match = re.search(r'language=([a-z]{3})', detail)
            if match:
                new_lang = match.group(1)
                entry["abstract_language"] = new_lang

        elif kind == "DISAGREE":
            field_lang_match = re.search(r'field=([a-z]{3}), detected=\((\w+), (\w+)\)', detail)
            if field_lang_match:
                old_lang, det1, det2 = field_lang_match.groups()
                # Replace with first detection
                entry["abstract_language"] = det1

                # Track mismatch stats
                key = f"{old_lang}->{det1}"
                confusion[key] += 1

    updated.append(entry)

NEW_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
# Save new dataset
with open(NEW_DATASET_PATH, "w", encoding="utf-8") as f:
    json.dump(updated, f, ensure_ascii=False, indent=2)

# Print stats
print("✅ Dataset updated: data_split_v6")
print("\n📊 Summary:")
print(f"  Total entries: {len(updated)}")
print(f"  Added language: {stats['ADD']}")
print(f"  Disagreed/fixed: {stats['DISAGREE']}")
print(f"  Skipped (empty abstract): {stats['SKIP']}\n")

if confusion:
    print("🧪 Disagreement breakdown:")
    for k, v in confusion.items():
        print(f"  {k}: {v} cases")
