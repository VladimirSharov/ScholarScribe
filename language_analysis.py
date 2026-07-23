import json
from pathlib import Path
from datetime import datetime
from detect_lang_utils import detect_two_langs  # from earlier module

# Paths
DATASET_PATH = Path("data_split_v4/full_dataset.json")
OUTPUT_DATASET_PATH = Path("data_split_v5/full_dataset.json")
LOG_PATH = Path("language_analysis_log.txt")

# For ISO conversion (e.g. fi => fin)
ISO_639_1_TO_3 = {
    "en": "eng",
    "fi": "fin",
    "sv": "swe"
    # Add more as needed
}

# Create output folder if it doesn't exist
OUTPUT_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

# Load data
with open(DATASET_PATH, encoding='utf-8') as f:
    data = json.load(f)

log_lines = []
updated_data = []

def standardize_iso(code):
    return ISO_639_1_TO_3.get(code.lower(), code.lower())

for entry in data:
    abstract_list = entry.get("abstract", [])
    abstract = abstract_list[0] if abstract_list else ""
    current_lang = entry.get("abstract_language", "").lower().strip()

    lang1, lang2 = detect_two_langs(abstract)
    lang1 = standardize_iso(lang1)
    lang2 = standardize_iso(lang2)

    decision = None
    log_reason = ""

    if not abstract.strip():
        log_lines.append(f"[SKIP] No abstract in {entry.get('identifier')}")
        updated_data.append(entry)
        continue

    if current_lang:
        if current_lang == lang1 or current_lang == lang2:
            # Agree with at least one detector
            pass
        else:
            log_reason = f"Disagreement: field={current_lang}, detected=({lang1}, {lang2})"
            log_lines.append(f"[DISAGREE] {entry.get('identifier')}: {log_reason}")
    else:
        if lang1 == lang2 and lang1 != "unknown":
            entry["abstract_language"] = lang1
            log_reason = f"Added language={lang1} from detection"
            log_lines.append(f"[ADD] {entry.get('identifier')}: {log_reason}")
        else:
            log_reason = f"Cannot determine language (detected=({lang1}, {lang2}))"
            log_lines.append(f"[UNSURE] {entry.get('identifier')}: {log_reason}")

    updated_data.append(entry)

# Save updated dataset
with open(OUTPUT_DATASET_PATH, "w", encoding="utf-8") as out_f:
    json.dump(updated_data, out_f, ensure_ascii=False, indent=2)

# Write log
with open(LOG_PATH, "a", encoding="utf-8") as log_f:
    log_f.write(f"\n\n====== LOG START [{datetime.now().isoformat()}] ======\n")
    log_f.write(f"Processed file: {DATASET_PATH}\n")
    log_f.write(f"Output file: {OUTPUT_DATASET_PATH}\n\n")
    log_f.write("\n".join(log_lines))
    log_f.write(f"\n====== LOG END ======\n")

print("✅ Language detection complete.")
print(f"📁 Updated dataset written to: {OUTPUT_DATASET_PATH}")
print(f"📄 Log written to: {LOG_PATH}")
