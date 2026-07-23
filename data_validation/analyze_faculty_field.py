import json
from collections import Counter, defaultdict
from pathlib import Path

# --- File Paths ---
input_path = Path("data_split_v4/full_dataset.json")
summary_path = Path("outputs/faculty_stats_summary.json")
value_counts_path = Path("outputs/faculty_value_counts.json")
combo_counts_path = Path("outputs/faculty_combinations.json")

# --- Load Dataset ---
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# --- Initialize Stats ---
total_entities = len(data)
entities_with_faculty = 0
total_faculty_fields = 0

faculty_value_counter = Counter()
faculty_combo_counter = Counter()
faculty_count_distribution = Counter()

# --- Process Records ---
for entity in data:
    faculty_values = entity.get("faculty", None)

    if faculty_values:
        entities_with_faculty += 1

        # Ensure list type
        if isinstance(faculty_values, str):
            faculty_values = [faculty_values]
        elif not isinstance(faculty_values, list):
            continue  # Skip bad format

        total_faculty_fields += len(faculty_values)

        # Count each faculty name individually
        for f in faculty_values:
            faculty_value_counter[f.strip()] += 1

        # Count combination patterns (sorted to normalize ordering)
        unique_combo = tuple(sorted(set(f.strip() for f in faculty_values)))
        faculty_combo_counter[unique_combo] += 1
        faculty_count_distribution[len(unique_combo)] += 1

# --- Save Outputs ---

# Summary stats
summary = {
    "total_entities": total_entities,
    "entities_with_faculty": entities_with_faculty,
    "total_faculty_fields": total_faculty_fields,
    "faculty_count_distribution": dict(faculty_count_distribution)
}
summary_path.parent.mkdir(parents=True, exist_ok=True)
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

# Individual faculty counts
with open(value_counts_path, "w", encoding="utf-8") as f:
    json.dump(faculty_value_counter.most_common(), f, indent=2, ensure_ascii=False)

# Combination patterns
combo_output = {
    " + ".join(combo): count for combo, count in faculty_combo_counter.most_common()
}
with open(combo_counts_path, "w", encoding="utf-8") as f:
    json.dump(combo_output, f, indent=2, ensure_ascii=False)

print("Analysis complete. Outputs written to /outputs/")
