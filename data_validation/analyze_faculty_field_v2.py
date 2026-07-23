import json
from collections import defaultdict, Counter
from pathlib import Path

# Input
INPUT_FILE = "data_split_v4/full_dataset.json"
OUTPUT_DIR = Path("outputs_2")
OUTPUT_DIR.mkdir(exist_ok=True)

# Standard faculty mapping
faculty_mapping = {
    "Faculty of Humanities and Social Sciences": "Humanities and Social Sciences",
    "Humanistis-yhteiskuntatieteellinen tiedekunta": "Humanities and Social Sciences",
    "Faculty of Information Technology": "Information Technology",
    "Informaatioteknologian tiedekunta": "Information Technology",
    "Faculty of Education and Psychology": "Education and Psychology",
    "Kasvatustieteiden ja psykologian tiedekunta": "Education and Psychology",
    "Faculty of Sport and Health Sciences": "Sport and Health Sciences",
    "Liikuntatieteellinen tiedekunta": "Sport and Health Sciences",
    "Faculty of Mathematics and Science": "Mathematics and Science",
    "Faculty of Sciences": "Mathematics and Science",
    "Matemaattis-luonnontieteellinen tiedekunta": "Mathematics and Science",
    "Faculty of Education": "Education",
    "Kasvatustieteiden tiedekunta": "Education",
    "Faculty of Humanities": "Humanities",
    "Humanistinen tiedekunta": "Humanities",
    "Faculty of Social Sciences": "Social Sciences",
    "Yhteiskuntatieteellinen tiedekunta": "Social Sciences",
    "Kauppakorkeakoulu": "Business and Economics",
    "School of Business and Economics": "Business and Economics",
    "Jyväskylän yliopiston kauppakorkeakoulu": "Business and Economics",
    "Jyväskylä University School of Business and Economics": "Business and Economics"
}

# Processing
seen_identifiers = set()
faculty_counter = Counter()
combination_counter = Counter()
faculty_count_distribution = Counter()
total_faculty_mentions = 0

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    db = json.load(f)

for entry in db:
    oid = entry.get("original_identifier")
    if not oid or oid in seen_identifiers:
        continue
    seen_identifiers.add(oid)

    raw_faculties = entry.get("faculty", [])
    total_faculty_mentions += len(raw_faculties)

    mapped_faculties = set()
    for fac in raw_faculties:
        if fac in faculty_mapping:
            mapped_faculties.add(faculty_mapping[fac])
    
    n = len(mapped_faculties)
    if n == 0:
        continue

    faculty_count_distribution[n] += 1
    for fac in mapped_faculties:
        faculty_counter[fac] += 1

    key = " + ".join(sorted(mapped_faculties))
    combination_counter[key] += 1

# Output
summary = {
    "total_entities": len(db),
    "deduplicated_entities": len(seen_identifiers),
    "total_faculty_mentions": total_faculty_mentions,
    "faculty_count_distribution": dict(faculty_count_distribution)
}

with open(OUTPUT_DIR / "faculty_summary_cleaned.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

with open(OUTPUT_DIR / "faculty_value_counts_cleaned.json", "w", encoding="utf-8") as f:
    json.dump(faculty_counter.most_common(), f, indent=2, ensure_ascii=False)

with open(OUTPUT_DIR / "faculty_combination_counts_cleaned.json", "w", encoding="utf-8") as f:
    json.dump(combination_counter.most_common(), f, indent=2, ensure_ascii=False)

print("✅ Cleaned faculty count saved to /outputs")
