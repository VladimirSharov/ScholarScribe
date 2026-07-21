# sort_tag_counts.py

import json
from pathlib import Path

def sort_and_write(input_path, output_path):
    with open(input_path, encoding="utf-8") as f:
        tag_counts = json.load(f)

    sorted_tags = dict(sorted(tag_counts.items(), key=lambda x: -x[1]))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sorted_tags, f, ensure_ascii=False, indent=2)

# Sort both subject and additional tags
sort_and_write(
    "boa_strangling/stats/tags/subject_tag_counts.json",
    "boa_strangling/stats/tags/subject_tag_counts_sorted.json"
)
sort_and_write(
    "boa_strangling/stats/tags/additional_tag_counts.json",
    "boa_strangling/stats/tags/additional_tag_counts_sorted.json"
)
