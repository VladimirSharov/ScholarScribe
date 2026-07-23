# boa_strangling/scripts/compute_stats.py

# Produces basic tag and text statistics from dataset
# Author: boa_strangling/stats v1
# Input: boa_strangling/data/full_dataset.json
# Output: boa_strangling/stats/stats_summary.json

import os

from utils.io import load_json, save_json
from utils.text_stats import compute_text_stats
from utils.tag_stats import compute_tag_stats

DATA_PATH = os.path.join("boa_strangling", "data", "full_dataset.json")
OUTPUT_PATH = os.path.join("boa_strangling", "stats", "stats_summary.json")

def main():
    data = load_json(DATA_PATH)

    stats = {
        "source_file": DATA_PATH,
        "produced_by": "scripts/compute_stats.py",
        "text_stats": compute_text_stats(data),
        "tag_stats": compute_tag_stats(data),
    }

    save_json(stats, OUTPUT_PATH)
    print("✅ Statistics written to:", OUTPUT_PATH)

if __name__ == "__main__":
    main()
