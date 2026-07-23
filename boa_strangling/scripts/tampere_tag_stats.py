"""Tag dictionary (tag -> count) and long-tail distribution plot for aligned Tampere data.

Run: python boa_strangling/scripts/tampere_tag_stats.py
"""
import json
from collections import Counter

import matplotlib.pyplot as plt

IN_PATH = "boa_strangling/data/tampere_aligned.jsonl"
COUNTS_PATH = "boa_strangling/stats/tags/tampere_tag_counts.json"
METADATA_PATH = "boa_strangling/stats/tags/tampere_tag_metadata.json"
PLOT_PATH = "boa_strangling/stats/tampere_tag_long_tail_distribution.png"

BAR_COLOR = "#4C72B0"  # single sequential hue -- magnitude, not identity, so one color throughout


def main():
    tag_counts = Counter()
    entities_with_tags = 0
    total = 0

    with open(IN_PATH, encoding="utf-8") as f:
        for line in f:
            entity = json.loads(line)
            total += 1
            tags = entity.get("tags") or []
            if tags:
                entities_with_tags += 1
            tag_counts.update(tags)

    sorted_counts = tag_counts.most_common()

    with open(COUNTS_PATH, "w", encoding="utf-8") as f:
        json.dump(dict(sorted_counts), f, indent=2, ensure_ascii=False)

    metadata = {
        "entities_total": total,
        "entities_with_tags": entities_with_tags,
        "unique_tags_total": len(tag_counts),
        "tag_occurrences_total": sum(tag_counts.values()),
        "top_20_tags": sorted_counts[:20],
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(json.dumps(metadata, indent=2, ensure_ascii=False))

    # Long-tail distribution: rank (x) vs count (y), log-log -- a straight-ish
    # downward line is the "long tail" signature; steep initial drop = few dominant tags.
    ranks = range(1, len(sorted_counts) + 1)
    counts = [c for _, c in sorted_counts]

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.plot(ranks, counts, color=BAR_COLOR, linewidth=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Tag rank (log scale)")
    ax.set_ylabel("Occurrence count (log scale)")
    ax.set_title(f"Tampere tag long-tail distribution ({len(sorted_counts)} unique tags)")
    ax.grid(True, which="both", linewidth=0.4, alpha=0.4)

    # Top-5 ranks sit close together in both x and y at this scale -- inline text
    # labels there collide. Mark with small numbers instead, full names in a legend box.
    top5 = sorted_counts[:5]
    legend_lines = []
    for rank, (tag, count) in enumerate(top5, start=1):
        ax.plot(rank, count, "o", color=BAR_COLOR, markersize=5)
        ax.annotate(str(rank), xy=(rank, count), xytext=(0, 8),
                    textcoords="offset points", fontsize=9, ha="center", fontweight="bold")
        short = tag if len(tag) <= 70 else tag[:67] + "..."
        legend_lines.append(f"{rank}. {short} ({count})")

    ax.text(0.98, 0.97, "\n".join(legend_lines), transform=ax.transAxes,
            fontsize=7.5, va="top", ha="right", family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7", alpha=0.9))

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    print(f"wrote {PLOT_PATH}")


if __name__ == "__main__":
    main()
