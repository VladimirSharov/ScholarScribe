"""Length stats (min/max/avg, word + char) for string fields in the aligned Tampere
data. One abstract = one entity, so these are computed post-split, not per raw thesis.

Run: python boa_strangling/scripts/compute_tampere_field_stats.py
"""
import json

IN_PATH = "boa_strangling/data/tampere_aligned.jsonl"
OUT_PATH = "boa_strangling/stats/tampere_field_stats.json"

STRING_FIELDS = ["title", "abstract"]


def summarize(lengths):
    n = len(lengths)
    return {
        "count": n,
        "min": min(lengths),
        "max": max(lengths),
        "avg": sum(lengths) / n,
    }


def main():
    char_lengths = {f: [] for f in STRING_FIELDS}
    word_lengths = {f: [] for f in STRING_FIELDS}
    total = 0

    with open(IN_PATH, encoding="utf-8") as f:
        for line in f:
            entity = json.loads(line)
            total += 1
            for field in STRING_FIELDS:
                text = entity.get(field) or ""
                char_lengths[field].append(len(text))
                word_lengths[field].append(len(text.split()))

    stats = {"entities": total}
    for field in STRING_FIELDS:
        stats[field] = {
            "char_length": summarize(char_lengths[field]),
            "word_length": summarize(word_lengths[field]),
        }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
