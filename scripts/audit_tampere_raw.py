"""One streaming pass over tampere_theses.jsonl to ground alignment decisions in real
numbers instead of assumptions. Read-only, no output file mutation of the source data.
Run: python boa_strangling/scripts/audit_tampere_raw.py
"""
import json
from collections import Counter

PATH = "boa_strangling/data/tampere_theses.jsonl"


def main():
    total = 0
    abstract_count_dist = Counter()  # len(abstracts) -> record count
    lang_work = Counter()
    lang_abstract = Counter()
    abstract_langs_seen = Counter()  # each lang inside abstracts[], across all records
    faculties_len_dist = Counter()
    faculty_value_samples = Counter()
    missing_title = 0
    missing_abstract = 0
    missing_tags = 0

    with open(PATH, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            total += 1

            abstracts = rec.get("abstracts") or []
            abstract_count_dist[len(abstracts)] += 1
            for a in abstracts:
                abstract_langs_seen[a.get("lang", "?")] += 1

            lang_work[rec.get("language_work", "?")] += 1
            lang_abstract[rec.get("language_abstract", "?")] += 1

            faculties = rec.get("faculties") or []
            faculties_len_dist[len(faculties)] += 1
            for fac in faculties:
                faculty_value_samples[fac] += 1

            if not rec.get("title"):
                missing_title += 1
            if not rec.get("abstract"):
                missing_abstract += 1
            if not rec.get("tags"):
                missing_tags += 1

    print(f"total records: {total}")
    print(f"missing title: {missing_title}, missing abstract: {missing_abstract}, missing tags: {missing_tags}")
    print()
    print("abstracts-per-record distribution (len(abstracts) -> record count):")
    for k in sorted(abstract_count_dist):
        print(f"  {k}: {abstract_count_dist[k]}")
    print()
    print("language codes seen inside abstracts[] (raw, 2-letter):")
    for k, v in abstract_langs_seen.most_common():
        print(f"  {k}: {v}")
    print()
    print("top-level language_work distribution:")
    for k, v in lang_work.most_common():
        print(f"  {k}: {v}")
    print()
    print("top-level language_abstract distribution:")
    for k, v in lang_abstract.most_common():
        print(f"  {k}: {v}")
    print()
    print("faculties-list-length distribution (len(faculties) -> record count):")
    for k in sorted(faculties_len_dist):
        print(f"  {k}: {faculties_len_dist[k]}")
    print()
    print("most common raw faculty string values (top 20):")
    for k, v in faculty_value_samples.most_common(20):
        print(f"  {v:6d}  {k!r}")


if __name__ == "__main__":
    main()
