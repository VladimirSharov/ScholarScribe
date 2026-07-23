"""Align tampere_theses.jsonl to a training-ready schema, one output entity per abstract.

Decisions this implements (see TAMPERE_INTEGRATION_NOTES.md for full rationale):
  - Records with zero abstracts are dropped (no usable text->tags pair).
  - Records with 2+ abstracts are split into one entity per abstract (mirrors JYU's
    split_abstract_v2.py augmentation strategy -- intentional, not deduped away).
  - Records with no tags are KEPT but flagged in the report (not decided yet whether
    to filter these; that's a training-time decision, not an alignment-time one).
  - Language codes converted 2-letter -> 3-letter (fin/eng/swe/... to match JYU convention).
  - No real faculty field yet (see ROADMAP.md #1.6) -- `type` is mapped to `degree_type`
    instead, which is reliable data already present.
  - Every entity keeps `original_id` (== `id` when not split) as a back-pointer to the
    source record, same purpose as JYU's `original_identifier`.

Run: python boa_strangling/scripts/align_tampere_schema.py
"""
import json
import re
from collections import Counter

IN_PATH = "boa_strangling/data/tampere_theses.jsonl"
OUT_PATH = "boa_strangling/data/tampere_aligned.jsonl"
REPORT_PATH = "boa_strangling/data/tampere_alignment_report.json"

LANG_MAP = {
    "fi": "fin", "en": "eng", "sv": "swe",
    "ru": "rus", "fr": "fre", "de": "ger", "it": "ita", "la": "lat", "et": "est",
}


def to_lang3(code):
    if not code:
        return None
    code = code.strip().lower()
    return LANG_MAP.get(code, code)


def parse_bilingual_label(raw):
    """'fi=Väitöskirja | en=Doctoral dissertation|' -> {'fi': ..., 'en': ...}"""
    if not raw:
        return {}
    out = {}
    for part in raw.split("|"):
        part = part.strip()
        if "=" in part:
            lang, _, text = part.partition("=")
            out[lang.strip().lstrip("khyzo")] = text.strip()
    return out


def classify_degree_type(type_raw):
    labels = parse_bilingual_label(type_raw)
    en = (labels.get("en") or "").lower()
    if "bachelor" in en:
        return "bachelor"
    if "master" in en:
        return "master"
    if "doctoral" in en:
        return "doctoral"
    if en:
        return "other"
    return None


def align_record(rec):
    """Yield one or more aligned entities for a single raw Tampere record."""
    abstracts = rec.get("abstracts") or []
    if not abstracts:
        return

    original_id = rec["id"]
    tags = rec.get("tags") or []
    degree_type = classify_degree_type(rec.get("type"))
    split = len(abstracts) > 1

    for idx, a in enumerate(abstracts):
        lang = a.get("lang") or rec.get("language_abstract") or rec.get("language_work")
        entity_id = f"{original_id}::abstract{idx}" if split else original_id
        yield {
            "id": entity_id,
            "original_id": original_id,
            "source": "TAU",
            "record_url": rec.get("record_url"),
            "title": rec.get("title"),
            "abstract": a.get("text"),
            "language": to_lang3(lang),
            "tags": tags,
            "has_tags": bool(tags),
            "degree_type": degree_type,
            "year": rec.get("year"),
        }


def main():
    raw_total = 0
    dropped_no_abstract = 0
    entities_written = 0
    split_records = 0
    entities_missing_tags = 0
    lang_dist = Counter()
    degree_type_dist = Counter()

    with open(IN_PATH, encoding="utf-8") as fin, open(OUT_PATH, "w", encoding="utf-8") as fout:
        for line in fin:
            raw_total += 1
            rec = json.loads(line)

            if not (rec.get("abstracts") or []):
                dropped_no_abstract += 1
                continue

            entities = list(align_record(rec))
            if len(entities) > 1:
                split_records += 1

            for e in entities:
                fout.write(json.dumps(e, ensure_ascii=False) + "\n")
                entities_written += 1
                lang_dist[e["language"]] += 1
                degree_type_dist[e["degree_type"]] += 1
                if not e["has_tags"]:
                    entities_missing_tags += 1

    report = {
        "raw_records_total": raw_total,
        "dropped_no_abstract": dropped_no_abstract,
        "source_records_split_multi_abstract": split_records,
        "entities_written": entities_written,
        "entities_missing_tags": entities_missing_tags,
        "language_distribution": dict(lang_dist.most_common()),
        "degree_type_distribution": dict(degree_type_dist.most_common()),
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
