#!/usr/bin/env python3
"""
generate_variants_final.py

Generate a split, filtered multi-label dataset variant from full_dataset.json,
with options set directly in this file. Outputs to a clean folder inside
boa_strangling/data/, auto-named based on active settings.
"""

import json, random, re, datetime
from pathlib import Path
from collections import Counter, defaultdict

### ==== CONFIG SECTION: SET OPTIONS HERE ====
OPTIONS = {
    "input": "boa_strangling/data/full_dataset.json",  # path to full dataset
    "min_tag_freq": 15,
    "min_abstract_words": 100,
    "keep_title": True,
    "add_lang_token": True,
    "upsample_rare": True,
    "upsample_threshold": 100,
    "split_by_language": False,
    "train_frac": 0.8,
    "val_frac": 0.1,
    "seed": 42
}
### ==== END CONFIG SECTION ====

def normalize_text(text):
    return re.sub(r"\s+", " ", text).strip()

def word_count(text):
    return len(text.split())

def load_data(path):
    return json.load(open(path, encoding="utf-8"))

def filter_language(data, langs):
    return [d for d in data if d.get("abstract_language") in langs]

def build_vocab(data, min_freq):
    c = Counter()
    for d in data:
        tags = d["subject_tags"] + d["additional_tags"]
        c.update(tags)
    return {t for t, f in c.items() if f >= min_freq}

def filter_and_prepare(data, vocab, opts):
    out = []
    for d in data:
        abs_ = normalize_text(d.get("abstract", [""])[0])
        if word_count(abs_) < opts["min_abstract_words"]:
            continue
        tags = [t for t in set(d["subject_tags"] + d["additional_tags"]) if t in vocab]
        if not tags:
            continue
        title = normalize_text(d.get("thesis_title", [""])[0]) if opts["keep_title"] else ""
        text = title + (" " + abs_ if title else abs_)
        if opts["add_lang_token"]:
            text = f"<LANG={d['abstract_language']}> " + text
        out.append({
            "id": d["identifier"],
            "text": text,
            "tags": tags,
            "lang": d["abstract_language"]
        })
    return out

def upsample_rare(data, threshold):
    buckets = defaultdict(list)
    for d in data:
        buckets[len(d["tags"])].append(d)
    out = list(data)
    for tagset_size, items in buckets.items():
        if len(items) < threshold:
            out.extend(random.choices(items, k=threshold - len(items)))
    return out

def split_data(data, opts):
    random.seed(opts["seed"])
    if opts["split_by_language"]:
        gens = {}
        for L in set(d["lang"] for d in data):
            sub = [d for d in data if d["lang"] == L]
            random.shuffle(sub)
            n = len(sub)
            t = int(n * opts["train_frac"])
            v = int(n * (opts["train_frac"] + opts["val_frac"]))
            gens.setdefault("train", []).extend(sub[:t])
            gens.setdefault("val",   []).extend(sub[t:v])
            gens.setdefault("test",  []).extend(sub[v:])
        return gens["train"], gens["val"], gens["test"]
    else:
        random.shuffle(data)
        n = len(data)
        t = int(n * opts["train_frac"])
        v = int(n * (opts["train_frac"] + opts["val_frac"]))
        return data[:t], data[t:v], data[v:]

def check_coverage(train, val, test):
    tset = {tag for d in train for tag in d["tags"]}
    vset = {tag for d in val   for tag in d["tags"]}
    qset = {tag for d in test  for tag in d["tags"]}
    missing_v = vset - tset
    missing_t = qset - tset
    return len(missing_v), len(vset), len(missing_t), len(qset)

def build_output_name(opts):
    parts = [
        f"f{opts['min_tag_freq']}",
        f"abs{opts['min_abstract_words']}"
    ]
    if opts["keep_title"]:
        parts.append("title")
    if opts["add_lang_token"]:
        parts.append("langtoken")
    if opts["upsample_rare"]:
        parts.append(f"upsample{opts['upsample_threshold']}")
    if opts["split_by_language"]:
        parts.append("langsplit")
    date = datetime.date.today().isoformat()
    return "data_" + "_".join(parts) + "_" + date

def main():
    opts = OPTIONS
    base_dir = Path("boa_strangling/data")
    suffix = build_output_name(opts)
    outdir = base_dir / suffix

    if outdir.exists() and any(outdir.iterdir()):
        raise SystemExit(f"ERROR: Output dir '{outdir}' already exists and is not empty.")
    outdir.mkdir(parents=True, exist_ok=True)

    # Load + process
    raw = load_data(opts["input"])
    original_count = len(raw)

    data = filter_language(raw, {"eng", "fin"})
    after_lang_count = len(data)

    vocab_counter = Counter()
    for d in data:
        vocab_counter.update(d["subject_tags"] + d["additional_tags"])
    vocab = {tag for tag, freq in vocab_counter.items() if freq >= opts["min_tag_freq"]}
    unique_valid_tags = len(vocab)

    data = filter_and_prepare(data, vocab, opts)
    after_tag_filter_count = len(data)

    if opts["upsample_rare"]:
        before_upsample = len(data)
        data = upsample_rare(data, opts["upsample_threshold"])
        after_upsample = len(data)
    else:
        before_upsample = after_upsample = len(data)

    final_count = len(data)

    train, val, test = split_data(data, opts)
    train_len, val_len, test_len = len(train), len(val), len(test)

    for name, split in [("train", train), ("val", val), ("test", test)]:
        json.dump(split, open(outdir / f"{name}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(sorted(vocab), open(outdir / "tag_vocab.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    mv, tv, mt, tt = check_coverage(train, val, test)

    print(f"\n✅ Output written to: {outdir}")
    print("Label coverage:")
    print(f"  Val unseen tags:  {mv}/{tv} ({100*mv/tv:.2f}%)")
    print(f"  Test unseen tags: {mt}/{tt} ({100*mt/tt:.2f}%)")

    # Save metadata
    metadata = {
        "produced_by": "generate_variants_final.py",
        "date": datetime.date.today().isoformat(),
        "source": str(Path(opts["input"]).resolve()),
        "output_dir": str(outdir.resolve()),
        "filters": {
            "language": ["eng", "fin"],
            "min_tag_freq": opts["min_tag_freq"],
            "min_abstract_words": opts["min_abstract_words"]
        },
        "split_ratio": {
            "train": opts["train_frac"],
            "val": opts["val_frac"],
            "test": 1 - opts["train_frac"] - opts["val_frac"]
        },
        "statistics": {
            "original_count": original_count,
            "after_language_filter": after_lang_count,
            "after_tag_filter": after_tag_filter_count,
            "final_count": final_count,
            "unique_valid_tags": unique_valid_tags,
            "train_count": train_len,
            "val_count": val_len,
            "test_count": test_len,
            "upsampled_from": before_upsample,
            "upsampled_to": after_upsample
        },
        "label_coverage": {
            "val_unseen_tags": {
                "count": mv,
                "total": tv,
                "percent": round(100 * mv / tv, 2)
            },
            "test_unseen_tags": {
                "count": mt,
                "total": tt,
                "percent": round(100 * mt / tt, 2)
            }
        },
        "tag_stats": {
            "top_tags": vocab_counter.most_common(20)
        }
    }

    json.dump(metadata, open(outdir / "metadata.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
