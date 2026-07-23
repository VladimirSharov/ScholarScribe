# ScholarScribe — Project Roadmap

> Last updated: 2026-07-03

---

## Phase 0 — Cleanup (do first, reduces confusion in all later phases)

### 0.1 Archive superseded files

Move to `depricated_versions/` (already exists as a trash bin, keep using it):

```
llm_generate_tags_sv2.py
llm_generate_tags_sv3.py
llm_generate_tags_sv4.py         → keep sv5.py
model_training_GCV_2.py           → keep GCV_3.py (or boa_strangling/scripts/train.py)
model_training_GCV_singleSoft.py  → verify if experiment is concluded first
data_split/                       → superseded by data_split_v4 → data_split_v6
data_split_v4/
evaluation_results/               → duplicate of 20250507 run; keep 20250518
evaluation_results_20250507_122810/
tag_evaluation_metrics.json (root) → use tag_evaluation_20250422_113039/
tag_visualization_data.json (root)
tag_language_analysis_20250417_095537.json  → keep 20250421
tag_language_analysis_20250417_152813.json
data_validation/tag_evaluation.py  → keep v4 (or v35 depending — see §0.2)
data_validation/tag_evaluation_v2.py
data_validation/tag_evaluation_v3.py
data_validation/analyze_faculty_field.py  → keep v2
testing_output_3.py               → NOTE: _2 is newer despite lower number
trash.py
llama_install.py, llama-test.py, llama-prompt.py  → if llamafile approach abandoned
comprehensive_debug_output/
comprehensive_debug_output1/
```

### 0.2 Decide on one canonical training script

Two competing training scripts exist:
- `model_training_GCV_3.py` — uses HuggingFace `Trainer` with `Datasets` pipeline
- `boa_strangling/scripts/train.py` — uses `MultiLabelBinarizer` + custom `MultiLabelDataset`

**Known issue in `model_training_GCV_3.py`:** the `_create_preprocess_function` stores labels as **lists of tag indices**, not a binary float vector. For `problem_type="multi_label_classification"`, HuggingFace expects labels of shape `[batch_size, num_labels]` with float 0.0/1.0 values. Passing raw index lists causes silent errors in loss computation and makes the checkpoint useless. `boa_strangling/scripts/train.py` handles this correctly via `mlb.transform()`.

Recommendation: treat `boa_strangling/scripts/train.py` as canonical.

### 0.3 Decide on one canonical dataset split

```
data_split/                              ← Dec 2024, oldest, delete
data_split_v4/                           ← Mar 2025, delete
data_split_v6/full_dataset.json          ← May 2025, unsplit source
boa_strangling/data/data_split_min5/     ← min 5 occurrences per tag
boa_strangling/data/data_split_fi_eng_min5_abs40/   ← Finnish+English, abs≥40 chars
boa_strangling/data/data_f15_abs100_*/   ← f=15 faculties, abs≥100 chars
boa_strangling/data/data_f20_abs100_*/   ← f=20 faculties, abs≥100 chars
```

The `boa_strangling/data/` variants were intentional experiments with different filters. Pick one and document it as "the dataset used for training" in README. The `data_split_fi_eng_min5_abs40` was used in the GCV_3 config; the `data_f15_abs100` was the last one used in `boa_strangling/scripts/train.py`.

### 0.4 Fix the .gitignore typo

`__pycahce__` → `__pycache__` — otherwise Python cache dirs ARE committed.

### 0.5 Commit the untracked files that matter

These exist only on disk, not in git:
- `language_analysis.py`, `detect_lang_utils.py`
- `model_training_GCV_3.py`
- `faculty_data_processor.py`, `faculty_visualizer.py`, `run_faculty_analysis.py`
- `rebuild_dataset_from_log.py`, `comparation_language_prompts.py`
- Entire `boa_strangling/` directory

---

## Phase 1 — Add a Second University Dataset

Goal: expand training data with another Finnish university repository (e.g., Tampere, Aalto, Helsinki).

### 1.1 Preparation before you start

- Confirm the target repository has an accessible API (OAI-PMH or REST).
- Check the tag/subject schema — are subjects in the same format as JYU (KOKO ontology, free text, both)?
- Decide whether to keep the two universities separate (for cross-institution experiments) or merge them into one dataset.

### 1.2 Data collection

- Extend or copy `data_collection/api_data_collector.py`.
- Save raw output to a named file e.g. `data_collection/output_TAU.json` — **do not overwrite `output.json`** (JYU source).
- Add a `source` field to every record at collection time (e.g. `"source": "JYU"` / `"TAU"`). Much easier to add now than to backfill later.

### 1.3 Pre-processing — known pitfalls (read before touching)

See Phase 2 for full pre-processing details. For a second dataset specifically:

- **Language detection**: run `language_analysis.py` to assign `abstract_language`. Don't assume all abstracts are Finnish just because the university is Finnish.
- **Tag normalization**: Subject tags from different universities may overlap but differ in case, spelling, or language. Decide whether to normalize (e.g. lowercase all) before building the vocabulary.
- **Faculty field**: the faculty names will be different between universities. Either map them to a shared taxonomy or add a `university` field and keep them separate.
- **Abstract splitting**: `split_abstract_v2.py` handles the JYU pattern where multiple abstracts are stored as a single string with a delimiter. Check whether the new university uses the same pattern.

### 1.4 After merging

- Re-run `boa_strangling/scripts/generate_tag_count.py` to see how the tag distribution changes.
- Re-run `boa_strangling/scripts/check_label_coverage.py` to verify that val/test sets still have 0% unseen tags after the new data is mixed in.
- Re-run the stratified split — the class imbalance shifts when you add a second source.

### 1.5 Current round — Tampere: harvest is done, alignment is not (2026-07-04)

`boa_strangling/tampere_thesis_harvest.py` already completed (66,340 records in
`boa_strangling/data/tampere_theses.jsonl`). Nothing downstream exists yet — no
script in the repo reads that file. Full technical detail (exact field names,
which JYU scripts are pattern-only vs reusable, open schema-gap decisions) is in
`boa_strangling/TAMPERE_INTEGRATION_NOTES.md` — read it before writing code so
existing gotchas (dual tag-field naming, 2-letter vs 3-letter language codes, the
JYU abstract-split double-counting footgun, four inconsistent faculty dicts) aren't
rediscovered from scratch. Cleanup/deletion tasks from Phase 0 are explicitly
**out of scope** for this round.

Order for this round:

1. **Schema alignment** — map `tampere_theses.jsonl` records onto the aligned
   training schema. Resolve the open decisions logged in
   `TAMPERE_INTEGRATION_NOTES.md` §"Open decisions" first (tag field name `tags`
   vs `combined_tags`, `faculty` vs `faculties`, no YSO/free-text tag split
   available for Tampere) — pick once, apply consistently, don't let a script
   default one way and another script assume the opposite. Also apply the same
   multi-abstract splitting JYU uses (`split_abstract_v2.py`'s approach): this is
   intentional augmentation, not a bug to avoid — text→tags is the core task and
   the abstract is the dominant signal, so each language's abstract becomes its
   own (text, tags) training pair instead of picking only the single longest one.
2. **Integrity check** — verify every aligned record has non-empty title,
   abstract, and tags before anything else runs on it. Reuse the pattern from
   `data_validation/json_data_integrity_checker.py` (JYU precedent) adapted to
   Tampere's flat (non-list) field shapes.
3. **Statistics** — compute, on the aligned Tampere data:
   - title/abstract length min/max/avg (word + char) — adapt
     `boa_strangling/scripts/utils/text_stats.py`'s `compute_text_stats()` rather
     than `images/visualization_distro.py`'s (the two disagree on
     all-elements-vs-first-element counting; `text_stats.py` is the one still
     wired into the live pipeline) so JYU/Tampere numbers are comparable.
   - tag count per record + full unique tag vocabulary, written as
     `tampere_tag_vocab.json` + a `metadata.json` with `"unique_valid_tags"` count,
     following the same convention `prep_split_dataset.py`/`generate_variants.py`
     use for JYU.
   - faculty distribution (raw `faculties` value counts — this is also step 0 of
     building the Tampere faculty dictionary, since none of JYU's four existing
     dictionaries will match).
   - language distribution, both raw (2-letter) and after conversion to the JYU
     3-letter convention. **Decided (2026-07-04):** target format is 3-letter codes
     throughout; primary training set for now is fin+eng only (Swedish deferred, not
     blocking). Finna scale reference: fin ≈170k, eng ≈140k, swe+ger ≈14k, rest ≈4k each.
4. **Tag-space similarity, JYU vs Tampere** — the most important comparison and
   currently has zero precedent in the repo (existing `tag_evaluation_v4.py`
   compares generated-vs-truth tags on one record, not vocab-vs-vocab across
   sources). Build a new script reusing `embedding_similarity()`'s pairwise-cosine
   machinery (`paraphrase-multilingual-mpnet-base-v2`) but run over the JYU
   `tag_vocab.json` vs the new `tampere_tag_vocab.json`. Do this *before* deciding
   how to merge — low overlap is itself a reason to keep tag spaces per-source
   rather than unioning them.
5. **Merge decision** — only after 1–4 give real numbers, decide whether to unify
   into one dataset (with a `university`/`source` field) or keep sources separate
   for cross-institution experiments, then proceed with §1.4 above.

Prerequisite audit already run: `boa_strangling/scripts/audit_tampere_raw.py`
(streaming, read-only) — real counts are in `TAMPERE_INTEGRATION_NOTES.md`
§"Ground-truth numbers". Key results that changed the plan: 30% of records have
no abstract at all (dropped at alignment), and the `faculties` field turns out to
carry no faculty data (verified against Trepo's live OAI `ListSets` — the repo's
set hierarchy has no faculty level, only document-type collections).

### 1.6 Deferred follow-up — faculty backfill via dc:contributor

Real faculty/department data exists in Trepo's per-record `dc:contributor` field
(confirmed via `GetRecord`, not currently parsed by `tampere_thesis_harvest.py`,
which only reads OAI set names). Fixing the harvester to extract faculty from
`dc:contributor` (its existing regex `facult|tiedekun|yksikk|school|unit` would
work correctly once pointed at the right field) and re-running the ~20-minute
full harvest is scoped but **intentionally deferred** — schema alignment for
title/abstract/tags proceeds first without a real faculty field (using `type`
→ `degree_type` instead, which is reliable). Do this once alignment/stats/tag-space
work is further along, not before.

---

## Phase 2 — Pre-processing Pipeline (with known issues)

### Canonical order

```
1. data_collection/api_data_collector.py
        → data_collection/output.json

2. data_preparation/v3_data_preparation.py
        → extracts title, abstract, tags, faculty, language fields
        → known issue: field "clean_full_text" is NOT produced here;
          some scripts downstream reference it but it doesn't exist

3. data_preparation/stratify_split_v2.py
        → splits into train/val/test with stratification
        → PITFALL (see §2.1)

4. data_preparation/split_abstract_v2.py
        → some entries contain multiple abstracts as one string; this splits them
        → must run BEFORE language detection, not after

5. language_analysis.py  (+ detect_lang_utils.py)
        → adds abstract_language field per entry
        → writes to data_split_v5/ (creates it; does not exist in git)

6. boa_strangling/scripts/prep_split_dataset.py
        → applies filters (min tag count, min abstract length, language subset)
        → produces boa_strangling/data/<variant>/
```

### 2.1 Split issue — stratification with rare labels

**What went wrong the first time:** stratified splitting fails or produces imbalanced splits when some tags appear only once in the entire dataset. `train_test_split` with `stratify=tags` requires each class to appear at least twice (once in train, once in val/test). With a long-tail tag distribution, many tags appear only 1–4 times total.

**The fix applied:** `boa_strangling/scripts/prep_split_dataset.py` filters out tags with fewer than `min_count` occurrences (currently 5) before splitting. After the split, `check_label_coverage.py` verifies that no val/test tag is absent from the train set.

**When adding new data:** re-check this. The combined dataset may re-introduce rare tags that pass the threshold on their own but cause coverage gaps.

### 2.2 Abstract splitting edge cases

Some thesis entries have both Finnish and English abstracts stored as one concatenated text. `split_abstract_v2.py` detects the delimiter. Edge cases seen:
- Abstract field is null or empty string → script should skip, but verify
- Only one language present → should not split

Run `data_validation/json_data_integrity_checker.py` after splitting to catch malformed entries.

---

## Phase 3 — Model Training Pipeline (with known issues)

### 3.1 The multilabel metric issue (supervisor feedback)

**The issue:** `sklearn.metrics.precision_recall_fscore_support` behaves differently depending on array shape.

- With a **1D array** (single integer per sample): treats it as multiclass — wrong for multilabel.
- With a **2D binary matrix** (samples × labels, 0/1 floats): treats it as multilabel — correct.

If labels are stored as lists of indices (e.g., `[3, 7, 12]` per sample) and then stacked, you get a 2D array where each row has different non-zero positions but the *values* are indices, not binary indicators — `precision_recall_fscore_support` will compute garbage.

**The fix:** always binarize first:
```python
mlb = MultiLabelBinarizer()
mlb.fit([all_tags_list])
train_labels = mlb.transform([[tag_list_per_item]])   # → shape (n_samples, n_labels), float 0/1
```

The `boa_strangling/scripts/train.py` does this correctly. `model_training_GCV_3.py` does NOT.

**For flatten:** if your loss function receives labels of shape `(batch_size, n_labels)` and you accidentally pass shape `(batch_size * n_labels,)` (flattened), the loss gets computed as if each label is an independent binary classification over all samples at once, which collapses the per-sample structure. Do not flatten unless the loss function explicitly expects it.

### 3.2 Model save checklist — the re-training trap

**What kept happening:** model weights were saved, but not the tokenizer or the label encoder, making the checkpoint unusable without re-running training. Each of these must be saved together:

```
Checkpoint directory should contain ALL of:
  ✓ model weights           (trainer.save_model() or model.save_pretrained())
  ✓ tokenizer               (tokenizer.save_pretrained(output_dir))
  ✓ label encoder / vocab   (pickle.dump(mlb) or json.dump(tag2id))
  ✓ config.json             (records num_labels, model_name, threshold)
```

In `boa_strangling/scripts/train.py`:
- `trainer.save_model(model_save_path)` ✓
- `tokenizer.save_pretrained(model_save_path)` ✓
- `pickle.dump(mlb, ...)` to `label_encoder.pkl` ✓

In `model_training_GCV_3.py`:
- tokenizer is saved via `DataCollatorWithPadding` passthrough ✓
- `tag_mapping` saved as pickle ✓
- **but**: the pickle filename includes a timestamp, making it hard to find when you reload

**Recommendation:** always save to a fixed filename (`tag2id.json`, not `tag_mapping_20250521.pickle`) so a reload script doesn't need to know the training time.

### 3.3 Duplicate code in training scripts

`model_training_GCV_3.py` and `boa_strangling/scripts/train.py` both implement:
- Dataset loading
- Label preparation
- Training argument setup
- Metrics computation
- Sample prediction reporting

They diverge in label handling (indices vs binary matrix) and metrics (basic macro vs adaptive threshold). Consider consolidating into one script once the dataset is finalized, keeping only `boa_strangling/scripts/train.py` as the maintained version.

### 3.4 Field name mismatch

`model_training_GCV_3.py` references `examples["clean_full_text"]` in the tokenizer preprocessing function. This field does **not exist** in the `boa_strangling/data/` datasets (they have `text`, `title`, `abstract`). The script will crash at tokenization.

If you run `model_training_GCV_3.py` again, change line 292:
```python
# Wrong:
tokenized_inputs = tokenizer(examples["clean_full_text"], ...)
# Right:
tokenized_inputs = tokenizer(examples["text"], ...)
```

---

## Phase 4 — Evaluation

- Current canonical: `boa_strangling/results/xlm_roberta_multilabel_20250523_*/`
- Evaluation scripts: `data_validation/tag_evaluation_v4.py` (most complete, adaptive threshold, multiple metrics)
- `tag_evaluation_lang.py` adds per-language breakdown — run this after v4 to get language-stratified results

Note: `tag_evaluation_v4.py` requires `Levenshtein`, `jellyfish`, `sentence_transformers` — check these are installed before running.

---

## Summary — priority order

| # | Task | Effort |
|---|------|--------|
| 1 | Fix `.gitignore` typo, commit untracked files | 30 min |
| 2 | Archive superseded scripts/dirs (see §0.1) | 1–2 h |
| 3 | Decide canonical training script (GCV_3 vs boa/train.py) | decision |
| 4 | Fix label binarization in GCV_3 if keeping it (§3.1) | 1 h |
| 5 | Fix save checklist to use fixed filenames (§3.2) | 30 min |
| 6 | Collect second university dataset (§1.2) | days |
| 7 | Run pre-processing pipeline on combined data, verify coverage (§2.1) | 1–2 h |
| 8 | Re-train on combined dataset | GPU time |
| 9 | Evaluate with tag_evaluation_v4.py + lang breakdown | 1 h |
