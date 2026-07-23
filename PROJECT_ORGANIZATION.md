# Project Organization — Duplicate & Versioned Files

> Generated 2026-07-03. Based on file modification timestamps and naming patterns.
> "Latest" = highest version number or most recent mtime. Does not guarantee it's the *best* — check notes.

---

## Root-level scripts

### `llm_generate_tags_sv*.py` — LLM tag generation

| File | Date | Size | Status |
|------|------|------|--------|
| `llm_generate_tags_sv2.py` | Mar 7, 2025 | 7 KB | old |
| `llm_generate_tags_sv3.py` | Dec 13, 2024 | 9.5 KB | old |
| `llm_generate_tags_sv4.py` | Mar 7, 2025 | 19 KB | old |
| `llm_generate_tags_sv5.py` | **Apr 12, 2025** | 13 KB | **LATEST** |

**Recommendation:** `sv5` is the most recent. `sv4` was the largest — may have been simplified in sv5.

---

### `model_training_GCV_*.py` — GCV model training

| File | Date | Size | Status |
|------|------|------|--------|
| `model_training_GCV_2.py` | Feb 27, 2025 | 15 KB | old |
| `model_training_GCV_singleSoft.py` | Feb 28, 2025 | 19 KB | variant (single soft label) |
| `model_training_GCV_3.py` | **May 21, 2025** | 15 KB | **LATEST** |

**Recommendation:** `GCV_3` is the latest. `singleSoft` is a separate experiment variant, not just a version bump.

---

### `testing_output_*.py` — model output testing

| File | Date | Size | Status |
|------|------|------|--------|
| `testing_output_3.py` | Feb 28, 2025 | 3.9 KB | older |
| `testing_output_2.py` | **Apr 11, 2025** | 3.3 KB | **LATEST** (despite lower number!) |

**Warning:** The numbering is reversed here — `_2` is newer than `_3` by 6 weeks. Check content before deleting.

---

### `xlm_roberta_prediction*.json` — XLM-RoBERTa prediction samples

| File | Date | Size | Status |
|------|------|------|--------|
| `xlm_roberta_prediction2.json` | Feb 28, 2025 | 3.3 KB | old |
| `xlm_roberta_prediction4.json` | Feb 28, 2025 | 3.3 KB | old |
| `xlm_roberta_prediction.json` | Feb 28, 2025 | 4.1 KB | old (largest — may be most complete) |
| `xlm_roberta_prediction5.json` | **Apr 11, 2025** | 3.3 KB | **LATEST** |

**Note:** `prediction.json` (no number) is largest but not the most recent. `prediction3.json` is missing — likely deleted.

---

## `data_validation/` — validation scripts

### `tag_evaluation*.py`

| File | Date | Size | Status |
|------|------|------|--------|
| `tag_evaluation.py` | Apr 4, 2025 | 9.5 KB | original |
| `tag_evaluation_v2.py` | Apr 11, 2025 | 7.9 KB | v2 |
| `tag_evaluation_v3.py` | May 9, 2025 | 10 KB | v3 |
| `tag_evaluation_v4.py` | May 9, 2025 | 24 KB | v4 (largest, most complete) |
| `tag_evaluation_v35.py` | **May 18, 2025** | 10.5 KB | **LATEST by date** |
| `tag_evaluation_lang.py` | Apr 22, 2025 | 4.2 KB | language-specific variant |

**Warning:** `v35` (likely "3.5") is the most recently modified but `v4` is much larger (24 KB vs 10 KB). These may serve different purposes. `tag_evaluation_lang.py` is a language-analysis variant, not a direct replacement.

**Recommendation:** `v4` for full evaluation, `v35` for whatever it added on top of v3. Keep `_lang` separately — different purpose.

---

### `analyze_faculty_field*.py`

| File | Date | Size | Status |
|------|------|------|--------|
| `analyze_faculty_field.py` | May 15, 2025 | 2.4 KB | original |
| `analyze_faculty_field_v2.py` | May 15, 2025 | 3.3 KB | **LATEST** (larger, more features) |

**Recommendation:** `v2` — same day but bigger.

---

### `output_json_dataset_field_metrics*.json`

| File | Date | Size | Status |
|------|------|------|--------|
| `output_json_dataset_field_metrics.json` | Nov 15, 2024 | 238 KB | larger |
| `output_json_dataset_field_metrics_5.json` | Nov 15, 2024 | 237 KB | `_5` suffix unclear meaning |

**Note:** Same date, nearly identical size. The `_5` suffix may refer to a dataset version, not a script version. Treat as two separate outputs, not strict duplicates.

---

## `images/` — visualization scripts

### `plot_faculty_summary*.py`

| File | Date | Size | Status |
|------|------|------|--------|
| `plot_faculty_summary.py` | May 15, 2025 | 2.1 KB | original |
| `plot_faculty_summary_v2.py` | May 15, 2025 | 18.6 KB | **LATEST** (much larger — major expansion) |

**Recommendation:** `v2` is clearly the developed version.

---

## Dataset split directories

| Directory | Date | Contents | Status |
|-----------|------|----------|--------|
| `data_split/` | Dec 9, 2024 | train/test/val + split script | oldest |
| `data_split_v4/` | Dec 13, 2024 – Mar 31, 2025 | full_dataset + v3 splits | middle |
| `data_split_v6/` | **May 17, 2025** | full_dataset only | **LATEST** |

**Note:** v5 is missing — either skipped or deleted. `data_split_v6/` has only the unsplit `full_dataset.json`, which may mean the split step hasn't been re-run yet after v6 preparation.

---

## Debug output directories

| Directory | Date | Index | Notes |
|-----------|------|-------|-------|
| `comprehensive_debug_output/` | Feb 28, 2025 | example_0 | first example |
| `comprehensive_debug_output1/` | Feb 28, 2025 | example_1 | second example |

**Note:** These are NOT strict duplicates — they contain debug output for two different input examples (index 0 and 1). Both created Feb 28, 2025. Keep both if you want both examples.

---

## Evaluation result directories

### `evaluation_results*` — tag similarity evaluation

| Directory | Date | Status |
|-----------|------|--------|
| `evaluation_results_20250507_122810/` | May 9, 2025 | timestamped run |
| `evaluation_results/` | May 9, 2025 | appears to be copy of the above |
| `evaluation_results_20250518_100333/` | **May 18, 2025** | **LATEST** |

**Warning:** `evaluation_results/` and `evaluation_results_20250507_122810/` have the same file dates and very similar sizes — likely one was copied from the other. The unnamed `evaluation_results/` is probably a "current" pointer that was not updated when the May 18 run was done.

---

### `tag_evaluation_*` — tag evaluation JSON results

| Location | Date | Status |
|----------|------|--------|
| `tag_evaluation_metrics.json` (root) | Apr 4, 2025 | oldest |
| `tag_evaluation_20250410_114631/` | Apr 11, 2025 | middle |
| `tag_evaluation_20250422_113039/` | **Apr 22, 2025** | **LATEST** |

**Recommendation:** Use `tag_evaluation_20250422_113039/` for current results.

---

## Root-level result JSONs

### `generated_tags_results*.json`

| File | Date | Size | Notes |
|------|------|------|-------|
| `generated_tags_results_separate_tags.json` | Mar 29, 2025 | 1.9 MB | separate-tags strategy |
| `generated_tags_results_union_tags.json` | Apr 11, 2025 | 1.8 MB | union-tags strategy |
| `generated_tags_results.json` | **Apr 14, 2025** | 2.0 MB | **LATEST** (final merged?) |

**Note:** These three likely represent different generation strategies, not pure version progression. Keep all three if you're still comparing strategies.

---

### `tag_language_analysis_*.json`

| File | Date | Size | Status |
|------|------|------|--------|
| `tag_language_analysis_20250417_095537.json` | Apr 22, 2025 | 3.9 MB | first run |
| `tag_language_analysis_20250417_152813.json` | Apr 22, 2025 | 4.2 MB | same day, ~6h later |
| `tag_language_analysis_20250421_105600.json` | Apr 22, 2025 | 4.7 MB | **LATEST & largest** |

**Recommendation:** `_20250421_105600` is the most complete (largest, latest).

---

## Faculty analysis outputs

### `outputs/` vs `outputs_2/`

| Directory | Date | Contents |
|-----------|------|----------|
| `outputs/` | May 15, 2025 | raw faculty stats (combinations, value counts, summary) |
| `outputs_2/` | May 15, 2025 | cleaned versions of same + 3 charts |

**Note:** `outputs_2/` is the cleaned/processed successor. Files in `outputs/` have no `_cleaned` suffix; `outputs_2/` has `_cleaned` versions. Both may be needed as source vs output.

### `outputs_visualizations/`

Separate purpose — full-resolution faculty visualizations (streamgraph, pie, legend). Not a duplicate of `outputs_2/`.

---

## `depricated_versions/field_translation/`

| File | Notes |
|------|-------|
| `translateFieldsNew.py` | original |
| `translateFieldsNew_test.py` | test script |
| `translateFieldsNew_v2.py` | **LATEST** |

Already in `depricated_versions/` — entire folder is old.

---

## Summary: files you can likely archive/delete

These are superseded by a newer version with clear evidence:

| File/Dir | Superseded by |
|----------|--------------|
| `llm_generate_tags_sv2.py` | `sv5.py` |
| `llm_generate_tags_sv3.py` | `sv5.py` |
| `llm_generate_tags_sv4.py` | `sv5.py` |
| `model_training_GCV_2.py` | `GCV_3.py` |
| `data_validation/analyze_faculty_field.py` | `analyze_faculty_field_v2.py` |
| `data_split/` (whole dir) | `data_split_v6/` |
| `data_split_v4/` (whole dir) | `data_split_v6/` |
| `evaluation_results/` (plain dir) | `evaluation_results_20250518_100333/` |
| `evaluation_results_20250507_122810/` | `evaluation_results_20250518_100333/` |
| `tag_evaluation_metrics.json` (root) | `tag_evaluation_20250422_113039/` |
| `tag_visualization_data.json` (root) | `tag_evaluation_20250422_113039/` |
| `tag_language_analysis_20250417_095537.json` | `tag_language_analysis_20250421_105600.json` |
| `tag_language_analysis_20250417_152813.json` | `tag_language_analysis_20250421_105600.json` |

### Things to verify before deleting

- `testing_output_2.py` vs `testing_output_3.py` — numbering is reversed; confirm purpose before touching
- `model_training_GCV_singleSoft.py` — may be a separate experiment, not just an old version
- `tag_evaluation_v4.py` vs `tag_evaluation_v35.py` — different sizes suggest different roles
- `generated_tags_results*.json` (3 files) — likely different strategies, not versions
- `comprehensive_debug_output/` + `comprehensive_debug_output1/` — different examples, not duplicates
