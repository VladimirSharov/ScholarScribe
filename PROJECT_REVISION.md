# ScholarScribe — Project Revision

Generated: 2026-06-29

---

## What This Project Is

An ML pipeline for **automatic subject-tag generation for Finnish university theses**, sourced from the JYU (University of Jyväskylä) open repository. The core idea: given a thesis title and abstract (multilingual — Finnish/English), predict the correct subject tags.

Two approaches are being developed in parallel:
1. **Fine-tuned XLM-RoBERTa** — multi-label classifier trained on thesis metadata
2. **LLM-based (Llama 3.1 8B)** — prompt-based tag generation via a local llamafile server

---

## Pipeline (Intended Order)

```
data_collection/api_data_collector.py
        ↓ output.json (13,499 raw entries)
data_preparation/v3_data_preparation.py
        ↓ data_preparation/dataset/full_dataset_v3.json
data_preparation/stratify_split_v2.py
        ↓ data_split_v3/ (train/val/test)
data_preparation/split_abstract_v2.py
        ↓ data_split_v4/ (split multi-abstract entries)
language_analysis.py
        ↓ data_split_v5/ (adds detected abstract_language)
[boa_strangling scripts] — tag vocab, splits, stats
        ↓ boa_strangling/data/data_split_fi_eng_min5_abs40/
model_training_GCV_2.py / boa_strangling/scripts/train.py
        ↓ model_embedding/model_output_*/
data_validation/tag_evaluation_v*.py
        ↓ evaluation_results_*/
```

---

## Directory Map

| Directory / File | Role | Status |
|---|---|---|
| `data_collection/` | API scraper from JYU repository | Done; `output.json` is the raw source |
| `data_preparation/` | Field extraction, splitting, abstract separation | Done; multiple versions |
| `data_validation/` | Integrity checks, tag evaluation, language analysis | Active, many versioned files |
| `data_split/`, `data_split_v4/`, `data_split_v6/` | Train/val/test JSON splits | Multiple coexisting versions |
| `boa_strangling/` | Cleaner rewrite of data prep + training pipeline | **Active / newest approach** |
| `model_embedding/` | Training checkpoints (20+ runs stored) | Accumulating outputs |
| `model_training_GCV_2.py`, `_GCV_3.py`, `_GCV_singleSoft.py` | XLM-R training scripts | Multiple versions, unclear which is current |
| `llm_generate_tags_sv2.py` – `sv5.py` | Llama-based tag generation | sv5 appears newest |
| `data_validation/tag_evaluation_v*.py` | Evaluation metrics (exact match, EMD, F1) | v4 is newest |
| `run_faculty_analysis.py` + `faculty_data_processor.py` + `faculty_visualizer.py` | Faculty distribution analysis (side task) | Untracked, recently added |
| `language_analysis.py` + `detect_lang_utils.py` | Detects abstract language, adds field | Untracked, recently added |
| `comparation_language_prompts.py` | Compares LLM prompt strategies across languages | Untracked |
| `depricated_versions/` | Old files moved here | Effectively a trash bin |
| `wandb/` | W&B training run logs (40+ runs from 2024–2025) | Large, not gitignored |
| `project_utils/` | `finalizer.py` (auto-commit helper), `recordProjectStructure.py` | Utility scripts |

---

## Current State

### What Works / Is Done
- Raw data collected: 13,499 entries from JYU API
- Dataset cleaned and split (train/val/test) in multiple formats
- XLM-RoBERTa trained and checkpointed across many runs (oldest: Nov 2024, newest: May 2025)
- `boa_strangling/` contains a cleaner dataset pipeline with stats and label coverage verified (0% unseen tags in val/test)
- Evaluation results exist for both LLM and fine-tuned model runs
- Faculty distribution analysis recently implemented as a visualization side task

### What Is In Progress
- `boa_strangling/scripts/train.py` — appears to be the current active training script
- `model_training_GCV_3.py` — untracked, probably a newer iteration of training
- `language_analysis.py` — adds `abstract_language` detection; outputs to `data_split_v5/` (not yet in git)
- Comparison of LLM vs fine-tuned model performance

---

## Issues

### 1. Navigation / Organization
- **Too many dataset versions**: `data_split/`, `data_split_v4/`, `data_split_v6/`, `boa_strangling/data/data_split_fi_eng_min5_abs40/`, `boa_strangling/data/data_split_min5/` — unclear which is canonical for training
- **Too many training scripts**: `model_training_GCV.py`, `_GCV_2.py`, `_GCV_3.py`, `_GCV_singleSoft.py`, `boa_strangling/scripts/train.py` — unclear which is the one to run
- **Too many LLM scripts**: `llm_generate_tags_sv2` through `sv5` — no indication of which is current
- **Too many tag evaluation scripts**: `tag_evaluation.py`, `_v2`, `_v3`, `_v35`, `_v4`, `_lang` — supersession unclear
- **Root level polluted**: Many loose `.py` files, `.json` outputs, `.png` charts, and `.log` files at root instead of organized in folders

### 2. Untracked Files (Large, Risky)
- 28 untracked files/directories including key source files: `faculty_data_processor.py`, `language_analysis.py`, `model_training_GCV_3.py`, `rebuild_dataset_from_log.py`, `comparation_language_prompts.py`
- If the local copy is lost, this work is gone — it is not in git

### 3. Hardcoded Paths Throughout
- Every script hardcodes its own input/output paths (not configurable via args)
- A change in one script's output path requires manually updating the next script in the chain
- Several scripts reference paths that may not exist (`data_split_v3/`, `prepared_datasets/`, `data_preparation/tempFieldMod/`, `data_split_v5/`)

### 4. `main.py` Is Abandoned
- Was intended as the central orchestrator but contains only aspirational notes/comments
- The project has no actual central entry point

### 5. `README.md` Is Outdated
- Describes the initial data collection stage only
- References a placeholder `git clone https://a.git`
- Does not mention XLM-R, Llama, boa_strangling, or the current pipeline

### 6. `requirements.txt` Is Not Portable
- Contains `file:///home/conda/feedstock_root/...` paths — machine-specific, cannot be used to reproduce the environment elsewhere
- Use `project_utils/requirements-frozen.txt` only as a reference

### 7. `wandb/` and `model_embedding/` Are Large
- 40+ W&B run logs
- 20+ model checkpoint directories
- These should either be gitignored or clearly noted as not for version control

### 8. `finalizer.py` Has a Bug
- `changelog.txt` entries show typos like `"roject_journal.txt"`, `"aculty_encoding.py"`, `"ata_validation/..."` — the script strips the first character of each filename when writing the log

### 9. Folder Name Typo
- `depricated_versions/` should be `deprecated_versions`

### 10. `.gitignore` Has a Typo
- `__pycahce__/` is written instead of `__pycache__/` — Python cache directories are NOT actually gitignored

### 11. `data_split_v5/` Does Not Exist
- `language_analysis.py` writes to `data_split_v5/` (creates it on first run — safe)
- `rebuild_dataset_from_log.py` reads from `data_split_v5/` and will crash with `FileNotFoundError` if `language_analysis.py` has not been run first

### 12. Schema Mismatch in `model_training_GCV_3.py`
- Default data paths point to `boa_strangling/data/data_split_fi_eng_min5_abs40/`
- But the preprocessing function references a field called `clean_full_text` — this field does not exist in that dataset (it has `title`, `abstract`, `language`, `combined_tags`)
- Script would crash at tokenization

### 13. Missing Packages Not Listed in `requirements.txt`
These are hard-imported in current untracked scripts and would cause immediate `ImportError`:

| Package | Used In |
|---|---|
| `langdetect` | `faculty_data_processor.py` (hard import), `tag_evaluation_lang.py` |
| `seaborn` | `faculty_visualizer.py`, multiple `images/` scripts |
| `openai` | `llm_generate_tags_sv2.py` – `sv5.py` |
| `Levenshtein` | `data_validation/tag_evaluation_v4.py` |
| `jellyfish` | `data_validation/tag_evaluation_v4.py` |
| `sentence_transformers` | `data_validation/tag_evaluation_v4.py` |
| `matplotlib` | Many scripts (likely pulled in transitively but not listed explicitly) |

### 14. `boa_strangling/scripts/compute_stats.py` Import Path Issue
- Uses `from utils.io import load_json` — only works if run from inside `boa_strangling/scripts/`
- Running from project root as `python boa_strangling/scripts/compute_stats.py` will fail with `ModuleNotFoundError`

---

## What Needs Attention (Priority Order)

1. **Commit untracked files** — `language_analysis.py`, `detect_lang_utils.py`, `model_training_GCV_3.py`, `faculty_data_processor.py`, `faculty_visualizer.py`, `run_faculty_analysis.py`, `rebuild_dataset_from_log.py`, `comparation_language_prompts.py`, and the entire `boa_strangling/` directory
2. **Decide on canonical dataset** — pick one split directory as the source of truth for all training going forward (likely `boa_strangling/data/data_split_fi_eng_min5_abs40/`)
3. **Decide on canonical training script** — likely `boa_strangling/scripts/train.py`; archive the others
4. **Update README.md** — describe what the project actually does now, how to run it, which files matter
5. **Add `.gitignore` entries** — `model_embedding/`, `wandb/`, `__pycache__/`, `*.log`, loose `*.png` at root
6. **Fix `finalizer.py`** — the bug strips the first character of each changed filename in the changelog

---

## Files That Are Likely Safe to Archive

- `llm_generate_tags_sv2.py`, `sv3.py`, `sv4.py` (keep `sv5.py` as current)
- `model_training_GCV.py`, `model_training_GCV_singleSoft.py` (if `boa_strangling/scripts/train.py` is current)
- `testing_output_2.py`, `testing_output_3.py` (hardcoded old checkpoint paths, for reference only)
- `llama_install.py`, `llama-test.py` (hermes model, no longer used per journal)
- `trash.py`
- `data_validation/tag_evaluation.py`, `_v2`, `_v3`, `_v35` (keep `_v4`)
- `comprehensive_debug_output/`, `comprehensive_debug_output1/`
