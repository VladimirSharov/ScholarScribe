# Annihilation Record

> Autopsy log for the old ScholarScribe graveyard (everything outside `boa_strangling/`).
> Each entry captures the *knowledge* a file/family held — what it was, what it tried,
> what technique distinguished it, where it sat in the old pipeline, what it produced —
> so the file itself can be deleted. Written to be liftable into the thesis
> ("approaches tried and discarded, and why").
>
> **Two tracks:**
> - **Code** (tracked in git): buried with `git rm` + commit → recoverable forever from history.
>   The record is for *understanding*, not recovery.
> - **Weights / large artifacts** (untracked, gitignored): buried with plain `rm` → **permanent**.
>   Here the record is the *only* surviving trace of "how it formed". See WEIGHTS LEDGER.
>
> Quiz answers that verified each burial live in `PAST_CLEARNESS.md`.

## Disk snapshot at start (2026-07-23)

Repo = ~180 G working tree + 37 G `.git` history. The conceptual mess is a few MB;
the disk pain is weights.

| Path | Size | In git? | Fate |
|---|---|---|---|
| `model_embedding/` (20+ old checkpoints) | 125 G | untracked | reclaim (record provenance first) |
| `boa_strangling/results/` | 41 G | untracked | mixed — holds the canonical keeper |
| `.git` history | 37 G | — | needs `git filter-repo` (separate, deferred) |
| `llama_model/` (Llama 3.1 8B) | 8.3 G | untracked | reclaim (re-downloadable) |
| `boa_strangling/outputs/` | 1.4 G | untracked | reclaim after keeper picked |
| all versioned root `.py` | few MB | **tracked** | git rm + commit (recoverable) |

**Update 2026-07-25 — the 37 G `.git` is already gone.** It measures **239 M** now. The plan
had been to leave it un-gc'd until the ritual ended (to keep recovery available), but an
**automatic repack** triggered by the annihilation commits dropped the unreachable pre-flush
objects, so the reclaim happened as a side effect rather than by the planned last rite.
**Recoverability is unharmed — verified by reading both anchors:** `a3c516c` (Cycle 2/3) and
`origin/master` `001e6d0` (Cycle 1) are still readable. The remaining 239 M *is* the preserved
history. The "last rite" now means only dropping `master`/`master-remote`'s old history — a few
hundred MB, cosmetic.

---

## CODE TRACK

### Family: `llm_generate_tags_sv*` — LLM (Llama 3.1 8B) tag generation

**Status: analyzed, awaiting quiz + burial. Keeper = `sv5`.**

The LLM branch of the project's two-approach design (fine-tuned XLM-R vs. prompted LLM).
All four talk to a local llamafile server (`http://localhost:8080/v1`) through the
`openai` client, feed it title+abstract, and ask for JSON tags. They are **not a clean
linear version chain** — they branch on two axes: *tag schema* (split `subject`/`additional`
vs. unified `tags`) and *scale/robustness* (single-example tester → batch → production).

| File | Real date | Role | Tag schema | Scale |
|---|---|---|---|---|
| `sv3` | **Dec 2024** (oldest!) | dataset runner w/ `limit`, confidence-sorted | unified `tags` | batch, timestamped output |
| `sv2` | Mar 2025 | single-example smoke test | split subject/additional | one entry |
| `sv4` | Mar 2025 | **multi-model benchmark harness** | split subject/additional | threaded, built-in P/R/F1 + comparative `.md` report |
| `sv5` | **Apr 2025 (keeper)** | production batch generator | unified `tags` | resumable, signal-safe, atomic writes, FI/EN prompts |

**What each tried to achieve / distinguishing technique:**

- **`sv2`** — minimal proof it works end-to-end: load dataset, process the *first* entry,
  regex-extract JSON (` ```json ` fence or first `{...}`), print. Truncates title+abstract
  to 100 chars. No saving. The "does the llamafile answer at all" smoke test.
- **`sv3`** (older than sv2 despite the number) — first *dataset* runner: `process_dataset(limit)`,
  configurable default path, saves timestamped JSON to `model_outputs/`. Switched the prompt to
  a **single unified `tags` list sorted by confidence** (dropped the subject/additional split).
  Top-of-file comment block is a genuine planning artifact (GPU `-ngl 999`, keep thesis-id + abstract
  language from prep step, Levenshtein/cosine metric ideas) — worth reading before deletion.
- **`sv4`** — the odd one out: a **`TagEvaluator` benchmark harness**, not a generator. Runs *several*
  model endpoints in parallel (`ThreadPoolExecutor`), computes precision/recall/F1 per entry itself,
  and emits a comparative markdown report ranking models. Reverted to the split subject/additional
  schema. Its metrics were later externalized into the `tag_evaluation_v*` family, which is why sv5
  could drop them.
- **`sv5`** (keeper) — production robustness: **resumable** (loads existing results, skips processed
  `identifier`s), **signal-safe** (SIGINT/SIGTERM → save-then-exit), **atomic writes** (temp file +
  `os.replace`), **per-language prompts** (Finnish vs. English), running metadata (timings, success/fail).
  Unified `tags` schema. This is the one to keep.

**Pipeline position:** parallel LLM arm. Input = `data_split_v4/full_dataset_v3_test.json`
(the old JYU test split). Output = `generated_tags_results*.json` at repo root
(`_separate_tags` / `_union_tags` / final merged). Those feed the `data_validation/tag_evaluation_v*`
scripts, which produce `tag_evaluation_*/` and `evaluation_results_*/`. The whole LLM arm is a
self-contained subsystem — **nothing in live `boa_strangling/` imports any of it.**

**Distinguishing techniques worth preserving (thesis material):**
- resumable + signal-safe + atomic-write batch pattern (sv5) — the "long job, don't lose work" lesson
- multi-model parallel benchmark-with-report pattern (sv4)
- confidence-sorted unified-tag prompt (sv3)
- per-language (FI/EN) prompt localization (sv5)

**Latent blunders carried by the whole family (see quiz):**
1. Prompt injects `tag_count = len(original tags)` — tells the model *how many* tags to produce,
   leaking ground-truth cardinality a real deployment wouldn't have.
2. All take `abstract[0]` / `title[0]` — the **first** abstract only — contradicting the later
   "2 abstracts is augmentation, take both" decision baked into the boa_strangling pipeline.
3. Truncation length drifted with no rationale: 100 → 1000 → 1500 chars across versions.

**Buried:** _pending quiz._ **Kept:** `llm_generate_tags_sv5.py` (for now; the LLM arm itself may be
retired later — separate decision).

---

### Family: `model_training_GCV_*` — XLM-R fine-tuning trainer (the forge)

**Status: analyzed, awaiting quiz + burial. Keeper = none (superseded whole by
`boa_strangling/scripts/train.py`).**

The fine-tuning arm of the two-approach design (the twin of the `llm_generate_tags_*`
LLM arm). All three train `xlm-roberta-base` for multi-label thesis-tag classification
via HF `Trainer` + wandb, wrapped in the same scaffold (`Config` → `ThesisModelTrainer`
→ `ThesisMetrics`). **"GCV"** = the old grid/config-driven trainer name. This family is
literally **the forge that produced the 125 G `model_embedding/` graveyard** — every run
writes `model_embedding/model_output_{mn,1n}_<timestamp>/` checkpoints + a
`tag_mapping_*.pickle`. Autopsy this family and the Cycle-3 weights lose their mystery.

**Recovery anchor (IMPORTANT — differs from Cycle 1):** these three live **only** in the
cleaning-iteration commit `a3c516c`, **not** on `origin/master` (that backup is an older
snapshot with no `boa_strangling/` and no root trainer). Recover with
`git show a3c516c:model_training_GCV_3.py`, **not** from `origin/master`.

| File | Real date | Role | Loss / labels | Text field | Data source |
|---|---|---|---|---|---|
| `GCV_2` | 2025-02-27 | base multi-label trainer | BCE, **correct multi-hot** (1.0) | `title[0]` + first abstract | `data_split_v4/full_dataset_v3_*` |
| `GCV_singleSoft` | 2025-02-28 | soft-label CE experiment | **CE, soft targets** (1/k distribution) | `title[0]` + first abstract | `data_split_v4/full_dataset_v3_*` |
| `GCV_3` | **2025-05-21 (latest)** | boa-migrated trainer | BCE, **labels REGRESSED to index lists** | `clean_full_text` | `boa_strangling/data/data_split_fi_eng_min5_abs40/` |

**What each tried / distinguishing technique:**

- **`GCV_2`** — the baseline that got the fundamentals *right*: `AutoModelForSequenceClassification`
  with `problem_type="multi_label_classification"` (→ `BCEWithLogitsLoss`), labels built as a
  proper **multi-hot** matrix (`np.zeros((n, |tags|))`, set 1.0). Computes tag vocab on the fly
  from the train split (`tag2id`), saves it as pickle. `ThesisMetrics`: macro P/R/F1 at a fixed
  0.5 threshold + a hand-rolled normalized TN/FP/FN/TP confusion matrix. Fixed threshold and
  on-the-fly vocab are its two weaknesses (both fixed downstream), but the label encoding is sound.
- **`GCV_singleSoft`** — the experimental fork (the interesting one). **Pastes the entire
  `XLMRobertaForSequenceClassification` source** into the file (via `from ...modeling_xlm_roberta
  import *` + a redefined class) purely to sit inside the loss branch and inspect it (commented
  `print(labels)` probes). Sets `problem_type="single_label_classification"` (→ `CrossEntropyLoss`)
  but feeds it **soft targets**: `multi_hot[i][label] = 1.0/len(label_list)` — a probability
  *distribution* over the true tags summing to 1, instead of a hard multi-hot. The hypothesis:
  treat multi-label as "spread unit probability mass across the correct tags" and let CE match the
  distribution — a soft-label knowledge-distillation-style trick against the sparsity/imbalance that
  plain BCE struggled with. Metrics disabled; lr dropped 2e-5 → 1e-5. Distinctive technique worth
  keeping: **soft label distribution (1/k) + CE** as an alternative to multi-hot + BCE.
- **`GCV_3`** — the last iteration; the only one migrated onto the **boa_strangling** data layout.
  Two genuine *improvements*: (a) loads `tag2id` from an **external `tag_vocab.json`** instead of
  recomputing per-run (kills train/val/test vocab drift), and (b) `lr_scheduler_type="constant"`.
  But it carries the **two documented regressions** (see `ROADMAP.md` §0.2/§3.1):
  1. **Label-binarization bug** — labels built as **ragged index lists** (`valid_tags =
     [tag2id[t] ...]`, `tokenized_inputs["labels"] = labels`), *not* a multi-hot matrix. With
     `multi_label_classification`/BCE this is wrong shape/semantics — a regression from `GCV_2`,
     which had it right. Worse, the empty-example fallback `valid_tags = [0]` stamps on **real tag
     id 0**, silently poisoning every tag-less row instead of leaving an all-zero target.
  2. **`clean_full_text` field** — tokenizes `examples["clean_full_text"]`, abandoning the explicit
     `title[0] + first-abstract` construction for a precomputed text column flagged as stale/bug-prone.

**Chronology irony (thesis-worthy):** version number tracks neither time nor quality here either —
`GCV_3` is newest yet *regressed* the one thing `GCV_2` got right (multi-hot labels). Progress on
plumbing (external vocab, boa data, constant LR) masked a correctness regression in the core target
encoding. This is the concrete origin of the "GCV has a label-binarization bug" warning in the docs.

**Pipeline position:** the fine-tuning training stage. Input = a `*_train/val/test.json` split
(old `data_split_v4/` for GCV_2/singleSoft; `boa_strangling/data/data_split_fi_eng_min5_abs40/`
for GCV_3). Output = `model_embedding/model_output_*/` checkpoints + `tag_mapping_*.pickle`
(→ the Cycle-3 weight graveyard) + wandb runs under project `thesis-tagger`. Consumed downstream
by the old evaluation/prediction scripts. **Nothing in live `boa_strangling/` imports any of them.**

**Where the capability went — `boa_strangling/scripts/train.py` (keeper) does it right:**
- labels via `MultiLabelBinarizer` → true multi-hot, fed as `torch.FloatTensor` (fixes bug 1);
  the tag field is `tags` (not `combined_tags`/`clean_full_text`) (fixes bug 2);
- per-class **`pos_weight` BCE** (`sqrt(log1p(neg/pos))`) instead of unweighted BCE — for rare tags;
- **per-frequency-bucket adaptive thresholds** instead of a fixed 0.5;
- stratified recall by frequency bucket + subset accuracy. GCV's fixed-0.5 + on-the-fly-vocab +
  regressed labels are all superseded. (The `singleSoft` soft-CE idea was *not* carried forward —
  it's the one genuinely novel thread that dies with this family; preserved here as a record.)

**Distinguishing techniques worth preserving (thesis material):**
- soft label distribution (1/k) + CrossEntropy as an alt to multi-hot + BCE (`singleSoft`)
- pasting-the-model-class to instrument the loss branch — a debugging move, not production
- externalizing `tag_vocab.json` so splits share one label space (`GCV_3`'s one good idea)
- normalized TN/FP/FN/TP confusion at eval time (`ThesisMetrics`)

**Latent blunders carried by the family (see quiz):**
1. **Label-binarization regression** in `GCV_3` (index lists, not multi-hot; `[0]` poison fallback).
2. **`title[0]` + `abstract[0]`** in GCV_2/singleSoft — same first-abstract-only betrayal as the LLM
   family; the "two abstracts = two records" augmentation decision throws the rest away.
3. **Fixed 0.5 threshold** for a long-tailed multi-label problem — guarantees rare tags never fire.
4. **On-the-fly vocab** (GCV_2/singleSoft) → the label space can drift between train and eval runs.

**Buried:** _pending quiz._ **Kept:** none — superseded whole by `boa_strangling/scripts/train.py`.

---

### Sub-family: alternate encoders (DeBERTa / SPECTER2) — 4 scripts inside `model_embedding/`

**Status: read, recorded, deleted.** Not a full ritual cycle — these 4 tracked `.py` files
lived *inside* `model_embedding/` and were caught when the 125 G weight dir was `rm`'d. User
did not recall using them; recorded here as ideas, then buried. (This is why
`git check-ignore model_embedding/` returned nothing earlier — tracked files under an
ignored dir; gitignore never applies to already-tracked files.)

| File | What it really is | Used on thesis data? |
|---|---|---|
| `deberta_model.py` | Tutorial template: `microsoft/deberta-v3-small` on the HF `knowledgator/events_classification_biotech` set (biotech fields, not theses) | No |
| `deberta_test.py` | Named "deberta" but loads **`xlm-roberta-base`** — early multi-label prototype on `data_split/full_dataset_*` | Yes (but it's XLM-R) |
| `specter2_model_Example.py` | allenai SPECTER2 README example, verbatim (sample "BERT"/"Attention" papers) | No |
| `specter2_model_Small.py` | Real use: SPECTER2 adapter embeds thesis `title+faculty+tags` → saves `embeddings/embedding_F_R.npy` | Yes |

**Two ideas worth preserving (the reason to record before deleting):**
1. **The multilingual pivot, captured in one file.** `deberta_test.py` is named after DeBERTa
   but the code has *already switched to XLM-R* — the fossil of the moment English
   `microsoft/deberta-v3-small` was dropped for multilingual `xlm-roberta-base`. DeBERTa
   didn't solve the fin+eng problem; it **lost to it**, and XLM-R won for exactly that reason.
2. **The embedding-extraction approach (what named `model_embedding/`).** `specter2_model_Small.py`
   uses `allenai/specter2` via the `adapters` lib to turn a thesis into a single CLS embedding
   vector saved as `.npy` — a *features-then-classify / similarity* path, distinct from the
   end-to-end fine-tuning that GCV/`train.py` took. Note it fed `subject_tags`+`additional_tags`
   *into* the embedding input — fine for clustering, but would be **label leakage** if ever used
   as features for tag prediction. The embedding path was not carried forward.

**Recovery anchor:** commit `a3c516c` (same as GCV — `origin/master` predates them).
**Loose ends (not weights, minor):** `embeddings/*.npy` and `prepared_datasets/faculty_related.json`
are the only on-disk products; check/reclaim separately if present.

**Buried:** together with the GCV commit (or a follow-up). **Kept:** none.

---

### Family: `data_preparation/` — the dataset-building arm (7 scripts)

**Status: analyzed, awaiting quiz + burial. Keeper = none (superseded by
`boa_strangling/scripts/prep_split_dataset.py` + `generate_variants.py`).**

Upstream of *everything* else in the graveyard: this is the family that turned JYX's raw
DSpace dump into the `full_dataset*.json` / `data_split_v*` files that the GCV trainers
(Cycle 2) and the LLM arm (Cycle 1) both consumed. Not a version chain — a **pipeline**,
plus one abandoned side branch.

| File | Real date | Role | Consumes | Produces |
|---|---|---|---|---|
| `v3_data_preparation.py` | Dec 2024 | flatten raw `dc.*` records → simplified entities | `data_collection/output.json` | `data_preparation/dataset/full_dataset_v3.json` |
| `stratify_split_v2.py` | Dec 2024 | faculty-stratified 60/20/20 split | `full_dataset_v3.json` | `data_split_v3/full_dataset_v3_{train,val,test}.json` |
| `split_abstract_v2.py` | Dec 2024 | **one record per abstract** (the augmentation) | `data_split_v3/*` | `data_split_v4/*` |
| `data_sew.py` | **Mar 2025 (newest)** | re-concatenate the 3 v4 splits into one corpus | `data_split_v4/*` | `data_split_v4/full_dataset.json` |
| `field_remover.py` | Nov 2024 | **deletes** `thesis_title` + `abstract` | `tempFieldMod/full_dataset_e.json` | `…_er.json` |
| `field_processor.py` | Nov 2024 | one-hot / binary-vector feature engineering | `…_er.json` | `…_erp.json` |
| `scheme_data.py` | Apr 2025 | hand-written schema doc | — | `db_schema.json` |

**The main chain (1→4) and what each contributed:**

- **`v3_data_preparation.py`** — the parser. JYX exports each thesis as a *list of
  `{key, value, language}` items* (`dc.title`, `dc.subject.yso`, `dc.contributor.tiedekunta`…);
  this flattens that into a flat entity via a `field_name_mapping` dict — **the origin of the
  project's field names** (`subject_tags`, `additional_tags`, `thesis_title`, `abstract`,
  `faculty`, `date_issued`, `identifier`). Also holds `standardize_language_code()`, the
  2-letter→3-letter ISO 639-2/B map (`fi→fin`, `en→eng`, …) — **the origin of the "3-letter is
  the target format" decision** still in force. Entities are kept only if *all* requested fields
  are present (`all(field in entry_copy …)`), which is where the corpus silently shrinks.
- **`stratify_split_v2.py`** — 60/20/20, **stratified by faculty** (`faculty[0]`), with
  small-group fallbacks: faculty with 2 members → train/test only, with 1 → train only;
  faculty-less entities all appended to train.
- **`split_abstract_v2.py`** — the augmentation step, and the one whose *idea* survives:
  a thesis with N abstracts becomes N records, each keeping the same title/tags/faculty,
  each getting a fresh `identifier` (`<orig>_abstract_<idx>_<uuid8>`) plus an
  `original_identifier` back-pointer. That back-pointer is what still makes it possible to
  group siblings — it is the reason the finding below could be measured at all.
- **`data_sew.py`** — the **bridge to `boa_strangling/`**: sews the three `data_split_v4`
  splits back into a single `full_dataset.json`, i.e. deliberately *undoes* the old split so
  the new pipeline could re-split the corpus on its own terms
  (`prep_split_dataset.py` → `data_split_fi_eng_min5_abs40/`). The live
  `boa_strangling/data/full_dataset.json` is the descendant of this file.

**The side branch (5→6) — the abandoned classical-features path:** `field_remover.py`
**deletes `thesis_title` and `abstract`**, then `field_processor.py` vectorizes what's left:
tags → multi-hot (`tags_vector`, with its own `class2id`), faculty → one-hot (behind a
`FACULTY_DICT` normalizing the 9 JYU faculties FI→EN), language → one-hot, year → int.
This is a **metadata-features → classifier** design: throw the text away, predict from
structured fields. It is the exact opposite of the framing the project settled on
(text → tags), and it died — nothing consumes `tempFieldMod/`. (It's also unrunnable today:
`OneHotEncoder(sparse=False)` was renamed `sparse_output` in sklearn ≥1.2, `TfidfVectorizer`
is imported but never used, and `pd.DataFrame(list_of_lists, columns=['faculty'])` breaks on
any multi-faculty row.)

**Pipeline position:** the head of the old pipeline. `data_collection/output.json` → *this
family* → `data_split_v4/` → consumed by **both** arms (GCV trainers, `llm_generate_tags_sv*`)
→ and, via `data_sew.py`, → `boa_strangling/data/full_dataset.json`, still live today.

**Distinguishing techniques worth preserving (thesis material):**
- **split-after-stratify ordering** — the augmentation runs *after* the train/val/test split,
  so both language siblings of a thesis land in the same split. Getting this backwards would
  put near-duplicate records (identical title, identical tags) on both sides of the split and
  inflate every score. This family got it right; say so in the thesis.
- `original_identifier` back-pointer surviving augmentation (makes sibling grouping auditable).
- the `dc.*` → project-field mapping and the ISO 639-2/B normalization — both still in force.
- small-group split fallbacks (n=2 → no val, n=1 → train) as an honest, if crude, answer to
  stratifying a long-tailed grouping variable.

**Latent blunders carried by the family (see quiz):**
1. **The `abstract_language` collapse — verified, and the most consequential.** Two faults
   compound: (a) `v3_data_preparation.py` assigns `simplified_entry['abstract_language']`
   **inside the per-item loop**, so with several abstracts each assignment overwrites the
   previous — only the **last** abstract's language survives — and it is a **scalar string**;
   (b) `split_abstract_v2.py` *does* contain the code to give each sibling its own language,
   but guards it with `isinstance(entity["abstract_language"], list)` — always False against a
   string, so it is **dead code**. Measured in `data_split_v4/full_dataset_v3_test.json`:
   **383 of 383** multi-abstract theses have the *identical* `abstract_language` on every
   sibling (380 of them stamped `eng`), while **382 of 383** sibling pairs are genuinely
   different texts. So roughly half of all augmented records carried a **wrong language label**.
   This is the concrete origin of the later language-detection work (`detect_lang_utils.py`,
   `language_analysis.py`) and of the Cycle-1 recollection that "the abstract's language wasn't
   in the original parse". **Already fixed downstream**: live
   `boa_strangling/data/full_dataset.json` has *distinct* sibling languages in 1966 of 1970
   cases. Worth knowing anyway, because `prep_split_dataset.py` still filters on exactly this
   field (`filter_language`) — a silent regression here would silently halve the corpus.
2. **Non-reproducible split.** `stratify_split_v2.py` calls `random.shuffle(data)` **unseeded**
   before splitting, while passing `random_state=42` to `train_test_split`. The seed is
   theatre: every run yields a different split. A plausible contributor to the `data_split_v*`
   sprawl — a split you cannot regenerate can only be versioned.
3. **Stratified by the wrong variable.** Faculty is a proxy; for a text→tags task the quantity
   with the punishing distribution is the **tag long tail**, which is not stratified at all —
   rare tags can land entirely in one split. Compounding it: faculty-less entities are *all*
   appended to train, and single-member faculties always go to train, so train absorbs every
   irregular case.
4. **Reporting theatre.** `language_counter` is constructed and written into
   `processing_metadata.json` but **never incremented** (`language_distribution` is always
   `{}`); `missing_lang_entries` is never used; `deprecated_entries` is incremented inside the
   per-output-file loop, so it counts once per configured dataset rather than once per entry.
   A metadata log that cannot be trusted is worse than none.

**Where the capability went:** `boa_strangling/scripts/prep_split_dataset.py` (seeded
`SEED = 42` throughout, explicit `MIN_TAG_FREQ`/`MIN_ABSTRACT_WORDS`/`VALID_LANGUAGES`
filters, a written `split_report.json`) + `generate_variants.py`. The augmentation *concept*
is carried forward explicitly — `boa_strangling/scripts/align_tampere_schema.py` cites
"`split_abstract_v2.py` augmentation strategy — intentional, not deduped away" in its docstring.

**Dangling reference to fix at burial:** `boa_strangling/main.py:107` and `:116` print
instructions to run `data_preparation/v3_data_preparation.py`. Narrative only (no import),
but it will point at a non-existent file after the burial — update those two lines in the
same commit.

**Recovery anchor:** on `origin/master` **and** `a3c516c` (this family predates the cleanup,
so both work): `git show origin/master:data_preparation/split_abstract_v2.py`.

**Buried:** 2026-07-25, all 7 files. **Kept:** none.

---

## WEIGHTS LEDGER (untracked — record is the only surviving trace)

### `model_embedding/` — 125 G — **reclaimed 2026-07-25**
The checkpoint + `tag_mapping.pickle` output of the `model_training_GCV_*` runs
(`model_output_mn_*` = multi-label, `model_output_1n_*` = single-soft/"flattened"), plus a
`tokenized_datasets/` cache. No idea lives here that isn't in the GCV autopsy above;
regenerable from `boa_strangling/scripts/train.py` + data. Permanent `rm`. (Also held 4
tracked alt-encoder scripts — see sub-family entry.)

### `llama_model/` — 8.3 G — **KEPT (2026-07-25, user decision — still works)**
- `Meta-Llama-3.1-8B-Instruct.Q8_0.llamafile` (8.7 G, Oct 2024) — the local LLM the
  `llm_generate_tags_sv*` family (Cycle 1) queried at `http://localhost:8080/v1`. A Mozilla
  **llamafile** (self-contained executable + Q8_0 GGUF weights), **downloaded, not trained →
  re-downloadable**. **It worked** and produced the LLM arm's real output — the generated-tag
  files at repo root (`generated_tags_results*.json`) and `model_outputs/`
  (`tag_generation_results_2024120*.json`, `llama-3-1-10b_tag_generation_results_*`). The later
  `main.log` failure (`wrong number of tensors; expected 292, got 291 → failed to load model`)
  is a **llama.cpp / GGUF version incompatibility on a *reload*** — a newer runtime couldn't
  load this old binary — **not** "never ran". Deleting the binary loses nothing: it's
  re-downloadable and its actual product (the tag files, kept outside this dir) is preserved.
  **User chose to KEEP it — it still works normally** (the `main.log` error was a one-off reload
  under a newer runtime, not a permanent break). *(Never launched during this cleanup —
  read-only throughout; running an 8 B LLM consumes shared GPU and is not done without checking
  `nvidia-smi` first.)*
- `llama_model/split/` (120 M) — **NOT weights**: a stray **min-freq-5 dataset split**
  (81,738 records, 8,728 labels, avg 4.64 tags/row; `train/val/test.jsonl` +
  `label_encoder.json` + `split_report.json`), misplaced inside the model dir on Jul 16.
  Its `split_report.json` is **parameter-identical to root `split/`** (same 81,738 records,
  8,728 labels, freq bands, 4.64 avg) — the *same* min-freq-5 split, just a leaner export
  (77 M vs root's 180 M train; same records, fewer fields). Root `split/` is the fuller kept
  copy → this one is **redundant, safe to delete** with the weights.
