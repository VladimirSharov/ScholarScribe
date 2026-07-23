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
