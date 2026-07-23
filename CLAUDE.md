# ScholarScribe

Multi-label tag classification for university thesis metadata: given a thesis's
title + abstract, predict subject tags. Primary/original data source is University
of Jyväskylä (JYU); Tampere University (TAU) is being added as a second source.
Core task framing: **text (title+abstract) → tags**. The abstract is treated as the
dominant training signal (long, information-dense), which shapes some pipeline
choices below — read the note on abstract splitting before "fixing" it.

## ACTIVE: Annihilation round (started 2026-07-23)

**What the user is doing here:** deliberately deleting the entire old graveyard
*outside* `boa_strangling/` — the loose root `.py` scripts, the old top-level code
dirs (`data_preparation/`, `data_validation/`, `images/`, `zdepricated_versions/`,
etc.), and ~174 G of dead weights (`model_embedding/` 125 G, `llama_model/` 8.3 G,
misc) — keeping **only the distilled knowledge, not the bodies**. The live pipeline
(`boa_strangling/`) is self-contained and imports nothing from the graveyard (verified),
so removal is safe. Two surviving artifacts at repo root:
- **`ANNIHILATION_RECORD.md`** — autopsy/inventory of each dead family: what it was,
  distinguishing technique, pipeline position, what it produced. Includes a disk ledger.
- **`PAST_CLEARNESS.md`** — the "soul": before each burial the user answers a **quiz**
  (their reckoning; the death must be a challenge, not a silent delete), recorded with
  resolutions. Holds the lessons / tricks / challenges / mistakes-and-fixes.

**The ritual (per family):** Claude does the reading + writes the autopsy (spares the
user the labor); poses a quiz challenge (the user's last contact with the thing's
shape); the user answers and gives the **final word**; only then is it deleted.
**The user runs the deletions themselves** — provide exact commands, don't execute.
Code → `git rm` (recoverable from history *for now*); weights → plain `rm` (permanent),
provenance recorded first.

**Progress:**
- Cycle 1 **done & cleared**: `llm_generate_tags` family — bury
  `zdepricated_versions/llm_generate_tags_sv2/sv3/sv4.py`, keep root `llm_generate_tags_sv5.py`.
  (User given the `git rm` commands; may not have run them yet.)
- Next candidates: `model_training_GCV_*` family (carries the label-binarization +
  `clean_full_text` bugs) and the 125 G `model_embedding/` old checkpoints (biggest disk win).

**Git bloat (open issue):** `.git` is **37 G** — large weights were committed before the
gitignore fixes; the user's pushes stall on size. Weights are already untracked/gitignored,
so incremental pushes shouldn't grow. The real fix is a **history flush** (orphan-branch
reset, or `git filter-repo --strip-blobs-bigger-than 50M`, then force-push + `git gc
--aggressive --prune=all`). **Do this LAST**, only after annihilation is complete and the
soul is fully captured — flushing history destroys the "recoverable via git" safety net the
code burials rely on. Do not flush mid-round. No background process from this work is running.

## Where to look first

- **`ROADMAP.md`** — phased plan (cleanup, second-university integration,
  pre-processing pitfalls, training, evaluation) with a priority-ordered summary
  table at the bottom. §1.5 is the active work right now (Tampere alignment).
- **`boa_strangling/TAMPERE_INTEGRATION_NOTES.md`** — technical detail for the
  active Tampere work: exact field names on both sides, which JYU scripts are
  reusable-as-pattern vs genuinely stale, and decisions already made vs still open.
- **`boa_strangling/THESIS_LOG.md`** — dated log of training runs: dataset used,
  training config, observed behavior, results. Written to be lifted directly into
  thesis text, so keep entries factual and concise (not a debugging narrative).
  **After any training run that produces real results, append a dated entry here**
  before considering the task done.
- **`PROJECT_ORGANIZATION.md`** — which duplicate/versioned file (`_v2`, `_v3`,
  timestamped) is the current one, for the many loose scripts at repo root.
- **`PROJECT_REVISION.md`** — project-state snapshot / issue list.
- **`boa_strangling/`** — the active, being-cleaned-up codebase (name is a pun:
  strangling the mess that makes the root directory hard to read). `main.py` is
  the step-based pipeline orchestrator for the JYU side. Root-level loose `.py`
  files and `zdepricated_versions/` (note the `z` prefix, sorts last — not a typo
  to "fix") are earlier/superseded iterations; treat them as reference, not as
  things to run directly, unless a doc above says otherwise.

## Where things stand (2026-07-04)

- Tampere harvest is **done**: `boa_strangling/data/tampere_theses.jsonl`, 66,340
  raw records, via `boa_strangling/tampere_thesis_harvest.py` (OAI-PMH).
- **Schema alignment is done** for title/abstract/tags:
  `boa_strangling/scripts/align_tampere_schema.py` → `boa_strangling/data/tampere_aligned.jsonl`
  (53,850 entities, one per abstract). Field-length stats and the full tag
  dictionary + long-tail plot are also done (`boa_strangling/scripts/compute_tampere_field_stats.py`,
  `boa_strangling/scripts/tampere_tag_stats.py`) — see
  `boa_strangling/TAMPERE_INTEGRATION_NOTES.md` §"Schema alignment — done" for
  exact file paths and numbers. **Not done yet**: integrity check pass, and the
  JYU-vs-Tampere tag-space similarity comparison — those are next per `ROADMAP.md` §1.5.
- Faculty data is real (found in `dc:contributor`, not the `faculties` field) but
  the harvester fix + re-harvest to backfill it is an intentionally **deferred**
  follow-up — see `ROADMAP.md` §1.6. Don't attempt this without being asked; it
  means re-hitting Trepo's OAI-PMH endpoint for all 66k records.
- `.gitignore` has been fixed several times this round: typo `__pycahce__` →
  `__pycache__`; removed a blanket `*.json` ignore that was hiding data repo-wide;
  added ignores for generated plots/output dirs; then had to add
  `boa_strangling/outputs/` (7.7GB of model checkpoints) and the large
  Tampere JSONL/JSON data files once they were causing `git add .` to hang for
  minutes — **check `git check-ignore -v <path>` before assuming a large file is
  covered**, a `results/`-style bare pattern only catches what it literally matches.
  Currently staged (`git add .` already run to fix the hang) but **not committed**
  — that's the user's call.
  Deletion/archiving of superseded files (`ROADMAP.md` Phase 0) is intentionally
  on hold — do not delete or `git rm` anything under that plan without being asked.
- Current branch is `remote`, tracking `origin/remote` (not `master`/`main`).

## Decisions already made — don't relitigate without reason

- **Tag field name going forward: `tags`**, not `combined_tags`. The live pipeline
  (`boa_strangling/scripts/generate_variants.py`, what `main.py` actually calls)
  uses `tags`; `combined_tags` is the older name still referenced in some docs and
  `data_validation/` scripts — those are stale, not authoritative.
- **Multi-language abstract splitting is intentional, not a bug.** When a thesis
  has abstracts in more than one language, JYU's `split_abstract_v2.py` produces
  one training record per language (duplicating title/tags/faculty). This is
  deliberate augmentation given the text→tags framing, not something to
  "deduplicate" or "fix" — Tampere alignment should replicate this rather than
  collapsing to a single best abstract.
- **Language codes: 3-letter (`fin`/`eng`/`swe`) is the target format.** For now,
  the primary training set stays fin+eng only, matching the current filter in
  `prep_split_dataset.py`/`generate_variants.py` — Swedish is deferred, not
  excluded forever. Finna scale reference (whole repo, not thesis-filtered): fin
  ≈170k, eng ≈140k, swe+ger ≈14k combined, other languages ≈4k each.
- **Canonical training script: `boa_strangling/scripts/train.py`**, not
  `model_training_GCV_3.py` (the latter has a label-binarization bug — see
  `ROADMAP.md` §0.2/§3.1 for detail).

## Working conventions — be polite with shared/external resources

- **External APIs** (Trepo OAI-PMH, Finna, anything not localhost): don't hammer
  them. Space out requests, don't parallelize aggressively, and treat a "let's
  re-harvest" decision as something to confirm first, not something to just do
  because it's technically possible — see the deferred faculty backfill above
  for a live example of this being paused rather than run immediately.
- **Modifying datasets**: prefer writing new files over silently overwriting
  existing ones, especially once something is treated as a real artifact (an
  aligned dataset, a trained checkpoint). If asked to add a field (e.g. real
  faculty data once the harvester fix lands), do it as a clearly-scoped,
  reviewable change — not a blanket in-place rewrite of the whole dataset without
  a chance to check the result first.
- **Shared compute for training** (this machine may run other people's jobs):
  before starting any training run, check `nvidia-smi` for current GPU
  utilization/memory across all visible GPUs and pick one with enough free
  headroom for the job — never assume exclusive access. Do not launch a job that
  would push a GPU near its memory limit if something else is already running on
  it; that risks crashing someone else's process. If everything looks busy, say
  so and ask rather than picking the "least bad" option silently.
