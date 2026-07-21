# ScholarScribe

Multi-label tag classification for university thesis metadata: given a
thesis's title + abstract, predict subject tags. Data sources are the
University of Jyväskylä (JYU), Tampere University (TAU), and Helsinki.
Core task framing: **text (title+abstract) → tags**. The abstract is treated
as the dominant training signal (long, information-dense), which shapes some
pipeline choices below — read the note on abstract splitting before "fixing"
it.

This repo is the second iteration of the project — it used to live as a
subdirectory (`boa_strangling/`) inside a larger, much messier repo (loose
duplicated scripts, several abandoned experiment iterations); that outer
layer has been dropped and this directory promoted to be the whole project.
See `README.md` for what changed and why.

## Where to look first

- **`README.md`** — project overview: what changed in this iteration
  (second university added, tag-frequency filtering, per-bucket adaptive
  thresholding, training-budget findings, the demo).
- **`TAMPERE_INTEGRATION_NOTES.md`** — technical detail for the
  multi-university schema alignment work: exact field names per source, which
  JYU scripts are reusable-as-pattern vs stale, decisions already made vs
  still open.
- **`THESIS_LOG.md`** — dated log of training runs: dataset used, training
  config, observed behavior, results. Written to be lifted directly into
  thesis text, so keep entries factual and concise (not a debugging
  narrative). **After any training run that produces real results, append a
  dated entry here** before considering the task done.
- **`main.py`** — the step-based pipeline orchestrator for the JYU side.

## Where things stand (2026-07-21)

- Multi-university merged dataset (JYU + Tampere + Helsinki) is the active
  training set: `data/data_multiuni_2026-07-16/`, 8,728 tags after a min-freq-5
  filter (down from 22,027 raw). See `THESIS_LOG.md` 2026-07-16 entry for the
  filtering rationale and what it did/didn't fix.
- Best trained checkpoint to date is `checkpoint-145404` (final checkpoint of
  the 2026-07-19 36-epoch run) — see `THESIS_LOG.md` 2026-07-19/2026-07-21
  entries for why the training-time "best" pick (`checkpoint-117000`) is
  actually behind once thresholds are compared fairly.
- Test-threshold-leakage in `scripts/train.py`'s final evaluation is fixed:
  thresholds are fit on val, frozen, then applied to test. `scripts/predict.py`
  has a `--full_test_eval` mode to re-score an already-trained checkpoint
  honestly without retraining, and a `--threshold_model_dir` flag to apply one
  checkpoint's calibration to another's weights (for fair cross-checkpoint
  comparison).
- A standalone demo (`demo/app.py`) exists; weights are transferred
  out-of-band (see `demo/README.md`), not committed to git (~500MB+ per
  checkpoint).
- Faculty data for Tampere is real (found in `dc:contributor`, not the
  `faculties` field) but the harvester fix + re-harvest to backfill it is an
  intentionally **deferred** follow-up — don't attempt this without being
  asked; it means re-hitting Trepo's OAI-PMH endpoint for all 66k records.
- This repo's git history was rewritten 2026-07-21 (single fresh commit,
  history from the old outer repo dropped — it added no value and was mostly
  orphaned garbage inflating `.git` to 30+GB). Normal `git add` / `git commit`
  / `git push` to `master` on `origin` works going forward.

## Decisions already made — don't relitigate without reason

- **Tag field name going forward: `tags`**, not `combined_tags`. The live
  pipeline (`scripts/generate_variants.py`, what `main.py` actually calls)
  uses `tags`; `combined_tags` is an older name still referenced in some
  historical artifacts — stale, not authoritative.
- **Multi-language abstract splitting is intentional, not a bug.** When a
  thesis has abstracts in more than one language, JYU's `split_abstract_v2.py`
  pattern produces one training record per language (duplicating
  title/tags/faculty). This is deliberate augmentation given the text→tags
  framing, not something to "deduplicate" or "fix" — Tampere/Helsinki
  alignment replicates this rather than collapsing to a single best abstract.
- **Language codes: 3-letter (`fin`/`eng`/`swe`) is the target format.** For
  now, the primary training set stays fin+eng only, matching the current
  filter in `prep_split_dataset.py`/`generate_variants.py` — Swedish is
  deferred, not excluded forever. Finna scale reference (whole repo, not
  thesis-filtered): fin ≈170k, eng ≈140k, swe+ger ≈14k combined, other
  languages ≈4k each.
- **Canonical training script: `scripts/train.py`.** An older competing
  script (`model_training_GCV_3.py`, from the first-iteration repo) had a
  label-binarization bug and was retired along with the rest of that repo.

## Working conventions — be polite with shared/external resources

- **External APIs** (Trepo OAI-PMH, Finna, anything not localhost): don't
  hammer them. Space out requests, don't parallelize aggressively, and treat
  a "let's re-harvest" decision as something to confirm first, not something
  to just do because it's technically possible — see the deferred faculty
  backfill above for a live example of this being paused rather than run
  immediately.
- **Modifying datasets**: prefer writing new files over silently overwriting
  existing ones, especially once something is treated as a real artifact (an
  aligned dataset, a trained checkpoint). If asked to add a field (e.g. real
  faculty data once the harvester fix lands), do it as a clearly-scoped,
  reviewable change — not a blanket in-place rewrite of the whole dataset
  without a chance to check the result first.
- **Shared compute for training** (this machine may run other people's jobs):
  before starting any training run, check `nvidia-smi` for current GPU
  utilization/memory across all visible GPUs and pick one with enough free
  headroom for the job — never assume exclusive access. Do not launch a job
  that would push a GPU near its memory limit if something else is already
  running on it; that risks crashing someone else's process. If everything
  looks busy, say so and ask rather than picking the "least bad" option
  silently.
