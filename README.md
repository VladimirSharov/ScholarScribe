# ScholarScribe

Multi-label tag classification for university thesis metadata: given a
thesis's title + abstract, predict its subject tags. Core task framing is
**text (title + abstract) → tags**, with the abstract treated as the dominant
training signal.

This repo is the **second iteration** of the project. The first iteration
(single-university, JYU only, a scatter of loose/duplicated scripts at a repo
root) has been retired — this repo starts fresh from what used to live under
an internal `boa_strangling/` folder, restructured to be the whole project
instead of a subdirectory buried inside a larger, harder-to-navigate one.
Nothing about the modeling approach changed as part of that move; it was a
housekeeping pass, not a rewrite.

## What changed in the second iteration

- **Second university added.** Originally JYU-only; Tampere University (TAU)
  data was harvested via OAI-PMH (66,340 raw records) and aligned onto the
  same title/abstract/tags schema, including replicating JYU's
  multi-abstract-per-thesis splitting (deliberate augmentation, not
  deduplicated away — see `TAMPERE_INTEGRATION_NOTES.md`). Helsinki was added
  the same way. The merged multi-university dataset is what all training runs
  since 2026-07-16 use.
- **Tag dictionary / frequency filtering.** The raw merged tag vocabulary was
  22,027 unique tags — far too sparse to learn most of them from a handful of
  examples each. A frequency floor (minimum 5 occurrences) cut this to 8,728
  tags. This alone did **not** fix the long tail: the two rarest buckets
  (5-9 and 10-49 occurrences, ~86% of the vocabulary) still showed exactly
  zero learned signal for several training runs after the cut, which reframed
  the problem — it's a per-class example-count ceiling, not primarily a
  vocabulary-size problem. Frequency filtering still matters (it removes tags
  that are pure noise for training — a tag seen once or twice can't be
  learned by any model, only memorized-if-lucky), but it's a precondition, not
  a fix on its own. See `THESIS_LOG.md`, 2026-07-16 entry, for the full
  numbers.
- **Per-frequency-bucket adaptive thresholding, not one global threshold.**
  Tags are bucketed by training-set frequency (`5-9`, `10-49`, `50-199`,
  `200+`), and the decision threshold is searched independently per bucket
  instead of applying one cutoff to every tag's sigmoid output. Without this,
  optimizing for micro-F1 (or plain accuracy) pushes the model toward a
  degenerate local optimum: "predict most of the ~236 head (200+) tags for
  every document." That inflates recall on the frequent tags — which dominate
  micro-averaged metrics, since micro-F1 weights every *prediction*, not every
  *tag*, equally — while doing nothing for the long tail. Macro-F1 (average
  per-tag F1, so a rare tag counts as much as a common one) exposes this
  failure mode immediately; per-bucket thresholds plus weighted BCE loss
  (`pos_weight` scaled by inverse frequency, log/sqrt-damped so it doesn't
  destabilize training) are the mitigation. Tracking both micro and macro,
  and per-bucket recall/F1 broken out separately, is what surfaced this — a
  single blended metric would have hidden it.
- **Training budget matters more than expected.** A 12-epoch run plateaued
  with the 5-9/10-49 buckets essentially at zero. Extending to 36 epochs
  (same data, same loss/threshold setup) was the single biggest improvement
  measured so far: f1_macro 0.014 → 0.163, and — more importantly for
  real-world usefulness than any aggregate score — the model started
  producing genuinely varied, per-document predictions instead of a
  near-constant blanket guess. Average predictions per document went from
  wildly over-broad (100+ tags/doc during the plateau) to close to the true
  average (~4.6-5.0 tags/doc) only once training ran long enough to escape
  that local optimum. See `THESIS_LOG.md`, 2026-07-18 and 2026-07-19 entries.
- **Test-threshold-leakage fixed.** Every prior run's reported test metrics
  had thresholds fit against the same split being scored (including the final
  test evaluation) — a subtle leak, since a threshold tuned on labels it's
  then being judged against is not a fair test. Thresholds are now fit once
  on the validation set, frozen, and applied to test. The leak turned out to
  be small in practice (~2-4% relative difference), but the fix generalizes:
  it's also what makes it possible to compare two checkpoints' *raw model
  quality* under one shared, fixed threshold, rather than each checkpoint
  quietly re-calibrating its own decision rule. See `THESIS_LOG.md`,
  2026-07-21 entries.
- **A standalone demo** (`demo/app.py`, Gradio) lets you type or pick a
  title+abstract and see predicted tags from the trained model, without
  needing the training datasets or a GPU. See `demo/README.md`.

## Where to look first

- **`THESIS_LOG.md`** — dated log of every training run: dataset used,
  config, observed behavior, results, and interpretation. Written to be
  lifted directly into thesis text — the authoritative source for "what
  actually happened and what it means," more detailed than this README.
- **`TAMPERE_INTEGRATION_NOTES.md`** — technical detail for the multi-university
  schema alignment work: exact field names per source, which JYU scripts were
  reusable-as-pattern vs stale, decisions made vs still open.
- **`main.py`** — step-based pipeline orchestrator for the JYU side.
- **`scripts/train.py`** — canonical training script (weighted BCE,
  per-bucket adaptive thresholding, stratified metrics).
- **`scripts/predict.py`** — score a saved checkpoint without retraining;
  `--full_test_eval` for an honest val-fit/test-frozen evaluation,
  `--threshold_model_dir` to apply one checkpoint's calibration to another's
  weights (used for the fair 3-way checkpoint comparisons in `THESIS_LOG.md`).
- **`demo/`** — standalone Gradio app + instructions for running it on
  another machine.

## Decisions already made — don't relitigate without reason

- **Tag field name: `tags`**, not `combined_tags` (an older name still
  present in some historical artifacts).
- **Multi-language abstract splitting is intentional, not a bug** — one
  training record per language when a thesis has multiple abstracts.
- **Language codes: 3-letter (`fin`/`eng`/`swe`).** Primary training set is
  fin+eng only for now; Swedish is deferred, not excluded forever.
- **Canonical training script: `scripts/train.py`.**

## Working conventions

- **External APIs** (Trepo OAI-PMH, Finna, etc.): don't hammer them — space
  out requests, treat re-harvesting as a decision to confirm first.
- **Datasets/checkpoints**: prefer writing new files over overwriting
  existing ones once something is a real artifact.
- **Shared compute**: check `nvidia-smi` (and CPU load) before launching
  training — this machine may be running other people's jobs.
