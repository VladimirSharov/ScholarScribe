# Thesis Log — Training Runs & Results

Running, dated record of training experiments: dataset used, training configuration,
observed training behavior, and results. Written to be lifted directly into thesis
text later — each entry should stay factual and not-overdetailed (properties,
settings, behavior, results), not a narration of debugging steps.

---

## 2026-07-12 — First multi-university run (xlm-roberta-base, unfiltered 22k tags)

**Dataset**: merged split from four sources (JYU, Tampere, Vaasa, Turku), produced
externally and bridged into training format via
`boa_strangling/scripts/prepare_multiuni_split.py` →
`boa_strangling/data/data_multiuni_2026-07-12/`.
- 82,752 records total: train 65,360 / val 8,704 / test 8,688
- 22,027 unique tags, avg 5.02 tags/record, kept unfiltered (no min-frequency cut)
- 46% of tags (10,065) occur only 3-4 times in the whole corpus
- 0 unseen val/test labels (full train-set coverage verified before training)

**Training setup**: `xlm-roberta-base`, sigmoid multi-label head
(`problem_type="multi_label_classification"`), input = `<LANG=xx> title abstract`
(max 512 tokens), batch 16 (eval 32), lr 2e-5, 500 warmup steps, weight decay 0.01,
fp16, single NVIDIA A4000 (16GB), early stopping patience 3 on `eval_f1_micro`,
planned 5 epochs (20,425 steps total).

**Training behavior**: loss dropped sharply (0.68 → 0.005) within the first ~500
steps — an artifact of BCE loss under extreme label sparsity (~5 positive labels
out of 22,027 per example), not genuine convergence. `f1_micro` rose from 0.003 to
~0.023 by step 2000, then plateaued (identical values across consecutive evals).
`f1_macro` stayed at ~0 throughout — the long-tail tags were never learned.
Diagnosis: the model's sigmoid outputs never crossed the lowest threshold (0.1)
tried by `compute_metrics`, so every reported metric actually came from a top-k=7
fallback heuristic rather than true calibrated thresholding.

**Result** (test set, best checkpoint = step 2500 / epoch 0.61, stopped early via
`EarlyStoppingCallback` well short of the 20,425-step plan):

| metric | value |
|---|---|
| f1_micro | 0.0250 |
| f1_macro | 0.0000133 |
| precision_micro | 0.0213 |
| recall_micro | 0.0301 |
| subset_accuracy | 0.0 |

Near the floor for a 22k-label space — consistent with the plateau diagnosis above,
not a meaningfully-trained tagger yet.

**Known issue hit**: `create_sample_predictions_report()` in `train.py` crashed
after test evaluation (`mlb.inverse_transform([predicted_labels])` — list passed
where an ndarray was expected) whenever the per-sample threshold also fell back to
top-k. Model, tokenizer, and `test_results.json` were saved successfully before
the crash; `config.json` and the qualitative sample-prediction report were not.

**Next planned round**: reworking loss (pos_weight/focal for rare-tag gradient
signal), thresholding (per-class or per-frequency-bucket instead of one global
value), and metrics (stratified recall by tag-frequency bucket) to address the
plateau above — tracked as a separate entry once run.

**Parallel work**: label dictionary improvements (frequency filtering / semantic
clustering of rare tags) are being handled in a separate session, not this one.

---

## 2026-07-12 (second run) — weighted BCE loss + per-bucket adaptive thresholding

**Motivation**: the run above plateaued because standard BCE loss produced
vanishing gradients on rare tags (sigmoid outputs never crossed the global
threshold search's floor of 0.1), and a single global threshold hid this behind
a top-k=7 fallback. Changes made to `boa_strangling/scripts/train.py`:
- `pos_weight = log1p(neg/pos)` per class in `BCEWithLogitsLoss` (via a custom
  `WeightedBCETrainer`), log-scaled to avoid exploding on freq-3 tags (raw ratio
  ~21,800x). Range observed: min 3.79, max 11.09, mean 9.29.
- Per-frequency-bucket adaptive thresholding (bands: 3-4, 5-9, 10-49, 50-199,
  200+, matching the corpus-wide bands but computed on training-set frequency),
  replacing the single global threshold + top-k fallback.
- Stratified recall/F1 per bucket added to `compute_metrics`, to expose exactly
  where the model does/doesn't learn (the old `f1_macro` alone couldn't).
- `early_stopping_patience` 3 → 5.
- Also fixed the `create_sample_predictions_report` crash from the previous
  entry as a side effect of reusing the same bucket-threshold code there.
- Engineering note: the naive implementation (calling sklearn's `f1_score` per
  threshold per bucket) took **minutes per eval** on this 22,027-column matrix;
  rewritten as plain numpy boolean-count operations, the same computation runs
  in ~1-25s. Necessary to keep eval overhead from dominating training time at
  this label-space scale.

**Training behavior**: loss again dropped fast, but early on the model swung to
the opposite failure mode from before - massive over-prediction
(`avg_predictions_per_sample` peaked at 21,082 out of 22,027 tags at step 1000),
before self-correcting down to ~250-390/sample by mid-run and oscillating there
without fully resolving. Early-stopped at step ~8500 of the planned 20,425 (patience
exhausted after step 6500's peak).

**Result** (test set):

| metric | run 1 | run 2 |
|---|---|---|
| f1_micro | 0.0250 | 0.0060 |
| f1_macro | 0.0000133 | 0.0001 |
| recall_micro | 0.0301 | 0.1550 |
| avg_predictions/sample | 7 (top-k artifact) | 253 (real, over-predicting) |

Per-bucket breakdown (new this run - the real value of the stratified metrics):

| bucket | recall | f1 |
|---|---|---|
| 3-4 (12,049 tags) | 0.0000 | 0.0000 |
| 5-9 (4,607 tags) | 0.0000 | 0.0000 |
| 10-49 (4,215 tags) | 0.0007 | 0.0008 |
| 50-199 (934 tags) | 0.3866 | 0.0044 |
| 200+ (222 tags) | 0.1678 | 0.0334 |

Qualitative sample report (100 test theses, first time this step ran without
crashing): 43/100 had at least one correctly predicted tag.

**Interpretation**: the headline f1_micro got worse, but it's now honest rather
than masked by a fallback heuristic. The real finding is the per-bucket split:
the model learned genuine signal for moderately-common/common tags (50-199,
200+) that the old aggregate `f1_macro≈0` completely hid, while the 76% of the
vocabulary with only 3-9 examples each (buckets 3-4/5-9) got zero signal -
threshold search never found a working value for those buckets at all. `pos_weight`
also clearly needs damping or annealing-in (not full strength from step 0) - it
fixed vanishing gradients but overshot into gross over-prediction that never
fully resolved in this run's budget.

**Next planned round**: anneal or reduce `pos_weight` strength; the 3-4/5-9
bucket result is a concrete argument for collapsing/merging those rarest tags
(separate label-dictionary session) rather than expecting them to be learnable
as independent classes from 3-9 examples each.

---

## 2026-07-16 — new merged split (freq-filtered to min-5, 8,728 tags) + damped pos_weight

**Dataset**: the label-dictionary work (parallel session) landed a new merged
split, replacing the previous one (moved to `split_freq3/`). Regenerated via the
same `prepare_multiuni_split.py` → `boa_strangling/data/data_multiuni_2026-07-16/`.
- 81,738 records: train 64,614 / val 8,543 / test 8,581
- **8,728 unique tags** (down from 22,027) — the 3-4 frequency band is now empty
  (min frequency floor raised to 5); bucket sizes: 5-9: 2,930, 10-49: 4,564,
  50-199: 998, 200+: 236
- avg 4.64 tags/record; 0 unseen val/test labels

**Training setup**: same as the 2026-07-12 second run (`xlm-roberta-base`,
per-bucket adaptive thresholding, stratified metrics, `early_stopping_patience=5`),
with one deliberate change: `pos_weight = sqrt(log1p(neg/pos))` instead of raw
`log1p(neg/pos)`, per that run's own "next planned round" note — raw log1p on
this dataset was still nearly as aggressive as before (mean 8.28, max 9.47,
since even the "common" end of an imbalanced 8.7k-tag space is still rare in
absolute terms), which risked repeating the over-prediction overshoot. sqrt
compressed this to mean 2.87 / max 3.08 (min 1.94) while preserving the same
rare-tag-gets-more-weight ordering.

**Training behavior**: the damping worked as intended — over-prediction still
spiked early (`avg_predictions_per_sample` peaked at 8,432/8,728 at step 500,
epoch 0.12) but self-corrected within one more eval cycle (down to ~174 by step
1000, epoch 0.25), an order of magnitude faster than the previous run's
self-correction. `f1_micro` peaked at step 1500 (epoch 0.37) and then plateaued/
oscillated without further improvement through step 4000 (epoch 0.99);
`EarlyStoppingCallback` (patience 5) stopped the run there — under 1 of the
planned 5 epochs.

**Result** (test set):

| metric | run 1 (22k tags) | run 2 (22k tags) | run 3 (8.7k tags) |
|---|---|---|---|
| f1_micro | 0.0250 | 0.0060 | 0.0160 |
| f1_macro | 0.0000133 | 0.0001 | 0.000258 |
| recall_micro | 0.0301 | 0.1550 | 0.2456 |
| avg_predictions/sample | 7 (top-k artifact) | 253 | 138 |

Per-bucket breakdown:

| bucket | recall | f1 |
|---|---|---|
| 5-9 (2,930 tags) | 0.0000 | 0.0000 |
| 10-49 (4,564 tags) | 0.0000 | 0.0000 |
| 50-199 (998 tags) | 0.0041 | 0.0034 |
| 200+ (236 tags) | 0.7511 | 0.0165 |

Per-bucket threshold search picked the grid floor (0.01) for 50-199/200+ and the
grid ceiling (0.46) for 5-9/10-49 — i.e. for the common buckets, "predict almost
everything positive" scored best (high recall, poor precision), and for the two
rarer buckets no threshold ever separated signal from noise.

Qualitative sample report (100 test theses): 59/100 had at least one correctly
predicted tag (vs 43/100 in run 2); avg precision 0.0096, avg recall 0.2532, avg
f1 0.0184.

**Interpretation**: shrinking the tag space to 8,728 (min freq 5) did not, by
itself, fix the long-tail blind spot — 5-9 and 10-49 (86% of the vocabulary)
still show exactly zero learned signal, same as the old 3-4/5-9 buckets did
before. The model's only real learned behavior is "predict most of the 236
head tags for most examples," which is why recall on 200+ is high (0.75) but
f1 stays near-zero (massive false-positive volume) and subset accuracy is 0.0
throughout. This reframes the earlier hypothesis: the problem is less about
tag-vocabulary size and more about per-class example count — 5-9 and 10-49
still don't have enough positive examples per class for the model to
discriminate them from the rest, regardless of pos_weight strength. The sqrt
damping did work as a fix for the *stability* problem (over-prediction
self-corrected 10x faster than the undamped version), but early stopping is
now triggering very early (under 1 epoch) once the model settles into this
"always predict the head tags" local optimum, without ever being pushed past it.

**Next planned round**: the head-tag-blanket-prediction pattern suggests the
per-bucket threshold search and/or loss weighting need to actively discourage
this local optimum (e.g. precision-aware threshold selection instead of pure
F1, or a per-bucket loss term) rather than relying on pos_weight alone; also
worth checking whether early stopping on `f1_micro` is cutting the run off
right as it's settling into the local optimum, before any of the several-epoch
budget needed to move past it has been spent.

---

## 2026-07-17 — full 5-epoch run, no early stopping, best-model selection by f1_macro

**Motivation**: run 3 above was cut off under 1 epoch by `EarlyStoppingCallback`
right as the model settled into a "blanket-predict-the-head-tags" local optimum
(high recall on 200+ tags, near-zero f1 everywhere else) - the open question was
whether more training budget would let it move past that optimum. Changes to
`boa_strangling/scripts/train.py` (made by the user directly): removed
`EarlyStoppingCallback` (`callbacks=[]`, full 5 epochs / ~20,195 steps always
run), `metric_for_best_model` switched `f1_micro` -> `f1_macro` (rewards spreading
correct predictions across classes rather than nailing the few common ones), and
`save_total_limit=2` added so checkpoint disk usage stays bounded over a full
run instead of accumulating one checkpoint per 500 steps. Verified before
launch: `save_total_limit` + `load_best_model_at_end=True` is safe on the
installed `transformers` 4.46.2 - `Trainer._sorted_checkpoints` explicitly
excludes `best_model_checkpoint` from rotation, so the best checkpoint is never
deleted even if it's not among the most recent 2.

Same dataset as run 3 (`data_multiuni_2026-07-16`, 8,728 tags) and same
sqrt-damped `pos_weight`.

**Training behavior**: epochs 0-2.2 reproduced run 3's plateau almost exactly
(f1_micro ~0.015-0.016, f1_macro ~0.00026-0.00034, ~137-158 predictions/sample,
recall_200+ ~0.73-0.81) - confirming run 3 was stopped exactly at this local
optimum, not past it. Around epoch 2.35 the model went through a brief collapse
(predictions/sample dropped to 6.8, recall_200+ dropped to 0.013, f1_macro
dropped to 0.0000332) and then recovered into a new, better-calibrated regime:
predictions/sample settled into the 13-55 range (down from ~140, much closer to
the true average of ~4.6) and both f1_micro and f1_macro trended upward for the
rest of training, reaching their run maximum in the last few evals before
epoch 5. Total wall time: 8,236s (~2h17m) for the full 20,195 steps.

**Result** (test set, best checkpoint essentially == final, since f1_macro was
still climbing at the last training eval):

| metric | run 3 (early-stopped, epoch 0.99) | run 4 (full 5 epochs) |
|---|---|---|
| f1_micro | 0.0160 | 0.0225 |
| f1_macro | 0.000258 | 0.00120 |
| recall_micro | 0.2456 | 0.1458 |
| avg_predictions/sample | 138 | 55.4 |

Per-bucket breakdown:

| bucket | run 3 recall / f1 | run 4 recall / f1 |
|---|---|---|
| 5-9 (2,930 tags) | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| 10-49 (4,564 tags) | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| 50-199 (998 tags) | 0.0041 / 0.0034 | 0.2443 / 0.0137 |
| 200+ (236 tags) | 0.7511 / 0.0165 | 0.2162 / 0.0892 |

Qualitative sample report (100 test theses): 45/100 had at least one correct
tag (vs 59/100 in run 3), but with much better precision/f1 per sample - avg
precision 0.0128 (vs 0.0096), avg recall 0.1386 (vs 0.2532), avg f1 0.0235
(vs 0.0184). Fewer samples "accidentally" hit via blanket-predicting the head
tags, more samples with a genuinely calibrated, smaller prediction set.

**Interpretation**: this confirms the hypothesis from run 3 directly - early
stopping on a short-patience window was cutting training off right as the model
fell into the "predict most head tags for everyone" local optimum, before the
budget needed to move past it had been spent. Given the full 5 epochs, the
model did move past it: f1_macro improved ~4.7x, the 50-199 bucket went from
near-zero to genuinely learning (recall 0.24, f1 0.014), and the 200+ bucket
traded raw recall for far better precision (f1 0.089, ~5.4x better than run 3).
The 5-9/10-49 buckets (86% of the vocabulary) still show exactly zero signal in
every run so far, regardless of pos_weight strength or training budget - this
is the strongest evidence yet that this is a per-class example-count ceiling
(5-49 examples total, split across train/val/test, isn't enough to discriminate
that many classes) rather than something more training time or loss tuning
alone will fix.

**Next planned round**: with the "stop too early" failure mode addressed,
remaining options are (a) revisit early stopping with a much longer patience
or a plateau-aware criterion instead of removing it outright, now that we know
the collapse-and-recover pattern at epoch ~2.3 is real and not noise; (b) target
the 5-9/10-49 buckets specifically, most likely via the label-dictionary
session (merging/collapsing rather than treating as independent classes) since
training-side changes across three runs haven't moved them off zero.

---

## 2026-07-18 — full 12-epoch run, no early stopping, `f1_macro` best-model selection

Same dataset as runs 3/4 (`data_multiuni_2026-07-16`, 8,728 tags), same
sqrt-damped `pos_weight`, `save_total_limit=2` + `load_best_model_at_end=True`
(safe per the run-4 checkpoint-rotation check). Only change: `num_train_epochs`
12 instead of 5, to see whether more budget keeps paying off past where run 4
ended. Total: 48,468 steps, wall time 24,159s (~6h43m) on GPU 1 (GPU 0 was idle
the whole time - no contention).

**Training behavior**: reproduced run 3/4's plateau through epoch ~2.2
(f1_micro ~0.0155-0.0158, f1_macro ~0.00025-0.00027, ~135-155
predictions/sample), then went through **three** distinct collapse-and-recover
regime shifts rather than run 4's one:

| epoch | event | predictions/sample | f1_micro | f1_macro |
|---|---|---|---|---|
| ~2.2 (pre-shift) | plateau (same as runs 3/4) | ~144 | 0.0155 | 0.00026 |
| ~2.35 | collapse | 14.4 | 0.0084 | 0.00004 |
| ~4.58 (recovered) | partial recovery, new regime | 86.2 | 0.0210 | 0.00246 |
| ~4.95 | second collapse | 9.6 | 0.0604 | 0.00224 |
| ~9.41 (recovered) | best validation `f1_macro` of the run | 43.2 | 0.0506 | 0.01977 |
| ~9.78 | third collapse | 3.8 | 0.1672 | 0.01433 |
| 12.0 (final) | held/slowly improved | 4.0 | 0.1796 | 0.01778 |

Each shift moves to a sparser, more-calibrated operating point (fewer
predictions/sample, closer to the true average of ~4.6) and, apart from a
transient dip right at each collapse, comes out ahead on `f1_macro` afterward.
`load_best_model_at_end` selected **checkpoint-38000** (epoch 9.41, the peak
just before the third shift) as best by validation `f1_macro`; checkpoint-48468
(final step) was also retained under `save_total_limit=2` as expected.

**Result** (full test set, 8,581 samples, best checkpoint = step 38000):

| metric | run 4 (5 epochs) | run 5 (12 epochs) |
|---|---|---|
| f1_micro | 0.0225 | **0.1715** |
| f1_macro | 0.00120 | **0.0140** |
| precision_micro | - | 0.1891 |
| recall_micro | 0.1458 | 0.1569 |
| avg_predictions/sample | 55.4 | **3.85** (true avg is 4.64) |
| subset_accuracy | - | 0.0040 |

Per-bucket breakdown:

| bucket | run 4 recall / f1 | run 5 recall / f1 |
|---|---|---|
| 5-9 (2,930 tags) | 0.0000 / 0.0000 | 0.0054 / 0.0082 |
| 10-49 (4,564 tags) | 0.0000 / 0.0000 | 0.0150 / 0.0256 |
| 50-199 (998 tags) | 0.2443 / 0.0137 | 0.1432 / 0.1382 |
| 200+ (236 tags) | 0.2162 / 0.0892 | 0.3312 / 0.2873 |

The 5-9/10-49 buckets are **no longer flat zero** for the first time across five
runs, though still barely above it - genuinely marginal, not yet a real result,
but the first crack in what looked like a hard per-class-example-count ceiling.

Qualitative sample report (100 random test theses, own thresholds): 55/100 had
at least one correct tag, avg precision 0.133, avg recall 0.229, avg f1 0.168 -
both the hit-rate and the precision/recall/f1 are well above run 4's on every
axis (run 4: 45/100, precision 0.0128, recall 0.1386, f1 0.0224).

**Caveat (applies to every run in this log, not just this one)**: per-bucket
thresholds are fit by maximizing F1 directly against the true labels of
whatever split is being scored - including the final test-set evaluation. This
is optimistic (the threshold search has oracle access to the labels it's then
scored against) and likely inflates the absolute test numbers somewhat. It does
not undermine the run-to-run comparisons above, since every run in this log
uses the identical procedure, but the absolute test f1 numbers should not be
read as what a truly held-out threshold (fit only on val, frozen, then applied
to test) would give. Worth fixing before any number here goes into the thesis
as a headline result.

**Interpretation**: more budget kept paying off well past where run 4 stopped -
f1_micro is now ~7.6x run 4's, f1_macro ~11.7x. The recurring collapse-and-
recover pattern (now seen 3x in this run alone, plus once in run 4) looks like
a real, repeatable training dynamic for this loss/data combination, not noise:
the model periodically re-discovers a sparser, better-calibrated prediction
policy, sacrificing recall on the plateau's incumbent policy for a large f1
gain once it resettles. Whether a 4th shift exists past epoch 12 is unknown -
the run wasn't long enough to tell whether the model had truly converged or
just paused before another shift.

**Tooling added alongside this run**: `boa_strangling/scripts/predict.py`
(load any `results/<run>/` checkpoint, refit thresholds from that run's own
val set, tag ad hoc text/a file/random test samples) and
`boa_strangling/scripts/compare_predictions.py` (diff two prediction reports -
by `id` when both have one, else positional with a warning - reporting
per-run precision/recall/f1, tag-set jaccard, and a 4-way correct/incorrect
breakdown). Both are model-agnostic (any source emitting
`{id, true_tags, predicted_tags}` records works), so a future llama-based
comparison run can be diffed the same way. `create_sample_predictions_report`
now also stores `id`/`university` per sample so future reports can be aligned
by id rather than falling back to position (run 5's own report has ids; runs
3/4's don't, predating this change).

**Next planned round**: (a) let a run go longer than 12 epochs to see whether
the collapse-and-recover pattern continues and keeps paying off, or whether
this run's endpoint is close to a real ceiling; (b) fix the test-threshold
leakage caveat above (fit on val, freeze, apply to test) before quoting
absolute numbers in the thesis; (c) the 5-9/10-49 buckets moving off exact
zero (even if barely) reopens the question of whether more budget alone closes
that gap eventually, versus still needing the label-dictionary session's
merging approach.

---

## 2026-07-19 — checkpoint-38000 vs checkpoint-48468 (run 5), under honest (val-fit, frozen) thresholds

The wandb `f1_macro` curve for the 2026-07-18 run shows step 38000 (epoch 9.41)
as its highest point - that's why `load_best_model_at_end` selected it as
"best" and saved it into `results/.../model/`. Given the test-threshold-leakage
caveat logged above, that "best" label is suspect: it's based on the online
per-step val eval, but the two checkpoints had never been compared against each
other under one identical, non-test-leaked procedure. Did that with
`predict.py`/`compare_predictions.py`: loaded both checkpoints' raw weights
directly, refit per-bucket thresholds independently for each from the run's
own `val.json` (never touching test labels for thresholding), then ran both on
the same fixed 150-sample draw from the test set (`--seed 42`, so both sides
see identical rows) and aligned the results by `id`.

**Result**: checkpoint-48468 (final step) clearly beats checkpoint-38000 under
this honest procedure - the wandb "peak" does not hold up.

| | ckpt-38000 (epoch 9.41, "best") | ckpt-48468 (final, epoch 12) |
|---|---|---|
| precision | 0.0373 | 0.1659 |
| recall | 0.2042 | 0.1192 |
| f1 (per-sample avg) | 0.0540 | 0.1214 |
| avg tags predicted/doc | 42.54 | 3.27 |
| val-fit thresholds | 5-9=0.01, 10-49=0.01, 50-199=0.11, 200+=0.21 | 5-9=0.01, 10-49=0.06, 50-199=0.11, 200+=0.26 |

(true avg tags/doc in this sample: ~4.6-4.9, matching the run-level average.)

Of 150 aligned samples: both models hit >=1 true tag on 55; only ckpt-38000 on
30 more; only ckpt-48468 on 2 more; neither on 63. Read alone, "only ckpt-38000
correct on 30 more samples" looks like an advantage for 38000 - but it's
explained entirely by volume: at ~42 guesses/doc it wins recall by brute-force
spray, not precision. Its per-sample f1 is under half of ckpt-48468's.

**Root cause of the swing**: the 10-49 frequency bucket (the largest bucket by
tag count) ties between threshold 0.01 and 0.06 on val for ckpt-38000, and the
tie breaks toward 0.01 (first-found-wins on ties, see `find_best_threshold_per_group`).
That one bucket's threshold choice is what drives the 42 vs 3 predictions/doc
gap - a small, near-tied threshold choice has an outsized effect on the total
prediction count because so many labels live in that bucket. This is a second,
distinct issue from the test-threshold-leakage caveat (that one is about which
split thresholds are fit on; this one is about tie-breaking instability within
a single split's fit) - both point the same direction: don't trust a single
checkpoint's apparent "peak" f1_macro without re-deriving its operating point
independently.

**One case inspected by hand** (`oai:jyx.jyu.fi:123456789/51462`, 745-char
Finnish abstract on user-centered design for mobile app development; true tags:
`käytettävyys`, `käyttäjäkeskeinen suunnittelu`, `käyttökelpoisuus`,
`mobiilisovellukset`):
- ckpt-38000: 18 predicted tags, hit 3/4 true (missed `mobiilisovellukset`),
  precision 0.167, recall 0.75. Alongside the 3 correct hits, predicted several
  plausible-sounding but wrong tags for this abstract: `hakukoneoptimookti`
  (SEO), `virtuaaliympäristö`, `mobiilipelit`, `tekoäly`.
- ckpt-48468: 7 predicted tags, hit 3/4 true (same recall, different miss -
  caught `mobiilisovellukset` instead of `käyttäjäkeskeinen suunnittelu`),
  precision 0.429.
- Both models' top-10 raw scores agree closely in ranking (`käyttäjäkokemus`
  and `käytettävyys` top two for both, at 0.47/0.45 for ckpt-38000 vs 0.61/0.56
  for ckpt-48468) - the underlying probability estimates are qualitatively
  similar and ckpt-48468's are if anything more confident on the true positives.
  The difference in output quality comes almost entirely from where the
  threshold cut lands, not from a fundamentally different ranking of
  candidates. Manually reading the abstract, ckpt-48468's shorter, cleaner list
  is clearly the more usable output of the two.

**Conclusion**: keep using checkpoint-48468 (already what `results/.../model/`
should point at going forward, unless the code is updated to reload it - right
now `model/` holds the weights of whichever checkpoint `load_best_model_at_end`
picked, i.e. checkpoint-38000). The wandb macro-graph "peak" at step 38000 is a
monitoring artifact of the same threshold-fitting instability documented in the
2026-07-18 entry's caveat, not evidence that continuing training past that
point hurts. This also means `metric_for_best_model="f1_macro"` combined with
per-step-refit thresholds is not a reliable checkpoint-selection signal as
currently implemented - worth keeping in mind for future runs (e.g. selecting
by final step, or by a metric computed under frozen thresholds, rather than by
the live per-step best).

Tooling note: `predict.py` gained `--model_dir` (point at any
`checkpoints/checkpoint-N` instead of the run's saved `model/`) and `--seed`
(so two calls draw the same test rows) to support this comparison. Full outputs
kept in `results/xlm_roberta_multilabel_20260718_104944/predictions/checkpoint_comparison_38000_vs_48468/`.

---

## 2026-07-19 (second run) — 36-epoch run, aborted LR-restart variant, then linear rerun

**Aborted variant**: first attempt at extending past 12 epochs added
`lr_scheduler_type="cosine_with_restarts"` (`num_cycles=6`) on top of the
2026-07-18 config, otherwise unchanged, 36 epochs. Killed manually at epoch
~16/36 (PID 475775, ~6h in) after the eval trajectory showed no path to
improvement: `f1_macro` flatlined at ~0.0002-0.0003 (vs the 2026-07-18 run's
final 0.0140) and `avg_predictions_per_sample` stuck at 100-150/doc (vs 3.85)
for the entire logged run, with the 5-9/10-49 bucket thresholds pinned at the
grid ceiling (0.46) throughout - i.e. the model never found separable signal in
those buckets at all, worse than the 12-epoch run. One brief dip to
avg_predictions=7-15 around epoch 5.45-6.19 looked like the same
collapse-and-recalibrate event seen repeatedly in the 2026-07-16 run, but it
reverted back to the diffuse regime by epoch 6.93 and never recurred - read as
the LR restart (a cycle roughly every 6 epochs under `num_cycles=6`/36 epochs)
kicking the model back out of the calibration it had started to find. Fix:
removed both `lr_scheduler_type` and `lr_scheduler_kwargs` from `train.py`,
reverting to the `Trainer` default (linear decay), matching the 2026-07-18 run.
`num_train_epochs=36` and the coarser `eval_steps`/`save_steps=1000`,
`save_total_limit=3` were kept. Rerun launched immediately after on GPU 1 (both
GPUs idle at the time).

**Result: by far the best run to date.**

| metric | 2026-07-18 (12 ep, linear) | 2026-07-19 (36 ep, linear) |
|---|---|---|
| f1_micro | 0.1715 | **0.3449** |
| f1_macro | 0.0140 | **0.1627** |
| precision_micro | 0.1891 | 0.3311 |
| recall_micro | 0.1569 | 0.3598 |
| avg predictions/sample | 3.85 | 5.04 |
| avg true/sample | 4.64 | 4.64 |
| subset_accuracy | 0.0040 | 0.0141 |
| train_runtime | 24,159s (~6h43m) | 49,045s (~13h37m) |

Per-bucket recall/f1 (2026-07-18 -> 2026-07-19) - the long-tail buckets that
were barely-nonzero last time are now genuinely working:

| bucket | recall (18th -> 19th) | f1 (18th -> 19th) |
|---|---|---|
| 5-9 | 0.0054 -> **0.2161** | 0.0082 -> **0.1699** |
| 10-49 | 0.0150 -> **0.2179** | 0.0256 -> **0.2417** |
| 50-199 | 0.1432 -> 0.3887 | 0.1382 -> 0.3494 |
| 200+ | 0.3312 -> 0.4923 | 0.2873 -> 0.4630 |

**Training dynamic - qualitatively different from the three-shift pattern in
the 2026-07-18 run.** Full trajectory (145 eval checkpoints, every 1000 steps):
early epochs (0-2) plateau exactly as before (`f1_macro` ~0.0002-0.0004,
avg_predictions 130-165). A brief partial wobble at epoch ~2.23 (avg_predictions
dips to 26, reverts by epoch 3.71) foreshadows the real transition: at **epoch
6.19**, avg_predictions drops sharply from ~41 to 4.8 and `f1_micro` jumps from
0.044 to 0.164 in a single eval step. Unlike the 2026-07-18 run, this shift does
not collapse-and-revert again - instead `f1_macro`/`f1_micro` climb *steadily
and roughly monotonically* for the next ~29 epochs (epoch 6 -> 36: f1_macro
0.013 -> 0.171, f1_micro 0.164 -> 0.361), with visible diminishing returns
setting in only in the final third (epoch 25 -> 36: f1_macro 0.159 -> 0.171,
much flatter than epoch 6 -> 20's 0.013 -> 0.131). `avg_predictions_per_sample`
settles and stays in the 4.3-5.5/doc range from epoch ~13 onward - close to the
true average of 4.64.

`load_best_model_at_end` (by validation `f1_macro`) selected checkpoint-117000
(epoch 28.97, best_metric 0.1732) over the final checkpoint-145404 (epoch 36,
val f1_macro ~0.171) - a narrow margin (~1.3% relative), unlike the 2026-07-18
run's checkpoint-38000-vs-48468 case where the "best" pick was actually far
worse under honest thresholds. Given how close these two checkpoints track
here, that earlier finding's caveat (per-step-refit thresholds make
`load_best_model_at_end` an unreliable selector) is worth rechecking here too,
but the stakes look much lower this time - not repeated in this session.
`save_total_limit=3` retained checkpoint-117000 (best), checkpoint-145000, and
checkpoint-145404 (final).

**Qualitative sample (from `create_sample_predictions_report`, 100 test-set
samples, thresholds 5-9=0.11/10-49=0.11/50-199=0.26/200+=0.41)**: precision
0.387, recall 0.390, f1 0.388, 97/100 samples got at least one prediction,
**78/100 got at least one correct tag** - up from the 2026-07-18 run's 55/100
(precision 0.133, recall 0.229, f1 0.168). Every axis improved substantially.

**Caveat carried forward**: `test_results.json`'s reported metrics still use
the same per-step, same-split threshold refitting flagged in the 2026-07-18
entry (`make_compute_metrics` fits thresholds against the split it's scoring,
including the final test call) - so these absolute numbers are still optimistic
versus a true val-fit/test-frozen procedure, same as every prior run. The
run-to-run comparison above stays valid since the same leaky procedure applies
identically on both sides.

**Interpretation**: this directly answers the open question from the
2026-07-18 entry ("try longer than 12 epochs") - more budget was not just a
minor incremental gain, it looks like it was the main thing missing. The
5-9/10-49 buckets going from "barely above zero" to real double-digit recall
suggests the earlier flat-zero ceiling across five runs was a
training-budget problem, not a hard architectural or label-dictionary-shaped
limit - though the label-dictionary/merging track (paused per the user's
2026-07-18 message) may still help further given the per-bucket numbers are
still well below the 200+ bucket's. The steady monotonic climb with only mild
flattening by epoch 36 suggests there may be a small amount of further
improvement available past 36 epochs, but the slope is shallow enough that
this looks close to a real ceiling for this setup (weighted BCE + per-bucket
thresholds + linear decay, unchanged data/pos_weight), rather than a run cut
short mid-improvement the way the 12-epoch run may have been.

**Next planned round**: (a) decide whether the test-threshold-leakage fix
(fit-on-val/freeze/apply-to-test) is worth implementing in `train.py` itself
now, given two consecutive runs have depended on it for real interpretation;
(b) if pursuing more epochs, watch specifically whether the epoch 25-36
flattening continues or whether it's another pre-shift plateau; (c) revisit
whether the label-dictionary/merging approach is still worth reconsidering for
the 5-9/10-49 buckets now that they're no longer flat zero, given how much
closer they now are to the 50-199/200+ buckets.

---

## 2026-07-21 — test-threshold-leakage fix

Item (a) from the previous entry's "Next planned round": implemented the
fit-on-val/freeze/apply-to-test procedure directly in `train.py`, and used it
to re-score the existing checkpoint-117000 (the 2026-07-19 run's saved "best"
model) rather than retraining.

**Code fix** (`boa_strangling/scripts/train.py`): `find_best_threshold_per_group`
was previously called inside `compute_metrics`, which fit thresholds fresh
against whatever split it was handed - including the final
`trainer.evaluate(test_dataset)` call, meaning the reported test thresholds
were chosen by looking at test labels directly. `main()` now fits thresholds
once against `val_dataset` via `trainer.predict()`, freezes that threshold
map, and applies it to `test_dataset` predictions to compute
`test_results.json` manually (same metric functions, same schema, `eval_`
prefix preserved for downstream tooling). `create_sample_predictions_report`
had the same bug on its 100-row sample (thresholds fit on the very rows being
scored) and now takes the frozen val-fit threshold map as a parameter instead.
Training-time per-eval-step thresholding (fit on val, used for
`metric_for_best_model` checkpoint selection) is untouched - that's normal
validation-set threshold tuning, not the leakage case.

`predict.py` gained a `--full_test_eval` flag (reusing `compute_prf_stats`/
`stratified_recall` from `train.py`) to score an entire test set with frozen,
val-fit thresholds and report the same aggregate schema, so an already-trained
checkpoint can be re-scored honestly without retraining.

**Honest re-evaluation of checkpoint-117000**: run via
`predict.py --model_dir .../checkpoints/checkpoint-117000 --full_test_eval`,
CPU-only (`CUDA_VISIBLE_DEVICES=""`, threads capped to 12) since both GPUs
were occupied by another user's jobs at the time. Result saved to
`boa_strangling/results/xlm_roberta_multilabel_20260719_151127/predictions/full_test_eval_checkpoint-117000.json`.
The original `test_results.json` for this run is checkpoint-117000's numbers
too (`model/` was confirmed via md5sum to be checkpoint-117000, being the
`load_best_model_at_end` selection) - so this is a direct before/after on the
same weights, isolating the leakage effect:

| metric | leaky (test-fit thresholds) | honest (val-fit, frozen) |
|---|---|---|
| f1_micro | 0.3449 | 0.3383 |
| f1_macro | 0.1627 | 0.1688 |
| precision_micro | 0.3311 | 0.3182 |
| recall_micro | 0.3598 | 0.3611 |
| subset_accuracy | 0.0141 | 0.0143 |
| avg_predictions_per_sample | 5.04 | 5.26 |

| bucket | threshold leaky→honest | recall leaky→honest | f1 leaky→honest |
|---|---|---|---|
| 5-9 | 0.06→0.06 | 0.216→0.216 | 0.170→0.170 |
| 10-49 | 0.21→0.16 | 0.218→0.266 | 0.242→0.241 |
| 50-199 | 0.26→0.31 | 0.389→0.346 | 0.349→0.349 |
| 200+ | 0.36→0.36 | 0.492→0.492 | 0.463→0.463 |

**Interpretation**: the leakage effect on this run was small - overall
f1_micro/f1_macro move by roughly 2-4% relative, in different directions
(f1_micro slightly down, f1_macro slightly up), not the dramatic swing seen in
the earlier honest 38000-vs-48468 checkpoint comparison. Two buckets
(5-9, 200+) land on the exact same threshold either way; only 10-49 and
50-199 shift, trading recall against each other while their F1s stay flat -
consistent with a near-tied threshold region rather than a real difference in
model quality. Net takeaway: the qualitative story of the 2026-07-19 run (huge
jump over the 12-epoch run, long-tail buckets moving off flat-zero) is robust
and not an artifact of the leakage; going forward, `train.py`'s
`test_results.json` is honest by construction, so this caveat no longer needs
to be repeated for runs after this one.

---

## 2026-07-21 (second entry) — 3-way checkpoint comparison, all saved checkpoints of the 2026-07-19 run

Compared all 3 checkpoints retained by `save_total_limit=3` from the
2026-07-19 run (`checkpoint-117000` = the `load_best_model_at_end` pick on val
`f1_macro`=0.1732; `checkpoint-145000`; `checkpoint-145404` = final step) on
an identical, fixed-seed (`--seed 42`), 100-row draw from the run's test set,
via `predict.py --model_dir <ckpt> --seed 42 --num_samples 100` (thresholds
fit fresh per checkpoint on val, frozen before scoring - honest by the
2026-07-21 fix above) and `compare_predictions.py` pairwise. Both GPUs were
confirmed idle before running. Full outputs, pairwise diffs, and a 10-row +
3-row qualitative selection are saved under
`boa_strangling/results/xlm_roberta_multilabel_20260719_151127/predictions/checkpoint_comparison_3way/`.

**Aggregate result (same 100 rows, all 3 checkpoints)**:

| checkpoint | precision | recall | f1 | true positives (of ~464 possible) | total tags predicted |
|---|---|---|---|---|---|
| 117000 (val-"best") | 0.379 | 0.345 | 0.309 | 175 | 509 |
| 145000 | 0.419 | 0.349 | 0.337 | 173 | 422 |
| 145404 (final) | 0.416 | 0.350 | 0.337 | 174 | 422 |

Recall/true-positive-count is essentially identical across all three (175 vs
173 vs 174) - the checkpoints are not meaningfully different in what they
manage to find. The difference is precision: `checkpoint-117000` predicts
~20% more tags overall (509 vs 422) for the same recall, which is why its
val-time "best" `f1_macro` (a narrow win, per the 2026-07-19 entry) does not
hold up as a real advantage on this test draw - it's actually the worst of
the three by f1 on the 100-row sample. `checkpoint-145000` and
`checkpoint-145404` (only 404 steps apart, effectively the same weights) are
close to indistinguishable: tag-set Jaccard overlap 0.988 across the 100 rows,
identical aggregate precision/recall/f1, and only 1 of the 100 rows where
their predictions actually differ.

**Root cause - per-checkpoint threshold calibration, not model capability**:
each checkpoint's val-fit thresholds differ meaningfully in the long-tail
buckets:

| bucket | 117000 | 145000 | 145404 |
|---|---|---|---|
| 5-9 | 0.06 | 0.11 | 0.11 |
| 10-49 | 0.16 | 0.21 | 0.21 |
| 50-199 | 0.31 | 0.31 | 0.31 |
| 200+ | 0.36 | 0.41 | 0.41 |

`checkpoint-117000`'s val-optimal thresholds are systematically lower for the
rarer buckets, so it fires more predictions per document - not because it
"knows more," but because its raw sigmoid outputs were less separated at that
point in training and needed a looser cutoff to hit its (narrowly) best val
F1. By the final ~400-2500 steps the model's calibration tightened, needing
higher thresholds for the same val F1, and that tighter calibration
generalizes to test as *better* precision at *equal* recall. This is a
concrete illustration of why "best checkpoint by val metric" is not
automatically "best checkpoint on new data" once threshold-fitting is in the
loop - the threshold search can find a different point on the
precision/recall trade-off for each checkpoint, and a narrow val-metric win
can hide a real test-time precision loss.

**Qualitative reading (10 of the 100 rows, selected for cross-checkpoint
disagreement, spanning JYU/Tampere/Vaasa sources)**: confirms the aggregate
story is a population-level tendency, not a per-document rule - on 9 of the
18 rows where correctness actually differed, `checkpoint-117000` won (usually
by finding 1-2 extra correct tags among several extra predictions, e.g.
`oai:trepo.tuni.fi:10024/149571`: 117000 predicts 10 tags/gets 4 right vs
145000's 6 tags/2 right); on the other 9, `checkpoint-117000` predicted
*fewer* or *zero* tags where 145000/145404 correctly predicted one (e.g.
`oai:trepo.tuni.fi:10024/237702`, `.../143820`) - i.e. the threshold
difference is a genuine per-example shift in the model's calibrated
confidence, not a uniform "117000 always predicts more" rule.

**One interesting example per checkpoint (own selection)**:
- **checkpoint-117000** - `oai:trepo.tuni.fi:10024/149571` (true_n=14): the
  clearest case of its recall payoff - predicts 10 tags and catches 4 correct
  vs the other checkpoints' 6 tags/2 correct, i.e. on richly-tagged documents
  its looser threshold genuinely finds more true tags, which is exactly the
  effect invisible in the aggregate precision-loss framing above.
- **checkpoint-145000** - `oai:trepo.tuni.fi:10024/237702` (true_n=2):
  `checkpoint-117000` predicts nothing here while `checkpoint-145000`
  predicts exactly `asuntomarkkinat` (correct, P=1.00). Notable because it's
  the opposite of the "117000 over-predicts" pattern - a reminder that the
  per-checkpoint threshold difference doesn't mechanically translate to "117000
  always predicts more per document," since the underlying sigmoid outputs
  for this doc/tag pair genuinely differ between checkpoints, not just the
  cutoff applied to them.
- **checkpoint-145404** - `oai:trepo.tuni.fi:10024/137618` (true_n=6): the
  single row (of 100) where `checkpoint-145404` differs from its near-twin
  `checkpoint-145000` - it adds `riskit` (correct), raising recall from 0.50
  to 0.67. The only qualitative trace, across this whole sample, of the extra
  404 training steps between the two - consistent with the shallow-but-still-
  positive late-training slope noted in the 2026-07-19 entry.

**Takeaway**: this checkpoint-selection gap is much narrower than the earlier
38000-vs-48468 case (that was a threshold near-tie that flipped a checkpoint
from "looks fine" to "clearly worse"; this is real checkpoints late in a
well-behaved run, all in the same performance neighborhood) - but it still
means `load_best_model_at_end`/`metric_for_best_model="f1_macro"` picked the
checkpoint that turns out to have marginally worse test-time precision at
equal recall. Not planning to change checkpoint-selection logic on the
strength of one comparison, but worth watching in future runs: if a val-time
"best" checkpoint keeps trending toward looser tail-bucket thresholds than
later checkpoints, that's a signal the model-selection metric and the actual
test objective (precision-weighted, in effect, once thresholds are refit
honestly) aren't perfectly aligned.

---

## 2026-07-21 (third entry) — same 3-way comparison under one fixed (shared) threshold

Follow-up to the entry above, which let each checkpoint fit its own val
thresholds - a confound, since a real capability difference and a pure
calibration difference look the same in that setup. Reran all 3 checkpoints
on the **full test set** (8581 rows, not the 100-row sample) using thresholds
fit once from `checkpoint-145000` and frozen identically for all three, via a
new `predict.py --threshold_model_dir` flag (fits thresholds from one
checkpoint's val predictions, scores a different checkpoint's weights with
them). Results under `.../predictions/checkpoint_comparison_fixed_thresholds_145k/`.

| metric | 117000 | 145000 | 145404 |
|---|---|---|---|
| f1_micro | 0.3469 | 0.3563 | 0.3565 |
| f1_macro | 0.1521 | 0.1669 | 0.1669 |
| precision_macro | 0.1529 | 0.1651 | 0.1649 |
| recall_macro | 0.1775 | 0.1977 | 0.1978 |
| recall (5-9) | 0.1249 | 0.1446 | 0.1450 |
| recall (10-49) | 0.2179 | 0.2417 | 0.2419 |
| recall (50-199) | 0.3458 | 0.3668 | 0.3674 |
| recall (200+) | 0.4579 | 0.4735 | 0.4736 |

Under one shared decision rule, `checkpoint-117000` is worse on **every**
metric and **every** frequency bucket - not a precision/recall trade-off this
time, a real, consistent gap (f1_macro -8.9% relative, recall_macro -10.2%,
widest in the rarer buckets: 5-9 recall -13.6%, narrowing to 200+ recall
-3.3%). `checkpoint-145000` and `checkpoint-145404` remain effectively
identical, as before.

**This resolves the previous entry's ambiguity.** The per-checkpoint-threshold
comparison showed `checkpoint-117000` occasionally winning on individual rows
(9/18 divergent cases) - that pattern was a side effect of letting each
checkpoint threshold-shop on val for its own best F1; `checkpoint-117000`'s
looser calibration bought it a few extra correct tags on richly-tagged
documents at a broader precision cost, which looked like a mixed picture at
the row level. Once that shopping freedom is removed, `checkpoint-117000`
isn't a different trade-off, it's simply less trained: worse recall *and*
worse precision than the two checkpoints ~28,000-28,404 steps ahead of it.

**Practical implication (revised from the previous entry)**: `load_best_model_at_end`
with `metric_for_best_model="f1_macro"` picked `checkpoint-117000` as "best"
during training only because thresholds were being fit fresh per
evaluation step on val, which let an under-trained checkpoint's val score
look competitive by tolerating a looser decision rule. That's specific to
this train-time thresholding setup, not a sign of a fundamentally misleading
metric - the actual final checkpoint (`checkpoint-145404`) is genuinely
better once thresholds are held fixed. No code change indicated by this
alone (the checkpoint saved as `model/` and used in all "run 6" reporting
happens to be `checkpoint-117000`, which per this comparison is very slightly
behind `checkpoint-145404`'s already-honest `test_results.json` numbers -
practically negligible, but if a future run's val-vs-fixed-threshold gap
looks large, that's the signal to reconsider using train-time per-eval
thresholding for `metric_for_best_model` at all).
