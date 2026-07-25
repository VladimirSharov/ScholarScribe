# Past Clearness — active-recall log

> Before each burial, a short quiz tests that you actually understand *why* the
> superseded thing is superseded — which (a) catches a "dead" file that's secretly
> load-bearing, and (b) leaves you able to defend these choices to a supervisor.
> The quiz **gates the burial**: files aren't deleted until the family's questions
> are answered. Your answers + the resolution (correct answer / the lesson) are
> recorded here as a permanent trace of the reasoning.

---

## Cycle 1 — `llm_generate_tags_sv*` family

**Dependency check:** no live `boa_strangling/` reference; only consumer is the
(also-graveyard) `tag_evaluation_v*` family via `generated_tags_results*.json`;
only other mentions are narrative comments in the abandoned root `main.py`. → safe.

**Quiz (answer in chat; I'll grade + record):**

1. Your `PROJECT_ORGANIZATION.md` picked "latest" partly by version number and file
   size. Two traps hide in this family — name them. (Hint: one is about the *number*,
   one is about *size*.)
2. `sv5` is the keeper. Name two robustness features it has that `sv3` lacks — the
   ones that matter for a multi-hour run over the whole dataset.
3. `sv4` is the largest file but is **not** just "a fuller sv5". What different *job*
   does it do, and where did that capability go afterwards?
4. Every version feeds the model `tag_count = len(original tags)`. Why is that a
   methodological problem for evaluating the LLM approach, and would a real deployment
   ever have that number?
5. All versions use `abstract[0]` (the first abstract only). Which later project
   decision does that quietly contradict, and which side is "right" for the text→tags framing?

**Reckoning (2026-07-23) — user's answers + resolution:**

1. *User:* the versions grew by copy-paste-then-adapt-a-feature, so file size isn't
   very telling. **✓ half.** Got the *size* trap: `sv4` is the biggest yet not the most
   advanced — size tracked scope, not lineage. The *number* trap went unnamed: **`sv3`
   (Dec 2024) is older than `sv2` (Mar 2025)** — the version integer doesn't track time.
   The copy-paste-and-branch insight is exactly why: lineage forked, it wasn't linear.

2. *User:* saving by checkpoints, not only at the end. **✓.** That's one. The second:
   **resumability** — `sv5` loads existing results and skips already-processed
   `identifier`s, so a killed run continues instead of restarting. (Plus signal-safe
   `SIGINT/SIGTERM` → save-then-exit, and atomic `temp`+`os.replace` writes.)

3. *User:* couldn't locate sv4; guessed it generated something besides tags.
   **✗ — corrected.** `sv4` lives at `zdepricated_versions/llm_generate_tags_sv4.py`.
   It's not a generator at all — it's a **multi-model benchmark harness** (`TagEvaluator`):
   runs several model endpoints in parallel, computes precision/recall/F1 itself, and
   writes a comparative markdown report ranking them. That evaluation capability later
   moved out into the `tag_evaluation_v*` family, which is why `sv5` could drop metrics
   and be a pure production generator.

4. *User:* early results were bad, so I made it emit the top-k highest tags with k =
   number of tags in the original. **✓ — and this is the confession.** Feeding
   `tag_count = len(original tags)` leaks the ground-truth **cardinality**: the model is
   told how many tags to produce. A real deployment never knows that count, so any score
   earned this way is optimistic. (Ties directly to the top-k threshold fallback later
   flagged in `THESIS_LOG`.)

5. *User:* was unaware two abstracts could exist; discovered it; was too lazy to rebuild
   the dataset so "played with 0 and 1" indices; eventually built a proper split-by-
   abstract dataset and restarted the pipeline — which is why the old project has many
   `data_split*` versions. Also recalled the abstract's **language** wasn't in the
   original parse, so it needed either language detection or going back to Theseus/Trepo
   to fetch it — feeding the per-language (FI/EN) prompt in `sv5`. **✓✓ full recall.**
   The betrayed decision: **two-language abstracts should become two training records
   (augmentation), not one** — `abstract[0]` throws the second away. Correct side for
   text→tags: keep both. This single realization spawned `split_abstract*.py`,
   `language_analysis.py`/`detect_lang_utils.py`, and the data_split sprawl.

**Verdict:** understanding verified. `sv2/sv3/sv4` cleared for burial; `sv5` kept.

---

## Cycle 2 — `model_training_GCV_*` family (`GCV_2`, `GCV_singleSoft`, `GCV_3`)

**Dependency check:** no live `boa_strangling/` reference (`grep -rl GCV boa_strangling/`
→ nothing). Only mentions are narrative (`main.py`, the doc set). Recovery anchor is
commit `a3c516c`, **not** `origin/master` (which predates these files). → safe.

**Quiz (answer in chat; I'll grade + record):**

1. `GCV_3` is the newest file (May 2025) and the one now flagged as *buggy*, while
   `GCV_2` (Feb 2025) is older and, on one specific point, *correct*. What is the one
   core thing `GCV_2` got right that `GCV_3` **regressed** — and why does that break a
   multi-label / `BCEWithLogitsLoss` setup specifically? (This is the "label-binarization
   bug" in your own words.)

2. `GCV_3`'s empty-example fallback does `valid_tags = [0]`. Two separate things are
   wrong with that single line. Name both. (Hint: one is about *shape/type*, one is
   about *which tag id 0 actually is*.)

3. `GCV_singleSoft` is the strange sibling. It (a) pastes the entire
   `XLMRobertaForSequenceClassification` class into the script, and (b) sets the target
   to `1.0 / len(label_list)` instead of `1.0`. For each: what was the *point* of doing
   it — what were you trying to achieve or observe? (This is the one genuinely novel idea
   in the family; it dies here, so say it while you can.)

4. All the GCV trainers score with `ThesisMetrics` at a **fixed 0.5 threshold**. For this
   dataset that's not just suboptimal — it structurally guarantees a whole class of tags
   scores ~0 recall no matter how good the model is. Which tags, and what did
   `train.py` replace the fixed threshold with to fix it?

5. Every GCV run wrote to `model_embedding/model_output_*_<timestamp>/`. Connect the dots:
   what *is* the 125 G `model_embedding/` directory you're about to reclaim in Cycle 3,
   in terms of this family? (One sentence — but it's the sentence that makes the weight
   deletion safe to do without ceremony.)

**Reckoning (2026-07-25) — user's answers + resolution. Grade ≈ 4/5.**

*Extra provenance the user supplied (recorded — not in the code):* the **first** GCV
lived in **Google Colab**; hitting issues there, it moved to VS Code (→ the root scripts).
Checkpoint-dir naming evolved **random id → date-stamp → semantic**: **`mn` = multi-label**
(original), **`1n` = "flattened" / single-soft**. `data_split_fi_eng_min5_abs40` = fi+eng
only, `min5` + `abs40` filters. *(Minor flag: `min5` is almost certainly minimum tag
**frequency** ≥5, i.e. drop tags seen <5×, not "≥5 tags per sample" — worth confirming
against `prep_split_dataset.py` before it goes in the thesis; `abs40` = min abstract length.)*

1. **✓.** Nailed it: `GCV_2` **builds the multi-hot matrix itself** ("zeros everywhere
   except one place"); `GCV_3` **relies on an existing set** and "forgot to put the row of
   1s — it should be a matrix." That *is* the bug: `GCV_3` feeds ragged **index lists**
   where BCE needs a fixed-width **multi-hot float vector** (length = |tags|, 1.0 at each
   true tag). BCE compares logits **element-wise** against a same-shape target, so an index
   list has neither the width nor the 0/1 semantics — wrong shape, wrong meaning.

2. **✗ — the one gap; filling it.** `valid_tags = [0]` for a tag-less row is wrong twice:
   (a) **shape/type** — still an index list, not a multi-hot vector (same disease as Q1);
   (b) **id 0 is a *real* tag** (the alphabetically-first tag in `tag2id`), so every empty
   example is silently taught to predict that real tag. The correct target for a tag-less
   row is an **all-zero** vector, not "tag 0". A quiet label-poisoning bug on top of the
   shape bug.

3. **✓ — and this is the confession worth keeping.** (a) The pasted
   `XLMRobertaForSequenceClassification` + `single_label_classification` path came from a
   **supervisor's suggestion to flatten** the problem and drive it through a **single-label**
   classifier instead of multi-label. (b) The `1.0/len(label_list)` soft target was the
   response to **bad/low confidence** under hard 1.0 labels — spread unit probability mass
   across the true tags so CrossEntropy matches a *distribution* rather than a hard class.
   This soft-CE idea is the **one genuinely novel thread** in the family and is **not**
   carried into `train.py` — it dies here, preserved only in this record.

4. **✓ (fix) / partial (which tags).** Correct that `train.py` **searches the best
   threshold per frequency group**, replacing GCV's single fixed 0.5 for all tags. The
   tags a fixed 0.5 silently zeroes out: the **rare / long-tail tags** — an imbalanced model
   seldom pushes their probability past 0.5, so their **recall ≈ 0** regardless of how well
   the representation actually separates them. (This is exactly why `train.py` also adds
   per-class `pos_weight` — same enemy, the long tail.)

5. **✓✓.** "`model_embedding/` are the products of the GCV scripts" — precisely.
   **The safe-deletion sentence:** *the 125 G `model_embedding/` is nothing but the
   checkpoint + `tag_mapping` output of these three GCV runs (`mn`/`1n`), regenerable from
   the scripts + data, holding no idea not already captured here — so it burns without
   ceremony once its provenance is recorded (it is).*

**Verdict:** understanding verified (Q2 was the blind spot — the `[0]` double-fault — now
recorded). `GCV_2 / GCV_singleSoft / GCV_3` cleared for burial; keeper = none (superseded
whole by `boa_strangling/scripts/train.py`). The Cycle-3 weight reclaim is now pre-justified.

