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

---

## Cycle 5 — `data_preparation/` family (7 scripts, the dataset-building arm)

**Dependency check:** no import anywhere in `boa_strangling/`. Three *textual* mentions only:
`boa_strangling/main.py:107` and `:116` print "then run `data_preparation/v3_data_preparation.py`"
(narrative instructions — will dangle after burial, fix in the same commit), and
`align_tampere_schema.py:6` cites `split_abstract_v2.py` in a docstring as the augmentation
precedent (a *reference to the idea*, not to the file — fine to leave). Recovery anchor:
**both** `origin/master` and `a3c516c` (this family predates the cleanup). → safe.

**Quiz (answer in chat; I'll grade + record):**

1. **The one you got right — defend it.** The chain runs
   `stratify_split_v2` (train/val/test) **then** `split_abstract_v2` (one record per abstract).
   Suppose you had run them the other way round — split the abstracts first, then do the
   60/20/20. What exactly goes wrong, and why would your reported test scores have *looked
   better* because of it?

2. **The verified blunder.** `split_abstract_v2.py` contains an explicit line meant to give
   each split sibling its own abstract language. In the actual `data_split_v4` output, all
   383 multi-abstract theses have the *same* language on every sibling — and 380 of them say
   `eng`. Two separate faults, one in each script, combine to cause this. Name both. (Hint:
   one is about *where an assignment sits in a loop*, the other about a **type** — and once you
   see the type, you'll see why the "fix" in `split_abstract_v2` never executed even once.)
   Bonus: which two later scripts exist *because of* this bug?

3. `stratify_split_v2.py` passes `random_state=42` to every `train_test_split` call, and is
   nevertheless **not reproducible** — run it twice, get two different splits. Why? And what
   does that explain about the number of `data_split_v*` directories in this repo?

4. The split is stratified **by faculty**. For a title+abstract → tags task, what is the
   variable you'd actually want stratified, and what does leaving it unstratified do to the
   long tail? Second half: three kinds of entity always end up in **train** regardless of the
   ratios — name them and say what bias that puts in the training set.

5. **The confession.** `field_remover.py` deletes `thesis_title` and `abstract`, and then
   `field_processor.py` turns what remains — faculty, language, year, tags — into one-hot and
   binary vectors. What were you building, what would it have predicted from what, and why do
   you think it was abandoned? (This is the branch that dies here; it's the road-not-taken
   worth one paragraph in the thesis.)

6. **Connect the dots (the safe-deletion sentence).** `data_sew.py` takes the three
   `data_split_v4` files and glues them back into one `full_dataset.json` — it *undoes* the
   split this same family just performed. Why would you deliberately do that, and what is the
   live file `boa_strangling/data/full_dataset.json` in terms of this family?

**Reckoning (2026-07-25) — user's answers + resolution. Grade ≈ 3.5/6, plus three pieces of
provenance that aren't in the code.**

*Extra provenance the user supplied (recorded — not recoverable from the files):*
- **Why the abstract split exists at all:** some theses carried 2+ abstracts, typically one
  **fin** and one **eng**. Splitting was the response to that, not a generic augmentation idea.
- **Why fin+eng only:** the other languages were **under ~10 % of the corpus and "akin noise"**,
  so they were dropped deliberately. (This is the origin of `VALID_LANGUAGES = {"fin","eng"}` at
  `boa_strangling/scripts/prep_split_dataset.py:11` — the filter is a *decision*, not a default.)
- **Why the ratios are 60/20/20:** an **8/1/1 split was tried first and abandoned — not enough
  samples**. That is the reason for the small-group fallback ladder at
  `data_preparation/stratify_split_v2.py:56-68` (n>2 normal, n==2 → train/test only, n==1 → train).

1. **Order / leakage — unanswered; filling it.** The chain is stratify *then* split-abstract, and
   that is the **right** order. Reversed, the fin and eng records of one thesis — **identical
   title, identical tags, identical faculty** — could land on opposite sides of the split. The
   model would then be scored on test rows whose labels it had already memorised from their
   twins in train, and **test scores would rise for a reason that has nothing to do with
   generalisation**. Same-thesis siblings must stay on the same side; this family enforced it.

2. **The `abstract_language` collapse — partially answered, and the user supplied the
   corroborating fact.** The user recalled *"first is usually fin, second eng if two abstracts"*.
   **Verified against `data_split_v4/full_dataset_v3_test.json`: 374 of 383 multi-abstract theses
   are (fin, eng) in index order** — and that is exactly why **380 of 383** are stamped `eng`.
   The two faults, named:
   (a) `data_preparation/v3_data_preparation.py:56-59` — the `abstract_language` assignment sits
   **inside the per-item loop**, so with several abstracts each one overwrites the last; only the
   **final** abstract's language survives, as a **scalar string**;
   (b) `data_preparation/split_abstract_v2.py:59-60` — the per-sibling language fix is guarded by
   `isinstance(entity["abstract_language"], list)`, which is **never true against a string**, so
   the fix is **dead code that never executed once**.
   The user's other half — *"I either used a script to detect language or went back to
   Theseus/Trepo to aggregate it"* — is the **repair**, and it worked: live
   `boa_strangling/data/full_dataset.json` now has distinct sibling languages in **1966 of 1970**
   cases. The two scripts that exist because of this bug: `detect_lang_utils.py` and
   `language_analysis.py` (both at repo root, both still awaiting their own cycle).

3. **Reproducibility — half.** *"Maybe I edited the seed or generated without it?"* — right
   instinct, exact cause: `data_preparation/stratify_split_v2.py:25` calls
   **`random.shuffle(data)` with no seed** before splitting, while every `train_test_split` below
   it passes `random_state=42`. The seed is theatre — the shuffle upstream of it is random every
   run, so the split cannot be regenerated. **A split you cannot reproduce can only be
   *versioned*** — which is a large part of why this repo has `data_split/`, `data_split_v4/`,
   `data_split_v6/`, `split/`, `split_freq3/`.

4. **Stratification — correct intent, one correction.** *"I want to stratify by tags"* — that is
   the right target: for text→tags the punishing distribution is the **tag long tail**, and
   faculty is only a proxy for it. The correction: *"too rare tags get only in train"* is not
   quite the failure mode — with no tag stratification a rare tag lands in **whichever split the
   shuffle happens to put it in**, so it can end up **only in test**, where it is unlearnable and
   scores a guaranteed zero. The three kinds of row that *do* always go to train, by construction:
   **faculty-less entities** (`stratify_split_v2.py:76-79`), **single-member faculties**
   (`:64-68`), and **the first 60 % of every faculty**. Train therefore absorbs every irregular
   case, and val/test are systematically *cleaner* than the data the model will meet in reality.

5. **The dead branch — user's reading accepted over mine.** The user suspects it was
   *"trends analysis / finding dependencies between such fields"*. That fits the evidence better
   than my "classical-features classifier" hypothesis: `field_remover.py:25` deletes
   **`thesis_title` and `abstract`** — the only inputs a tag classifier could use — leaving
   faculty/language/year/tags, which `field_processor.py:22-101` turns into aligned vectors
   (`tags_vector`, `faculty_vector`, `language_vector`, `year`). Vectorising *labels alongside
   metadata* is the shape of a **correlation/association study** ("which faculties co-occur with
   which tags, how does that drift by year"), not of a predictive model — as features for tag
   prediction the `tags_vector` would be pure label leakage. Recorded as: **an exploratory
   metadata-analysis branch, abandoned**; nothing consumes `tempFieldMod/`. (It is also
   unrunnable today — `OneHotEncoder(sparse=False)` was renamed `sparse_output` in sklearn ≥1.2.)

6. **`data_sew.py` — ✓✓, both halves right, and the second is the sharper one.** The user gave
   (a) *"statistics for the full dataset were hard to do per-split, so I glued it back"* and
   (b) *"maybe to run the whole thing through llama"*. Both hold, and **(b) is the principled
   one: the LLM arm is zero-shot — it is never trained, so a train/val/test split is meaningless
   to it.** `llm_generate_tags_sv*` needs *the corpus*, not a split; `data_sew.py:12` reassembles
   exactly that. **The safe-deletion sentence:** *the live
   `boa_strangling/data/full_dataset.json` is the direct descendant of `data_sew.py`'s output —
   the old pipeline's corpus, re-glued so the new pipeline (`prep_split_dataset.py`) could
   re-split it on its own seeded terms; the data survives, so only the scripts die.*

**Verdict:** understanding verified. The blind spots (Q1 leakage rationale, Q3 exact cause,
Q4 always-train bias) are now recorded above; Q2's diagnosis was *confirmed* by the user's own
recollection of fin-then-eng ordering. `data_preparation/` **buried 2026-07-25** (commit
`40f0431`); keeper = none.

---

## Cycle 6 — `data_validation/` family (13 scripts: 6 evaluators, 4 JSON checkers, 3 one-offs)

**Dependency check — THE RITUAL JUST EARNED ITS KEEP.** `tag_evaluation_v4.py` is **not dead**:
`boa_strangling/main.py:303-312` is Step 4 of the live pipeline — it builds the path,
**`sys.exit(1)`s if the file is missing**, and **runs it as a subprocess**. It is never
*imported*, which is exactly why every previous "no imports from the graveyard" check passed
over it. This is the load-bearing corpse the quiz-gate exists to catch. (`main.py:324-337`
similarly runs three `images/` scripts, but those skip gracefully — a problem for that cycle,
not this one.) Everything else in the directory is unreferenced. Recovery anchor: `origin/master`
and `a3c516c`. → safe **except** `tag_evaluation_v4.py`.

**Quiz (answer in chat; I'll grade + record):**

1. **The metric you reported for a year.** Every evaluator computes
   `len(matches) / max(len(original), len(generated))` and calls it **"exact match ratio"**
   (`tag_evaluation.py:6-15`). Two problems. (a) It is not exact match — what would exact match
   actually be, and what is this formula really measuring? (b) Now connect it to your Cycle-1
   confession: you fed the generator `tag_count = len(original tags)`. **What happens to that
   denominator when the model is forced to output exactly as many tags as the gold set — and
   what does the metric silently collapse into?**

2. **The best moment in this family — say why you did it.** `v3` computes its "EMD" as: for
   each gold tag embedding take the **minimum** cosine distance to any generated tag, then
   average (`tag_evaluation_v3.py:45-64`). Nine days after v4, you wrote `v35` replacing that
   with `ot.emd` — a real optimal-transport solve over a cost matrix
   (`tag_evaluation_v35.py:28-47`). What is wrong with the first version — what can it not
   see that transport can? (Hint: think about a model that outputs one excellent tag and nine
   junk ones, versus one that outputs ten decent ones.)

3. **v4 and v3.5 are siblings, not successors.** `v4` (May 9) went for Levenshtein +
   Jaro-Winkler + translate-to-English; `v35` (May 18) went for embeddings + optimal transport.
   Both are attacking the *same* underlying problem — the one that makes plain string equality
   the wrong scorer for this dataset. **Name that problem**, and say which of the two branches
   you think actually addresses it, and why the other is treating a symptom.

4. **The nastiest small thing.** In `v1`/`v2`/`v4` the metric opens with
   `if not original_tags or not generated_tags: return 0.0`. Describe the case where the
   **gold** side is empty, what score that thesis contributes, and why that quietly biases
   every average you ever reported. What should it have done instead?

5. **The bug report that became a feature.** `json_dataset_field_metric.py:82-86` flags
   `"Multiple abstracts found"` as an **anomaly**, and `:75-79` flags any record whose faculty
   count is **≠ 2** as unusual. For each: what was the underlying reality, and which of the two
   turned out to be a genuine defect versus a misunderstanding of your own data? (One of these
   is the exact moment from Cycle 1 Q5, caught in code months before you understood it.)

6. **The forced keeper — your call is needed.** `tag_evaluation_v4.py` cannot be deleted: it is
   Step 4 of `main.py`. But note `v4:444` and `:453` call `input()` and `main.py:312` runs it
   as a non-interactive subprocess. (a) What does that mean Step 4 has actually been computing
   every time it ran? (b) If Step 4 were ported into `boa_strangling/scripts/`, which ideas
   from this family would you carry — v4's string-similarity suite, or something else?

**Reckoning:** _awaiting your answers._

