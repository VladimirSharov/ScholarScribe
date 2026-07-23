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

