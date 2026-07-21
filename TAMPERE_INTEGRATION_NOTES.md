# Tampere Integration — Prior Techniques & Known Gaps

> Written 2026-07-04. Purpose: before writing any Tampere alignment code, this
> records what already exists (JYU-side precedent) vs what's actually missing,
> so work doesn't re-derive facts or silently pick an inconsistent convention.
> Source of truth is the code itself — if this doc and the code disagree, the code wins.

## Status of the harvest itself

`boa_strangling/tampere_thesis_harvest.py` has already run to completion:
- `boa_strangling/data/tampere_theses.jsonl` — 66,340 lines, one JSON object per line.
- `boa_strangling/data/tampere_harvest_state.json` — `{"count": 66340, "complete": true}`.
- Source: OAI-PMH (`trepo-oai`), not the Finna fallback (at least for the sampled record).

**Nothing downstream of the harvest exists yet.** `boa_strangling/main.py` never calls
`tampere_thesis_harvest.py` and no script in the repo reads `tampere_theses.jsonl`.
Everything below is "pattern to follow", not "code to reuse as-is" — field names and
data shapes differ from JYU in ways that require an explicit mapping layer.

## Tampere raw schema (per-record keys in `tampere_theses.jsonl`)

```
source, id, record_url, title, abstract, abstracts, tags, year,
faculties, type, language_work, language_abstract
```

- `abstracts` is a list of `{"text": ..., "lang": ...}`. `parse_oai_record()` sorts by
  text length descending and picks the **longest** as the single `abstract` /
  `language_abstract` — already resolves the "which abstract if there are several"
  question, and does it better than JYU's approach (see below).
- `faculties` (plural) is a flat list of strings pulled from OAI `setSpec`→`setName`,
  filtered by regex `facult|tiedekun|yksikk|school|unit`, falling back to keeping
  all set names if none match. JYU's equivalent field is singular `faculty`.
- `tags` is a single flat list — Tampere does **not** distinguish a controlled
  vocabulary (YSO) from free-text tags the way JYU's `dc.subject.yso` vs
  `dc.subject.other` split does. This is a real schema gap, not a naming one.
- `language_work` / `language_abstract` are 2-letter ISO 639-1 (`fi`, `en`, `sv`).

## JYU target schema (traced end-to-end)

Raw DSpace → `data_preparation/v3_data_preparation.py` (`field_name_mapping`):

```
dc.contributor.tiedekunta → faculty
dc.subject.yso            → subject_tags
dc.subject.other          → additional_tags
dc.title                  → thesis_title
dc.description.abstract   → abstract
dc.language.iso           → language
dc.date.issued            → date_issued
dc.identifier.uri         → identifier
```

Every field is a **list**, even single-valued ones (JYU/DSpace multivalue convention)
— confirmed in `zdepricated_versions/prepared_datasets/full_details.json`.

Two competing *final* training shapes exist downstream in `boa_strangling/scripts/`:

```
prep_split_dataset.py (older):  {identifier, title, abstract, language, combined_tags}
generate_variants.py (current): {id, text, tags, lang}   ← text = title+abstract, optional "<LANG=xx> " prefix
```

`generate_variants.py`'s shape is the one actually driving the datasets currently
used by `boa_strangling/scripts/train.py` (`data_f15_abs100_*`, `data_f20_abs100_*`).
**Use `tags` (not `combined_tags`) as the field name for any new alignment work** —
`combined_tags` is the historical name still referenced in `PROJECT_REVISION.md` and
older scripts, but it's not what the live pipeline consumes.

## Tag handling precedent

- `tags` / `combined_tags` vocab + count metadata convention: both
  `prep_split_dataset.py` and `generate_variants.py` write a `tag_vocab.json`
  (sorted vocab list) and a `metadata.json` containing `"unique_valid_tags": N`
  after filtering to tags with frequency ≥ `MIN_TAG_FREQ` (currently 5). Reuse this
  convention for a `tampere_tag_vocab.json` + metadata so the two are directly comparable.
- Raw (pre-filter) per-source tag counts: `boa_strangling/scripts/generate_tag_count.py`
  + `sort_tag_counts.py` → `boa_strangling/stats/tags/{subject,additional}_tag_counts*.json`.
  These operate on JYU's already-split `subject_tags`/`additional_tags`; Tampere has
  no such split, so a Tampere equivalent can only produce one merged tag-count file
  unless/until the controlled-vocab-vs-free-text gap above gets a resolution.
- **No cross-source tag-space comparison exists anywhere in the repo.** Everything in
  `data_validation/tag_evaluation_v4.py` (`exact_match_ratio`, `fuzzy_match_ratio`,
  `jaro_winkler_match_ratio`, `embedding_similarity` via
  `paraphrase-multilingual-mpnet-base-v2`, cosine sim, 0.75 threshold) compares
  **generated tags vs ground truth on the same record**, not vocab-vs-vocab across
  universities. `embedding_similarity()`'s pairwise-cosine machinery is the most
  directly reusable piece for a JYU-vocab vs Tampere-vocab similarity script — the
  matching logic just needs to run over two vocab lists instead of one record's
  predicted/true tags.
- Orphaned artifacts `zdepricated_versions/embedding_A_T*.npy/.pt` look like they may
  have been an earlier attempt at this exact cross-source comparison, but no
  generating/consuming code survives anywhere in the repo — don't assume anything
  about their contents; treat as unrecoverable.

## Faculty handling precedent

Four independent, mutually inconsistent, JYU-only hardcoded dictionaries exist:
`data_preparation/field_processor.py` (`FACULTY_DICT`, 9 entries),
`data_validation/analyze_faculty_field_v2.py` (`faculty_mapping`, 20 entries),
`faculty_data_processor.py` (`canonical_faculties`, 10 entries, plus its own
language-detection + fuzzy EN/FI pairing logic), and the static seed list
`zdepricated_versions/prepared_datasets/unique_faculty_pairs.json` (11 pairs, has a
double-space data-quality wart in one entry: `"Yhteiskuntatieteellinen  tiedekunta"`).

**None of these will match anything from Tampere.** Tampere faculty names come from
OAI `setName` and will be structurally different strings. A new Tampere-specific
dictionary needs to be built from scratch — treat the JYU dictionaries only as a
"this is the pattern: raw-name → canonical-English-label dict" reference.

**Field name mismatch:** JYU pipeline field is singular `faculty`; Tampere harvester
field is plural `faculties`. Pick one name for the aligned schema and rename explicitly.

## Language handling precedent

Two incompatible code formats coexist in the repo:
- 2-letter ISO 639-1 (`fi`, `en`, `sv`) — Tampere's native format, and what
  `detect_lang_utils.py`'s detectors (`langdetect`, `lingua`) emit.
- 3-letter ISO 639-3/639-2B (`fin`, `eng`, `swe`) — JYU pipeline's format, via
  `data_preparation/v3_data_preparation.py`'s `standardize_language_code()` and
  `language_analysis.py`'s `ISO_639_1_TO_3` dict. Both map `fi→fin`, `en→eng`, `sv→swe`.

**Only Finnish + English actually reach training data.** `VALID_LANGUAGES`/hardcoded
`{"eng","fin"}` in both `prep_split_dataset.py` and `generate_variants.py` drop
Swedish at the filter step — Swedish *is* detected and kept in the raw/prepared data,
just excluded from what the model trains on.

**Decision (2026-07-04):** keep 3-letter ISO codes as the target format throughout
(`fin`/`eng`/`swe`, matching JYU's existing convention), but for now **only fin+eng
are the primary training set**, same as the current JYU filter — Swedish stays a
future addition, not a blocker for this round. Scale reference from Tampere's Finna
metadata (whole-repository, not yet filtered to theses only): fin ≈170k, eng ≈140k,
swe+ger ≈14k combined, all other languages ≈4k each. Swedish is a small enough slice
that deferring it costs little; revisit once fin+eng alignment is working end-to-end.

**Dual-language abstracts — JYU vs Tampere handle this differently, JYU's is intentional:**
- JYU: `data_preparation/split_abstract_v2.py` splits one entry-with-N-abstracts into
  N separate output records (duplicating title/tags/faculty), each tagged with a
  synthetic id + `original_identifier` back-pointer. **This is a deliberate design
  choice, not a bug**: the core task is text→tags, and the abstract (long,
  information-dense) is the dominant training signal — splitting multi-language
  abstracts into separate records is intentional data augmentation, giving more
  (text, tags) training pairs from the same tag set rather than discarding one
  language's abstract. Do carry the caveat that this means raw language-distribution
  *counts* (e.g. `images/misleading_dataset_lang_stat.py`) are counting
  abstract-instances, not unique theses — fine for training-pair purposes, just don't
  misread it as "N distinct theses were in language X."
- Tampere: currently keeps all abstracts in one record (`abstracts: [{"text","lang"}, ...]`)
  and the harvester picks the single longest one as primary `abstract`. To match the
  JYU augmentation strategy (and not lose training pairs), Tampere alignment should
  likely **split multi-language `abstracts` into separate records the same way**
  `split_abstract_v2.py` does, rather than keeping only the single longest abstract —
  otherwise Tampere loses out on the extra training pairs JYU intentionally gets.

## Stats/plotting precedent — two implementations disagree, pick one

- `images/visualization_distro.py` (older, plots): character length via `len()`,
  counted over **every** element of the `thesis_title`/`abstract` lists.
- `boa_strangling/scripts/utils/text_stats.py` (current, no plots): word+char length,
  but only over the **first** list element (`entry["thesis_title"][0]`). Feeds
  `boa_strangling/scripts/compute_stats.py` → `boa_strangling/stats/stats_summary.json`.
  Also computes `abstract_language_distribution` via `Counter`.

These two give genuinely different numbers on the same data (all-elements vs
first-element). Tampere's fields are plain strings, not lists — pick
`text_stats.py`'s shape as the target (it's the one still actively wired into
`main.py`) and adapt it to take a string directly, so Tampere and JYU stats are
comparable apples-to-apples rather than differing for code-path reasons.

## Ground-truth numbers from a full streaming pass (2026-07-04)

Ran `boa_strangling/scripts/audit_tampere_raw.py` over all 66,340 records:

- **Multi-abstract records: 7,296 (11%)**, not the ~8 originally assumed — 7,238
  with 2 abstracts, 55 with 3, 2 with 4, 1 with 5. The split-per-language plan
  (see above) is unchanged, just at a much larger scale than expected.
- **19,848 records (30%) have zero abstract at all** (`abstracts: []`, and the
  top-level `abstract` field empty to match). **Decision: dropped at the
  alignment step** — no abstract means no usable (text→tags) pair given the
  abstract-is-the-dominant-signal framing.
- **1,192 records (1.8%) have no tags.** Not yet decided whether to drop —
  kept in the aligned output for now with a flag, revisit at the training-filter step.
- **`faculties` field contains no faculty data.** Verified directly against
  Trepo's live OAI-PMH `ListSets`: the repo has exactly one top-level community
  (`Trepo`) and 14 collections, all organized by **document type + access level**
  (Bachelor's/Master's/Doctoral/Articles/Monographs, each with a restricted-access
  variant) — no faculty-level community/collection exists in the set hierarchy at
  all. So `faculties` = `['Trepo', <degree-type string, duplicate of the `type` field>]`
  for every one of the 66,340 records — not a harvester bug, the OAI *sets*
  genuinely don't carry faculty.
- **Faculty data does exist, just elsewhere**: confirmed via `GetRecord` on
  several ids that `dc:contributor` carries it directly, e.g.
  `"Informaatiotieteiden tiedekunta - Faculty of Information Sciences"` (faculty),
  `"...laitos - Department of ..."` (department), sometimes
  `"...yksikkö - School of ..."` (post-reorg unit, no separate faculty in newer
  records), always ending with `"University of Tampere"`. The harvester's existing
  regex (`facult|tiedekun|yksikk|school|unit`) would work correctly here — it's
  just never been pointed at `dc:contributor`, only at OAI set names.
  **Deferred (2026-07-04):** fixing the harvester to parse `dc:contributor` and
  re-running the harvest (~20 min, same as the original run — raw XML wasn't
  persisted so this can't be recovered from the existing jsonl without re-querying
  Trepo) is a known, scoped follow-up, not done as part of this round. Schema
  alignment proceeds without a real faculty field for now; `type`
  (bachelor/master/doctoral/etc., cleanly available already) is used as
  `degree_type` instead, since it's genuinely reliable, unlike `faculties`.

## Schema alignment — done for title/abstract/tags (2026-07-04)

Scripts, in run order:
- `boa_strangling/scripts/audit_tampere_raw.py` — read-only streaming audit of the
  raw file; produced the ground-truth numbers above.
- `boa_strangling/scripts/align_tampere_schema.py` — reads
  `boa_strangling/data/tampere_theses.jsonl`, writes
  `boa_strangling/data/tampere_aligned.jsonl` (53,850 entities) +
  `boa_strangling/data/tampere_alignment_report.json` (counts/distributions).
  One entity per abstract (records with 2+ abstracts split, matching JYU's
  intentional augmentation strategy); records with zero abstracts dropped.
  ID scheme: `id = "{original_id}::abstract{idx}"` when split, else the raw
  Trepo OAI id unchanged; `original_id` always present as the back-pointer.
  Simpler than JYU's `original_identifier` + random `uuid8` suffix, since Trepo's
  OAI ids (`oai:trepo.tuni.fi:10024/NNNNN`) are already globally unique — a
  deterministic index suffix is enough, no need for a random component.
  Language converted to 3-letter per-abstract (falls back to
  `language_abstract`/`language_work` if the individual abstract's own `lang` is
  missing). `degree_type` (bachelor/master/doctoral/other) derived from the
  existing `type` field, standing in for faculty until §1.6 is done.
- `boa_strangling/scripts/compute_tampere_field_stats.py` — title/abstract
  char+word length min/max/avg on the aligned (post-split) entities, written to
  `boa_strangling/stats/tampere_field_stats.json`. Abstracts average ~2678 chars
  / ~303 words; titles ~88 chars / ~9 words.
- `boa_strangling/scripts/tampere_tag_stats.py` — full tag dictionary
  (`boa_strangling/stats/tags/tampere_tag_counts.json`, tag→count, 68,302 unique
  tags across 255,943 occurrences) + metadata
  (`boa_strangling/stats/tags/tampere_tag_metadata.json`) + long-tail plot
  (`boa_strangling/stats/tampere_tag_long_tail_distribution.png`).
  **Finding**: the highest-frequency tags are degree-**programme** names (e.g.
  `"Tieto- ja sähköteknikan kandidaattiohjelma - Bachelor's Programme in Computing
  and Electrical Engineering"`, 864 occurrences), not topical keywords — genuine
  topic tags (`"machine learning"`, `"varhaiskasvatus"`) appear further down the
  distribution. `dc.subject` at Tampere bundles two semantically different tag
  kinds into one flat list; this is a data-quality nuance for later tag
  normalization work, not something resolved yet.

## Open decisions before writing alignment code

1. Field name for tags in the aligned schema: `tags` (recommended, matches live
   pipeline) vs `combined_tags` (matches older docs/scripts). Pick one, don't mix.
2. Whether Tampere tags get treated as one bucket (no YSO/free-text split — the
   honest answer, since Tampere doesn't provide that distinction) or forced into
   the two-bucket shape some validation scripts still expect.
3. Faculty field name: `faculty` (JYU) vs `faculties` (Tampere) — pick one, and
   whether to add a separate `university`/`source` field instead of trying to
   force both universities' faculties into one shared taxonomy.
4. Swedish: exclude (to match current JYU-derived training filter) or include
   (Tampere already has the data, and dropping it costs no extra work — just a
   convention choice) — needs a conscious decision either way.
5. No merge-order/precedent exists for combining vocabularies from two sources —
   the tag-space similarity step (see above) should run *before* deciding whether
   and how to merge, since low overlap might argue for keeping tag spaces separate
   per source rather than unioning them.
