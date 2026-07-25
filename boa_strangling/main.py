#!/usr/bin/env python3
"""
boa_strangling/main.py  —  Pipeline entry point.

Run from the project root:
  python boa_strangling/main.py              # resume from where you left off
  python boa_strangling/main.py --step 2    # run only step 2
  python boa_strangling/main.py --from 1    # run step 1 and everything after
  python boa_strangling/main.py --status    # show progress without running anything

Steps
-----
  0     Data collection   (manual — see note below)
  1     Dataset stats     compute_stats, generate_tag_count, sort_tag_counts
  2     Preprocess/split  generate_variants.py  →  boa_strangling/data/<variant>/
  2.5   Coverage check    verify no unseen tags in val/test before spending GPU time
  3     Model training    scripts/train.py
  4     Evaluation        data_validation/tag_evaluation_v4.py
  5     Plots             images/visualization_distro.py etc.

State is saved to boa_strangling/pipeline_state.json after each completed step,
so you can close the terminal and resume later without repeating work.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # project root
BOA  = Path(__file__).resolve().parent          # boa_strangling/
STATE_FILE = BOA / "pipeline_state.json"

STEP_ORDER = [0, 1, 2, 2.5, 3, 4, 5]

STEP_NAMES = {
    0:   "Data collection (manual step)",
    1:   "Dataset statistics",
    2:   "Preprocess & split",
    2.5: "Coverage check",
    3:   "Model training",
    4:   "Evaluation",
    5:   "Plots",
}


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {
        "last_completed_step": None,
        "data_dir": None,
        "model_dir": None,
        "steps": {}
    }


def save_state(state: dict):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def mark_done(state: dict, step: float, **extra):
    key = str(step)
    state["steps"][key] = {
        "status": "done",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **extra
    }
    state["last_completed_step"] = step
    save_state(state)


def mark_skipped(state: dict, step: float, note: str):
    key = str(step)
    state["steps"][key] = {"status": "skipped", "note": note}
    state["last_completed_step"] = step
    save_state(state)


# ---------------------------------------------------------------------------
# Subprocess helper
# ---------------------------------------------------------------------------

def run(cmd: list[str], cwd: Path = ROOT) -> int:
    """Run a subprocess, stream its output, return exit code."""
    result = subprocess.run(cmd, cwd=str(cwd))
    return result.returncode


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step_0(state: dict):
    """
    Data collection is a long-running API job — run it separately.
    Expected output: data_collection/output.json
    The full_dataset.json in boa_strangling/data/ is an INHERITED artifact, not
    regenerated here: it was produced by the old data_preparation/ family
    (v3_data_preparation -> stratify_split_v2 -> split_abstract_v2 -> data_sew),
    which is buried. See ANNIHILATION_RECORD.md "Family: data_preparation/".
    """
    raw_out  = ROOT / "data_collection" / "output.json"
    full_out = BOA / "data" / "full_dataset.json"

    print("\n[Step 0] Data collection")
    print("  This step is MANUAL — the API job takes hours.")
    print(f"  Collector:   data_collection/api_data_collector.py")
    print(f"  Raw output:  {raw_out.relative_to(ROOT)}")
    print(f"  Prep chain:  data_preparation/ (buried — see ANNIHILATION_RECORD.md)")
    print(f"  Final input: {full_out.relative_to(ROOT)}")

    if full_out.exists():
        size_mb = full_out.stat().st_size / 1e6
        print(f"\n  full_dataset.json already present ({size_mb:.1f} MB) — marking done.")
        mark_done(state, 0, full_dataset=str(full_out.relative_to(ROOT)))
    else:
        print("\n  full_dataset.json not found. Run the collector first, then re-run this script.")
        mark_skipped(state, 0, "full_dataset.json missing — run data_collection manually")
        sys.exit(1)


def step_1(state: dict):
    """Compute dataset statistics and tag counts."""
    print("\n[Step 1] Dataset statistics")

    # compute_stats.py uses 'from utils.io import ...' — must run from scripts/
    scripts_dir = BOA / "scripts"

    print("  → compute_stats.py")
    rc = run([sys.executable, "compute_stats.py"], cwd=scripts_dir)
    if rc != 0:
        print("  ERROR: compute_stats.py failed.")
        sys.exit(rc)

    print("  → generate_tag_count.py")
    rc = run([sys.executable, str(BOA / "scripts" / "generate_tag_count.py")])
    if rc != 0:
        print("  ERROR: generate_tag_count.py failed.")
        sys.exit(rc)

    print("  → sort_tag_counts.py")
    rc = run([sys.executable, str(BOA / "scripts" / "sort_tag_counts.py")])
    if rc != 0:
        print("  ERROR: sort_tag_counts.py failed.")
        sys.exit(rc)

    print(f"  Stats written to: boa_strangling/stats/")
    mark_done(state, 1, outputs=["boa_strangling/stats/stats_summary.json",
                                  "boa_strangling/stats/tags/"])


def step_2(state: dict):
    """
    Filter, prepare, and split the dataset.
    Edit OPTIONS in boa_strangling/scripts/generate_variants.py before running.
    The output directory name is auto-generated from those options.
    """
    print("\n[Step 2] Preprocess & split")
    print("  Using: boa_strangling/scripts/generate_variants.py")
    print("  Edit the OPTIONS dict in that file first if you want different filters.")
    print()

    rc = run([sys.executable, str(BOA / "scripts" / "generate_variants.py")])
    if rc != 0:
        print("  ERROR: generate_variants.py failed.")
        sys.exit(rc)

    # Find the most recently created data_* directory to record it
    data_dirs = sorted(
        [d for d in (BOA / "data").iterdir() if d.is_dir() and d.name.startswith("data_")],
        key=lambda d: d.stat().st_mtime,
        reverse=True
    )
    if not data_dirs:
        print("  ERROR: no output directory found under boa_strangling/data/")
        sys.exit(1)

    latest = data_dirs[0]
    state["data_dir"] = str(latest.relative_to(ROOT))
    print(f"  Active data_dir set to: {state['data_dir']}")
    mark_done(state, 2, data_dir=state["data_dir"])


def step_2_5(state: dict):
    """
    Verify label coverage: no tag in val/test should be absent from train.
    This runs before training so you don't waste GPU time on a bad split.
    """
    print("\n[Step 2.5] Coverage check")

    data_dir = state.get("data_dir")
    if not data_dir:
        print("  ERROR: data_dir not set — run step 2 first.")
        sys.exit(1)

    # check_label_coverage.py has a hardcoded path; patch it on the fly via env or just run it
    # It reads combined_tags; generate_variants.py produces 'tags' instead — note below
    split_dir = ROOT / data_dir

    # Inline coverage check so it works regardless of which script produced the split
    ok = True
    results = {}
    for split_name in ("val", "test"):
        split_file = split_dir / f"{split_name}.json"
        train_file = split_dir / "train.json"

        if not split_file.exists() or not train_file.exists():
            print(f"  ERROR: missing {split_file} or {train_file}")
            sys.exit(1)

        train_data = json.loads(train_file.read_text(encoding="utf-8"))
        split_data = json.loads(split_file.read_text(encoding="utf-8"))

        # Support both field names: 'tags' (generate_variants) and 'combined_tags' (prep_split)
        def get_tags(entry):
            return entry.get("tags") or entry.get("combined_tags") or []

        train_tags = {t for e in train_data for t in get_tags(e)}
        split_tags = {t for e in split_data for t in get_tags(e)}
        unseen = split_tags - train_tags
        pct = 100 * len(unseen) / len(split_tags) if split_tags else 0

        results[split_name] = {
            "total_tags": len(split_tags),
            "unseen_in_train": len(unseen),
            "percent_unseen": round(pct, 2)
        }

        print(f"  {split_name}: {len(unseen)}/{len(split_tags)} tags unseen in train ({pct:.2f}%)")
        if unseen:
            ok = False
            print(f"    Sample unseen: {list(unseen)[:5]}")

    if ok:
        print("  Coverage OK — all val/test tags appear in train.")
    else:
        print("  WARNING: unseen tags found. Model will never predict these correctly.")
        print("  Consider lowering min_tag_freq or re-running generate_variants.py.")

    mark_done(state, 2.5, coverage_ok=ok, coverage=results)


def step_3(state: dict):
    """Train the model. Uses boa_strangling/scripts/train.py."""
    print("\n[Step 3] Model training")

    data_dir = state.get("data_dir")
    if not data_dir:
        print("  ERROR: data_dir not set — run step 2 first.")
        sys.exit(1)

    print(f"  Data dir: {data_dir}")
    print("  Script:   boa_strangling/scripts/train.py")
    print()

    # train.py has DATA_DIR hardcoded — remind the user to check it
    train_script = BOA / "scripts" / "train.py"
    script_text = train_script.read_text(encoding="utf-8")
    if data_dir not in script_text:
        print(f"  NOTE: train.py DATA_DIR does not match the active data_dir.")
        print(f"    Active: {data_dir}")
        print(f"    Open boa_strangling/scripts/train.py and update DATA_DIR before continuing.")
        resp = input("  Continue anyway? [y/N] ").strip().lower()
        if resp != "y":
            print("  Aborted. Update train.py and re-run --step 3.")
            sys.exit(0)

    rc = run([sys.executable, str(train_script)])
    if rc != 0:
        print("  ERROR: training failed.")
        sys.exit(rc)

    # Find the most recently created result directory
    results_dirs = sorted(
        [d for d in (BOA / "results").iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True
    )
    if results_dirs:
        state["model_dir"] = str(results_dirs[0].relative_to(ROOT))
        print(f"  Model saved to: {state['model_dir']}")

    mark_done(state, 3, model_dir=state.get("model_dir"))


def step_4(state: dict):
    """Run evaluation against the trained model."""
    print("\n[Step 4] Evaluation")

    model_dir = state.get("model_dir")
    if not model_dir:
        print("  WARNING: model_dir not recorded in state. Proceeding anyway.")

    eval_script = ROOT / "data_validation" / "tag_evaluation_v4.py"
    if not eval_script.exists():
        print(f"  ERROR: {eval_script} not found.")
        sys.exit(1)

    print(f"  Script: data_validation/tag_evaluation_v4.py")
    print("  NOTE: tag_evaluation_v4.py requires: langdetect, Levenshtein, jellyfish, sentence_transformers")
    print()

    rc = run([sys.executable, str(eval_script)])
    if rc != 0:
        print("  ERROR: evaluation failed.")
        sys.exit(rc)

    mark_done(state, 4)


def step_5(state: dict):
    """Produce investigation plots and visualizations."""
    print("\n[Step 5] Plots & investigation")

    scripts = [
        ROOT / "images" / "visualization_distro.py",
        ROOT / "images" / "plot_faculty_summary_v2.py",
        ROOT / "images" / "visualization_conf_lang.py",
    ]

    for script in scripts:
        if not script.exists():
            print(f"  SKIP (not found): {script.relative_to(ROOT)}")
            continue
        print(f"  → {script.relative_to(ROOT)}")
        rc = run([sys.executable, str(script)])
        if rc != 0:
            print(f"  WARNING: {script.name} exited with code {rc} — continuing.")

    mark_done(state, 5)


# ---------------------------------------------------------------------------
# Step dispatch
# ---------------------------------------------------------------------------

STEP_FNS = {
    0:   step_0,
    1:   step_1,
    2:   step_2,
    2.5: step_2_5,
    3:   step_3,
    4:   step_4,
    5:   step_5,
}


def print_status(state: dict):
    last = state.get("last_completed_step")
    print("\n=== Pipeline state ===")
    print(f"  State file:  {STATE_FILE.relative_to(ROOT)}")
    print(f"  Last done:   step {last}" if last is not None else "  Last done:   (nothing yet)")
    if state.get("data_dir"):
        print(f"  Data dir:    {state['data_dir']}")
    if state.get("model_dir"):
        print(f"  Model dir:   {state['model_dir']}")
    print()
    for step in STEP_ORDER:
        key = str(step)
        info = state["steps"].get(key, {})
        status = info.get("status", "pending")
        label = STEP_NAMES[step]
        ts = info.get("timestamp", "")
        marker = {"done": "✓", "skipped": "~", "pending": " "}.get(status, "?")
        ts_str = f"  [{ts}]" if ts else ""
        print(f"  {marker} step {step:<4} {label}{ts_str}")
    print()


def next_step_after(last) -> float | None:
    """Return the first step in STEP_ORDER that comes after `last`."""
    if last is None:
        return STEP_ORDER[0]
    for i, s in enumerate(STEP_ORDER):
        if s == last and i + 1 < len(STEP_ORDER):
            return STEP_ORDER[i + 1]
    return None


def run_steps(steps_to_run: list[float], state: dict):
    for step in steps_to_run:
        fn = STEP_FNS.get(step)
        if fn is None:
            print(f"Unknown step: {step}")
            sys.exit(1)
        fn(state)
        print(f"  [step {step} complete]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ScholarScribe boa_strangling pipeline runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--step",   type=float, metavar="N", help="Run only step N")
    group.add_argument("--from",   type=float, metavar="N", dest="from_step",
                       help="Run from step N to the end")
    group.add_argument("--status", action="store_true", help="Show state and exit")
    args = parser.parse_args()

    state = load_state()

    if args.status:
        print_status(state)
        return

    if args.step is not None:
        if args.step not in STEP_FNS:
            print(f"Unknown step: {args.step}. Valid: {STEP_ORDER}")
            sys.exit(1)
        print_status(state)
        run_steps([args.step], state)
        return

    if args.from_step is not None:
        if args.from_step not in STEP_ORDER:
            print(f"Unknown step: {args.from_step}. Valid: {STEP_ORDER}")
            sys.exit(1)
        idx = STEP_ORDER.index(args.from_step)
        steps = STEP_ORDER[idx:]
        print_status(state)
        run_steps(steps, state)
        return

    # Default: resume from where we left off
    last = state.get("last_completed_step")
    nxt = next_step_after(last)

    print_status(state)

    if nxt is None:
        print("All steps completed. Use --step N or --from N to re-run.")
        return

    print(f"Resuming from step {nxt}: {STEP_NAMES[nxt]}")
    run_steps(STEP_ORDER[STEP_ORDER.index(nxt):], state)


if __name__ == "__main__":
    main()
