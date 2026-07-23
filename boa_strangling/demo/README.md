# ScholarScribe tag-prediction demo

A small Gradio app: type (or pick) a title + abstract, get predicted subject
tags from the trained model. Runs standalone - no access to the training
datasets or the full ~3.2GB-per-checkpoint training state needed, just the
`package/` folder next to `app.py`.

Two models can be loaded side by side:
- **best** = `checkpoint-145404` (the actual final checkpoint of the
  2026-07-19 run)
- **comparison** = `checkpoint-117000` (an earlier checkpoint from the same
  run that a training-time validation metric briefly favored, but which
  later analysis showed is behind on every metric once thresholds are
  compared fairly)

See `THESIS_LOG.md` in the main repo, 2026-07-21 entries, for the full
analysis behind this choice and behind the two threshold modes in the UI.

## Getting the weights onto another machine

The full package (both checkpoints) is ~1.15GB; a single checkpoint is
~575MB. Both are too large for a plain "download file" click over a slow or
flaky connection (that works fine for small files like a single `.md`, not
for hundreds of MB) - use one of these instead, from a terminal on the
*receiving* machine (not this one):

**Full bundle (recommended - both checkpoints, ~1.1GB):**
Packaged at `boa_strangling/demo/bundles/scholarscribe_demo_full.tar` on this
machine. Contains `app.py`, `requirements.txt`, and everything under
`package/`, including both `weights/best/` and `weights/comparison/` - the
compare view works immediately after extracting, no second download needed.

- **WinSCP (Windows, recommended)**: most reliable for large files over an
  unreliable connection - resumes automatically if the connection drops.
  Point it at the same host/user/key you already use for VS Code Remote-SSH
  (check `%USERPROFILE%\.ssh\config`, or import directly from there), then
  drag `boa_strangling/demo/bundles/scholarscribe_demo_full.tar` to a
  local folder.
- **scp (Windows 10 1809+ has this built into PowerShell, no install needed)**:
  ```
  scp <your-ssh-host-alias>:/home/sharovv/ScholarScribe/boa_strangling/demo/bundles/scholarscribe_demo_full.tar C:\Users\<you>\Downloads\
  ```
  No resume - if it drops partway, just rerun.
- **rsync (if you have WSL/Git Bash with rsync)** - best for a flaky
  connection, skips bytes already transferred on retry:
  ```
  rsync -avP <your-ssh-host-alias>:/home/sharovv/ScholarScribe/boa_strangling/demo/bundles/scholarscribe_demo_full.tar ./
  ```

Then on Windows, extract it (Windows 10 1803+ has `tar.exe` built into
PowerShell/CMD, or use 7-Zip):
```
tar -xf scholarscribe_demo_full.tar
```
This recreates `app.py`, `requirements.txt`, and `package/` (both checkpoints)
in the current folder.

**Smaller download, `best`-only (~549MB)**: if 1.1GB is too much for your
connection, `boa_strangling/demo/bundles/scholarscribe_demo_best_only.tar`
is still available - same steps as above, just without the compare view
until you separately fetch `boa_strangling/demo/package/weights/comparison/`
(another ~570MB, WinSCP/scp/rsync a directory instead of a tar) and drop it
into `package/weights/comparison/` next to `best`.

## Running

```
pip install -r requirements.txt
python app.py
```
Opens a local Gradio server (default `http://127.0.0.1:7860`) - open that URL
in a browser. Works on CPU (weights are stored fp16 for a smaller download
but auto-upcast to fp32 on CPU load, since fp16 matmul on CPU-only PyTorch is
unreliable/slow); uses CUDA automatically if available.

## Regenerating the package (on the training machine, not needed on Windows)

`boa_strangling/scripts/build_demo_package.py` rebuilds everything under
`package/` from the run's checkpoints - fp16 conversion, tokenizer copy, and
threshold/bucket metadata read from the already-computed honest evaluation
outputs under `boa_strangling/results/.../predictions/`. `package/` itself is
gitignored (binary, too large for normal git) - only `app.py`,
`requirements.txt`, this README, and `curated_examples.json` are tracked.

`curated_examples.json` (copied verbatim to `package/examples.json` by the
build script) is hand-picked, not regenerated: each entry is chosen to show
something specific - a case where `checkpoint-117000` beats the final
checkpoint, a long-tail-heavy thesis, a clean failure (both checkpoints score
zero), a micro-vs-macro-F1 illustration, and two self-authored synthetic
texts with no real ground truth. Edit that file directly rather than
re-running the selection logic if you want to change the examples.
