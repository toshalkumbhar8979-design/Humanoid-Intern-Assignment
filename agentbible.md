# Agent bible for this repository

These are the working rules for any agent (human or AI) changing this project. They override convenience.

## 1. What the project is

A pipeline that takes two clips of the same room (before and after a few objects moved, were added or removed),
reconstructs both with MASt3R, aligns them to a floor frame, and reports changes in 3D. The output must be
verifiable, so the rules below are about honesty of results as much as about code.

## 2. Results and honesty

- Never tune a threshold until the pipeline reports something. Tuning to fit the noise is not a result.
  Calibrate against the no-change control, and report the control's false alarms next to every real result.
- A check that did not run is a limitation, never a pass. Say which checks ran and which did not.
- A method that confirms nothing on the real pair is reported as confirming nothing. Do not turn "unconfirmed"
  into "changed" to make the output look fuller.
- Numbers in DESIGN.md and README.md must come from the current code and the current output files. When code
  changes, re-run the affected stage and update the numbers.
- Every new technique is tested on the real pair and on the control. The control is the null: it must not produce changes.

## 3. Writing style for documents

- The docs are written in the first person of the author ("I chose", "I measured").
- Write without second-person address in README.md, DESIGN.md, research.md or this file. Do not name the reader in the prose.
- Call the input files "clips" or "captures" in the docs. File names such as `before.mp4` are fine.
- Keep the status table in DESIGN.md section 8 current. Each item is one of: Fixed, Partly fixed, Open, By design.
  A "Fixed" item needs a test or a measurement named in its note.

## 4. Running and verifying

- Python: `.venv/Scripts/python.exe` on Windows (Git Bash), `.venv/bin/python` elsewhere.
- Set `PYTHONDONTWRITEBYTECODE=1`. MASt3R's submodules otherwise get `__pycache__` folders that show up as dirty.
- Full pipeline: `python scripts/run_all.py`. It takes about 25-30 minutes on the RTX 5050 for the real clips.
  Run it in the background with a log and poll the log. Do not rely on a foreground command surviving a timeout.
- Object stage alone: `scripts/detect_objects.py`, with the real pair and with the control as the null.
  Options: `--detector yolo` (COCO classes) or `--detector yolo-world` (room vocabulary in `ROOM_VOCAB`).
- Tests:
  - `python tests/test_evaluate_changes.py` (scorer, 8 tests)
  - `python tests/test_pipeline_glue.py` (post-reconstruction stages on ground-truth renders; writes `outputs/glue_test`)
  - `python tests/test_change_detection_gt.py` (synthetic ground truth)
- Check exit codes with `echo EXIT=$?` after each command. When a pipeline command pipes into `tail` or `grep`,
  capture the exit status of the real command first (`set -o pipefail`).
- Model progress bars from ultralytics flood the log. Redirect output to a file and inspect the last lines.

## 5. Data and git

- `data/`, `outputs/`, `models/` and `third_party/*/__pycache__` are git-ignored. Do not commit clips, weights or reconstructions.
- Commit only the files changed by the task. Check `git status --short` first. Do not stage whole directories blindly.
- Do not push unless the author asks for it.

## 6. Where things are

| Path | What it does |
|---|---|
| `scripts/run_all.py` | Runs every stage in order, including the control and the object stage with its null |
| `scripts/detect_objects.py` | Object stage: YOLO boxes lifted to 3D, footprint visibility check, `objects.json` |
| `scripts/detect_changes.py` | Voxel stage, with `--null` for the control |
| `scripts/evaluate_changes.py` | Scorer against a ground-truth file |
| `src/registration/align.py` | Floor fit, gravity-prior ICP alignment |
| `src/change_detection/voxel_change.py` | Ray-carved voxel occupancy and regions |
| `DESIGN.md` | Design choices, results, status table |
| `research.md` | Sources read for each technique and what each one still leaves open |
