# What Changed in the Room?

I recorded two phone videos of the same small indoor corner, once before and once after I moved, added and
removed a few objects. This pipeline reconstructs both captures into one 3D frame, aligns them, and reports what
changed. It reports changes in two ways: as 3D regions (voxel carving) and as individual objects (YOLO boxes lifted
to 3D, then matched across captures with a metric displacement).

```
before.mp4 ─► frames ─► MASt3R ─┐
                                ├─► align ─┬─► voxel change detection ─► regions, graphs, bird's-eye map
after.mp4  ─► frames ─► MASt3R ─┘          └─► YOLOv8 + 3D lift ─► objects: stayed / moved / added / removed
                                              (control capture as the null check for both)
```

Design choices, trade-offs and the measured limitations are in [DESIGN.md](DESIGN.md).

## Results on my clips (current code)

| Method | Real pair | No-change control (null) |
|---|---|---|
| Voxel regions (`detect_changes.py`, 5 cm voxels, default tolerance) | 0 changes | 17 false regions, largest 90 voxels |
| Objects (`detect_objects.py`, YOLOv8n, min 3 frames) | 6 stayed, 1 moved, 4 added, 1 removed | 6 stayed, 0 moved, 2 added, 2 removed |

The object result is not yet reliable. The control shows four false added/removed objects on an unchanged scene,
and the one "moved" object (a bed, 0.36 m) also moves 35 cm vertically, which is probably a mismatch. The voxel
result is 0 because real changes are below the reconstruction noise at every voxel size and tolerance I tested
(see DESIGN.md section 3). I have no ground truth for my clips yet.

## Installation

Python 3.11 and an NVIDIA GPU (I tested on an RTX 5050 Laptop, 8 GB).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows (Git Bash): source .venv/Scripts/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
git submodule update --init --recursive    # MASt3R, DUSt3R and CroCo are pinned submodules
```

Model weights: MASt3R (metric ViT-L) and OWLv2 come from Hugging Face on first use. YOLOv8n is stored in
`models/yolov8n.pt` (git-ignored). I downloaded it once and point the script at that file.

## Reproduce

Put the captures in `data/before/before.mp4` and `data/after/after.mp4`, then:

```bash
python scripts/run_all.py                 # full pipeline, with OWLv2 labels
python scripts/run_all.py --no-labeler    # skip the OWLv2 labels (what I ran on my clips)
```

Each stage is also a standalone script in `scripts/`:

| Stage | Script | Output |
|---|---|---|
| 1. Frames + quality filter | `extract_frames.py` | `data/*/frames/`, `manifest.json` |
| 2. Dense reconstruction (MASt3R) | `reconstruct.py` | `outputs/reconstructions/*.npz`, `*.ply` |
| 3. Alignment | `align.py` | `after_aligned.npz`, `alignment/alignment.json` |
| 3b. No-change control | `make_control.py`, `reconstruct.py`, `align.py` | `outputs/control/` |
| 4. Voxel change regions | `detect_changes.py` (`--null` = control) | `change_maps/changes.pkl`, `changes.json` |
| 4b. Object changes | `detect_objects.py` (control as null) | `objects/objects.json`, `objects_null/objects.json` |
| 5. Labels + descriptions | `describe_changes.py` | `descriptions/changes.md` |
| 6. Visualisation | `visualize.py`, `make_graphs.py` | `visualizations/`, `graphs/` |

## Capture protocol I used

1. A small indoor corner (a desk corner), two clips of about 30 s each.
2. Between the clips I moved, added and removed a few objects. I did not measure them.
3. I walked around the same area with good overlap, at 1080p and 30 fps, avoiding backlight.

## Graphs (no ground truth needed)

`run_all.py` writes three graphs to `outputs/graphs/`:

- `top_down_difference.png`: top-down occupancy in the floor frame. Red cells are occupied only before, blue
  only after, grey in both. Coverage differences also show up, so isolated cells are noise.
- `noise_floor.png`: region sizes on the real pair against the no-change control, with the pipeline's threshold.
- `floor_heights.png`: how far surface points sit above the fitted floor, per capture.

## Scoring against ground truth

I did not measure the changes, so I score by type only. Copy `data/real_ground_truth.template.json`, delete
the positions, and list each change I made with its type (moved, added or removed). Then:

```bash
python scripts/evaluate_changes.py outputs/change_maps/changes.json my_truth.json
```

It prints each match, miss and false alarm, then precision and recall. Positions are optional. The
`--frame floor` mode is not checked against ground truth yet.

## Verification status

| Check | Result |
|---|---|
| Glue test (`tests/test_pipeline_glue.py`) on ground-truth renders | PASS, exit 0; recovers 2.4 deg alignment error and all four synthetic changes |
| Scorer unit tests (`tests/test_evaluate_changes.py`) | 8/8 pass, exit 0 |
| Scorer on synthetic detections | 4/4 matched at 0.00-0.11 m |
| Full pipeline on my two clips (`run_all.py --no-labeler`) | exit 0, with the object stage and its null check |
| Graphs (`make_graphs.py`) | written to `outputs/graphs/`; I could not view the images in this environment, so I checked them through their numbers |
| Voxel regions on my clips | 0 changes; the control's largest artefact is at least as big as the real pair's largest candidate at every setting I swept |
| Object detection on my clips | runs (exit 0); real pair and null both produce added/removed objects, see Results above |

## Repository layout

```
scripts/        one CLI per stage, plus run_all.py, evaluate_changes.py and the synthetic generator
src/
  reconstruction/   MASt3R backend
  registration/     floor estimation, gravity-prior ICP alignment
  change_detection/ voxel carving, clustering, moved/added/removed classification
  semantics/        OWLv2 labelling of the best camera crop
  visualization/    bird's-eye PNG and interactive plotly viewer
tests/          glue test, ground-truth check, scorer unit tests
models/         YOLOv8n weights (git-ignored)
demo/           example outputs that are committed
outputs/        generated (git-ignored)
third_party/    MASt3R, DUSt3R and CroCo (submodules)
```

## Hardware and time

- MASt3R at 512 px, capped at 40 frames per capture (`--max-frames`). With 84-89 frames the run ran out of memory.
  At 60 frames the 8 GB GPU filled and global alignment ran about 13 times slower, so I kept 40.
- The full pipeline on my clips took about 25-30 minutes on the RTX 5050, including the object stage.

## Licences

MASt3R and its weights are CC BY-NC-SA 4.0, so this project is for non-commercial use. OWLv2 is Apache-2.0. YOLOv8
(ultralytics) is AGPL-3.0, so the object stage has the same licence restriction as the rest.
