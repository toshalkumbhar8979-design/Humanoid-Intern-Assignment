# Design note

## 1. Shared 3D representation: MASt3R, not COLMAP + Gaussian Splatting

I chose MASt3R (metric checkpoint) for four reasons:

- It predicts dense pointmaps and camera poses from unordered phone frames. It needs no feature matching or
  bundle adjustment to get a usable reconstruction, and it works on textureless surfaces such as desks and walls.
- Each frame has a confident pointmap and a camera pose in one shared world frame, which every detector below needs.
- It runs in seconds per sequence on a laptop GPU, so anyone can rerun it.

COLMAP + 3DGS would look better, but photorealism does not improve change detection, so I left it as an
optional path for visualisation.

Trade-off: MASt3R's weights are CC BY-NC-SA, so the project is non-commercial.

Memory: a 60-frame reconstruction fits in the 8 GB of the RTX 5050. I sampled `nvidia-smi` every 3 s during the
run: the peak was 7,896 MiB of 8,151 MiB, and the run finished with exit 0. That is a reconstruction-only
measurement, with about 255 MiB to spare, so the full pipeline at 60 frames is not measured yet.

## 2. Alignment: gravity prior, yaw sweep, then ICP

Two captures have arbitrary relative poses. FPFH+RANSAC alone gave a symmetric wrong solution (180 deg,
fitness 0.65) on the synthetic room, because the flat floor repeats itself.

1. **Gravity prior.** A RANSAC plane fit finds the floor in each reconstruction. The normal is signed using
   camera 0, since OpenCV cameras look with y pointing down. This removes two rotational degrees of freedom.
2. **Candidates.** Yaw in 30-degree steps about the floor normal, plus the FPFH+RANSAC solution.
3. **ICP from every candidate** (point-to-plane). I keep the best by fitness, then inlier RMSE.
4. **Trimmed refinement.** I drop the worst 15% of residuals and refit, so moved objects do not pull the
   static background.
5. **Colour refinement (opt-in, `align.py --color`).** A colour-aware ICP (Park, Zhou and Koltun, ICCV 2017, as
   implemented in Open3D) runs after step 4 at three voxel scales. The colour term locks the sliding direction
   that point-to-plane ICP cannot see on a flat floor (see `research.md`, section 1).

**Measured effect of colour refinement:**

| Pair | Without colour | With colour |
|---|---|---|
| Synthetic, known 25 deg / 0.4 m perturbation | 2.41 deg, 4.4 cm, fitness 0.658 | 1.13 deg, 2.1 cm, fitness 0.678 |
| Real clips, before to after | fitness 0.653, inlier RMSE 2.19 cm | fitness 0.625, inlier RMSE 2.14 cm |

On the synthetic pair the colour step reaches the 2 deg / 5 cm target, which point-to-plane alone did not. On my
real clips the change is mixed: RMSE drops by half a millimetre and fitness drops by 0.03. So I kept colour
refinement off by default. I have one synthetic case and one real pair, so this is not enough to enable it.

On my real clips the alignment fitness is 0.65 (before to after) and 0.53 (control to before), with inlier RMSE
of 2.2-2.4 cm. I have no ground-truth transform for my clips, so those numbers cannot show accuracy.

## 3. Voxel change regions: why they find zero changes on my clips

**What the detector does.** Each capture is carved into a voxel grid (5 cm by default). Surface points mark
occupied voxels. Camera rays mark the empty voxels in front of each surface as free. A voxel is **removed** if
it is occupied before, free after, and the after capture has no surface within `occ_tolerance_vox` voxels.
It is **added** by the reverse rule. Connected clusters of removed or added voxels become regions.

**Result on my clips.** Before the tolerance, the masks are large: 622 removed-candidate voxels and 376
added-candidate voxels. With the default 10 cm tolerance, 23 and 13 remain. Of the before surface voxels,
77% have an after surface within 10 cm, and 88% of after voxels have a before surface within 10 cm. The two
reconstructions disagree at about the size of the changes, so the tolerance removes most real evidence along
with the noise.

**Sweep against the control.** I swept voxel size and tolerance. For each setting I compared the real pair with
the control (same room, reconstructed from other frames):

| Voxel | Tolerance (voxels) | Real: regions, largest | Control: regions, largest |
|---|---|---|---|
| 0.05 m | 0 | 8, 610 | 4, 1100 |
| 0.05 m | 1 | 14, 155 | 13, 346 |
| 0.05 m | 2 | 10, 9 | 17, 90 |
| 0.10 m | 0 | 4, 81 | 3, 174 |
| 0.10 m | 1 | 3, 3 | 6, 31 |
| 0.10 m | 2 | 0 | 1, 5 |
| 0.15 m | 0 | 1, 33 | 2, 52 |
| 0.15 m | 1 | 0 | 3, 1 |
| 0.15 m | 2 | 0 | 0 |

In every row, the control's largest artefact is at least as big as the real pair's largest candidate. No setting
separates my changes from reconstruction noise. I did not tune the detector until it reported something; that would
only fit the noise. The voxel detector stays at its default (5 cm, tolerance 2), so its 0 result is the honest result.

**Why my clips differ from the synthetic scene.** On the synthetic scene both captures come from the same
renderer, so the surfaces agree to a fraction of a voxel and the detector separates the four changes cleanly.
Real phone captures do not agree at that level.

## 4. Object-level changes (YOLO + 3D + footprint visibility check)

Object detection avoids the voxel problem. A moved chair's centre moves by about a metre, while surface noise
is a few centimetres, so comparing object centres is much more robust than comparing surface shells.

**Method (`scripts/detect_objects.py`).**

- The detector runs on each of the reconstructed frames (40 per capture, the same frames as the pointmaps).
  Two detectors are available:
  - `--detector yolo`: YOLOv8n with the 80 COCO classes.
  - `--detector yolo-world` (used by `run_all.py`): YOLO-World (`yolov8s-worldv2`) with an open vocabulary of
    30 room-object names (`ROOM_VOCAB`). I fixed the list before looking at any result. On the first 40 real
    before-frames it finds 20 classes at confidence 0.3 or higher, including desk, laptop, mouse, backpack,
    mug and pillow, which COCO cannot name.
- Each box's central 40% is looked up in that frame's pointmap, using the points above the frame's median
  confidence. The median of those points is the object's 3D position in the shared floor frame, in metres.
- Observations of the same class within 0.3 m are merged into one object per capture. An object must appear in
  at least 3 frames.
- Same-class objects are matched across captures by Hungarian assignment on 3D distance. A match under 0.3 m is
  **stayed**; 0.3-4 m is a candidate **moved**.
- **Footprint visibility check.** Each unmatched object, and the old and new spot of each candidate move, is tested
  at 9 points: the centre and 8 points on a 0.15 m ring at the same height. Each point is projected into every frame
  of the other capture, using that frame's pose and focal length. I checked the projection by reprojecting each
  frame's own pointmap onto its pixel grid: the median error is 0 px on the frames I tested. At the projected
  pixel, the other capture's depth decides the vote:
  - **empty**: the surface is behind the point by more than 0.15 m, so the ray reaches the point in free space;
  - **occupied**: a surface sits within 0.4 m in front of the point, or up to 0.15 m behind it;
  - **hidden**: a surface lies more than 0.4 m in front, so the point cannot be seen.
- A point is empty when at least 2 frames vote empty and none vote occupied. A spot is **empty** when at least two
  thirds of its voting sample points are empty, and **occupied** when at least one third are occupied.
- An unmatched before object is **removed** only if its spot is empty in the after capture; an unmatched after
  object is **added** only if its spot is empty in the before capture. A candidate move is **moved** only if both
  its old and new spots are empty.
- Everything else is **unconfirmed**. It is kept in the output with its reason, and it is not counted as a change.
- Outputs: `objects.json` (one record per object, with `kind`, positions and the votes) and `objects.md`
  (plain-language list, generated from `objects.json`).

**Null check.** I run the same stage with the no-change control as the "after" side. Every object should come out
as stayed or unconfirmed. Any moved, removed or added result is a false alarm.

**Results on my clips (min 3 frames):**

| Run | Stayed | Moved | Removed | Added | Unconfirmed |
|---|---|---|---|---|---|
| Real pair, COCO, centre point test (first visibility version) | 6 | 0 | 0 | 0 | 6 |
| Real pair, COCO, footprint test | 6 | 0 | 0 | 0 | 6 |
| Real pair, YOLO-World, footprint test | 7 | 0 | 0 | 0 | 12 |
| Control, COCO, footprint test | 6 | 0 | 0 | 0 | 4 |
| Control, YOLO-World, footprint test | 10 | 0 | 0 | 0 | 6 |

Before the visibility check, the COCO stage reported 1 moved, 4 added and 1 removed on the real pair, and 4
added/removed on the control. The footprint check removes every false alarm on the control, in both detectors,
and it confirms no change on the real pair. I count that as the honest result. The control is one run, so
"zero false alarms" is measured on one unchanged scene only.

The remaining real-pair cases, with the YOLO-World run:

- **Bed** (YOLO-World matched pair, 1.02 m displacement, -0.80 m vertical): both spots are occupied. The vertical rule
  (section 4) marks it unconfirmed. The earlier 0.36 m / 35 cm figure was from COCO. This fits a mismatch rather than a move. A bed also appears as a before-only object at its own spot.
- **Backpack, cup, phone, refrigerator, bag, bottle, chair, mug, pillow (2)**: each spot is occupied in the other
  capture, or not in view there. `objects.md` lists the reason for each.

**Scoring objects.** `evaluate_changes.py --objects` maps `objects.json` to the scorer's schema. Only confirmed
`moved`, `added` and `removed` records count as detections; `stayed` and `unconfirmed` do not. There are 9 unit tests
(passing), including one for the object mapping. On the real clips the scorer has no real ground truth to score
against yet. That needs the author's list of changes.

**Why it can still fail.**

- YOLO-World only finds the 30 prompts in `ROOM_VOCAB`. An object with another name is missed, and I have not
  tested prompts beyond that list.
- Box centres on thin or occluded objects can land on background, which biases the 3D position.
- The footprint is a horizontal ring at the object's centre height. It does not cover an object's full height,
  so a tall object standing on a surface can still be judged by its support.
- Only 40 pointmaps per capture exist, so an object seen in few frames has little evidence.

## 5. Semantics and language

**Voxel regions.** For each region, the box is projected into its best-view camera. That crop is labelled by OWLv2
(open vocabulary, over a room-object list). The colour name comes from the mean surface colour of the change. The
sentence template uses metric size and displacement, plus a left/right and toward/away phrase from the camera's
viewpoint. On the synthetic scene OWLv2 returned no label for the chair and labelled the box "laptop" at 0.15
confidence. Both are expected at 128x72 resolution. On real crops, OWLv2 agrees with YOLO-World on 24 of 34 comparable crops (`scripts/check_labels.py`). That is agreement between two detectors, not accuracy. My real
clips produced no voxel regions, so `descriptions/changes.md` is empty for the real pair.

**Object changes.** `objects.md` describes each moved, added and removed object in metres in the floor frame, and
lists every unconfirmed object with the reason it stayed unconfirmed. It does not use appearance or colour. The
object labels are YOLO class names. On the real pair it lists no confirmed change, only the unconfirmed ones.

## 6. Visualisation, graphs, floor and scoring

- **Bird's-eye map** (`bev.png`): the room is rotated so the floor is horizontal, using the floor fit in section 6.
  Changes are shown as coloured point sets with labelled convex hulls. This is only drawn for voxel regions.
- **Interactive viewer** (`viewer.html`): plotly 3D, with colour point clouds and toggleable change regions.
- **Graphs** (`scripts/make_graphs.py`, written by `run_all.py` to `outputs/graphs/`):
  - `top_down_difference.png`: floor-frame occupancy. Cells occupied only before are red, only after blue, both grey.
    On my clips: 323 cells only before, 102 only after.
  - `noise_floor.png`: region sizes on the real pair against the control, with the threshold (91 voxels). No real region reaches it.
  - `floor_heights.png`: heights of surface points above the fitted floor, per capture, with the 0 line.
  - I checked the graphs through their numbers only; I could not view the PNGs here.

**Floor choice.** `estimate_up` fits the top six planes and keeps the horizontal ones (normal within 53 deg of
camera 0's y axis). The lowest of those is the floor. The largest plane was often a desk or wall, so I stopped
using it. Each chosen plane is refined by least squares over all remaining points, and the result is the median
of five RANSAC runs. Plain RANSAC gave a 4-16 deg spread on the same points; the median brings five of six runs
within about 1 deg. One run is still about 10 deg off.

**Floor residual.** At the final code, 18.8% of before-capture points lie below the fitted floor, 30% of
after-capture points and 15.6% of control points. Keeping only the top 15% by confidence, about 5% of
before points lie more than 5 cm below. The floor normal moves 15-25 deg as the confidence cut changes, so the
floor is not well determined on these reconstructions. I think the cause is depth noise or non-planar warp,
not a wrong plane choice.

**Scoring (`scripts/evaluate_changes.py`).** It compares a change file with a ground-truth file. Each entry has
a type (added, removed or moved). Positions are optional. Without them, an entry matches any unused detection of
that type, so I only need the type of each change I made. With positions, a detection must lie within a tolerance
(0.5 m by default) of every given position. Matching is greedy and one-to-one.

- Unit tests (`tests/test_evaluate_changes.py`): 9/9 pass, including the object mapping.
- End to end on synthetic detections: 4/4 matched at 0.00-0.11 m (raw frame).
- `--frame floor` runs, but I have not checked it. The synthetic ground truth is stored in the raw frame, and I do
  not have ground truth in the floor frame, so this check cannot run yet.

## 7. What the pipeline produces, and what it does not yet

The submission plan asks for three things: a shared 3D representation, moved/added/removed regions shown in 3D,
and a language description of each change.

| Required output | Produced | Status on my real clips |
|---|---|---|
| Shared 3D representation | `reconstructions/*.npz`, `*.ply`, aligned to the floor frame | Produced. Alignment fitness 0.65, inlier RMSE 2.2-2.4 cm. |
| Changes visualised in 3D | `visualizations/viewer.html`, `bev.png` | Produced for voxel regions only; the real pair has none. Object changes are not drawn yet. |
| Changes described in language | `descriptions/changes.md`, `objects.md` | `changes.md` is empty on the real pair. `objects.md` lists only unconfirmed objects there. |
| Object-level changes | `objects/objects.json`, `objects_null/objects.json` | Produced. 0 confirmed changes on the real pair; 0 false alarms on the control. |

## 8. Limitations: status

Status values: **Fixed** (a test or a measurement in the note), **Partly fixed**, **Open**, **By design**.

| Item | Status | Note |
|---|---|---|
| Voxel regions find 0 changes on my clips | **Open** | 40-frame run: 0 regions. 60-frame run: 1 added region (19.4 L, 155 voxels) that passes the control cut-off. The control's largest artefact there is 133 voxels, so the margin is 22 voxels. The noise floor was 90 voxels at 40 frames and 133 at 60, so it moves between runs. Why still open: one candidate at that margin is not a reliable detection, and tuning the cut-off to fit my clips would be overfitting. |
| Object method has false alarms on an unchanged scene | **Partly fixed** | Footprint check plus YOLO-World: 0 false alarms on the control, COCO and YOLO-World. One control only. |
| Real-pair object changes not confirmed | **Open** | 40 frames: 0 confirmed. 60 frames (labeler on): 1 confirmed, a `bag` added at (0.81, 0.45, 0.03) m, seen in 7 frames. It is about 0.15 m from the 155-voxel region. 12 unconfirmed, each with its reason in `objects.md`. Not checked against the real scene. Why still open: scoring it needs your change list. |
| Point test on the object centre | **Fixed** | Replaced by a 9-point footprint test (section 4). Measured on the real pair and the control. |
| Bed "move" has a vertical jump | **Partly fixed** | Rule: a matched move with a vertical jump over 0.15 m is unconfirmed, with its reason. YOLO-World's bed jump is -0.80 m; the backpack's is -1.04 m (both unconfirmed). The rule is a physical assumption (floor objects do not rise), so an object on a bed or chair can be wrongly flagged. Tested in `tests/test_detect_objects.py`. The cause (centre mismatch) is not fixed. Tried comparing object bottoms (10th-percentile height) on the 60-frame pair: one bed went from -0.11 to 0.00 m, but the flagged bed got worse (-0.31 to -0.42 m), so I reverted it. |
| Floor residual (about 19% of points below the floor) | **Open** | Measured on the 40-frame before capture: 20.3% of points are below the median floor offset. A 3 cm margin still leaves 15.4% below, and only 5% sit below the 5th percentile. So the offset choice explains only part of it; about 15% of points lie more than 3 cm under the floor. Likely depth warp in the reconstruction. Why still open: fixing it means changing the reconstruction, and I have not tried one. |
| Floor normal unstable across RANSAC runs (4-16 deg) | **Fixed** | Measured on the 40-frame before capture: 20 trials at 15 runs, largest deviation from the consensus 1.03 deg, none over 5 deg. The same 20 trials at 5 runs: largest 2.30 deg, none over 5 deg. The earlier "1 of 8 at 5 runs" did not reproduce, so the 15-run default is kept because it halves the spread, at about 2.5 s per call. Previously (superseded): Default is now a median of 15 RANSAC runs (was 5). On the real before capture, 0 of 8 trials came out bad at 15 runs, against 1 of 8 at 5 runs (bad = a median pulled about 10 deg off by one RANSAC run). About 2 s per call. Only 8 trials. Previously: Least-squares refit and median of five runs: five of six within about 1 deg; one run still about 10 deg off. |
| No way to score detections against what I changed | **Partly fixed** | `evaluate_changes.py` scores voxel output and, with `--objects`, object output (9 unit tests). Real scoring needs the author's list of changes. |
| No graphs | **Fixed** | `make_graphs.py` writes three PNGs; checked through numbers only. |
| Object-level detection did not exist | **Fixed (experimental)** | `detect_objects.py` runs in `run_all`, with the footprint check and a null check. |
| Repository not on GitHub; submodule state dirty | **Fixed** | Pushed. MASt3R, DUSt3R and CroCo are pinned submodules, and a fresh recursive clone works. Each run regenerates `__pycache__` in the submodules; set `PYTHONDONTWRITEBYTECODE=1` to avoid it. |
| `--frame floor` scoring mode unchecked | **Partly fixed** | Bug fixed: the truth stayed in the raw frame while detections were rotated. `--truth-frame floor` now handles both cases, and a unit test on a tilted synthetic floor passes (10/10 in `test_evaluate_changes.py`). Not measured on real data. Why still open: scoring on real footage needs the author's changes in the floor frame. |
| Descriptions file is empty on the real pair | **Partly fixed** | Object descriptions (`objects.md`) are written. No confirmed change exists on the real pair, so `changes.md` stays empty. |
| Registration error about 2.4 deg / 4 cm on the synthetic pair | **Partly fixed** | Colour refinement reaches 1.13 deg / 2.1 cm on the synthetic pair (target met). It is opt-in, and on the real pair the effect is mixed. |
| Objects under 50 voxels are dropped | **Open** | Measured at `min_voxels=1`. 40 frames: real-pair regions 1-9 voxels; control 14 below 50 and 3 at 58-90. 60 frames: real pair 1-155 voxels (9 below 50, 1 at 155); control 12 below 50 and 2 at 71-133. The 50 cut-off does not remove any real candidate that is between 10 and 50 voxels; the control rule sets the real cut-off (91, then 134). Why still open: that rule is set by noise on one control, and it changed between runs. |
| YOLO COCO classes only | **Partly fixed** | YOLO-World with a 30-name room vocabulary (`--detector yolo-world`, used by `run_all`). Only those 30 names are tested. |
| Labels (OWLv2) not validated on real footage | **Partly fixed** | `check_labels.py` on 67 real crops: on 34 that both detectors can name, OWLv2 agrees with YOLO-World on 24 (70.6%). Agreement between two detectors, not accuracy against truth. Mug crops disagree (0 of 4). |
| Occlusion handled only by the 'seen' count and the hidden vote | **Partly fixed** | Hidden spots give no vote; a spot seen from one frame still scores low. |
| 60-frame reconstruction exceeds 8 GB VRAM | **Fixed** | 60 frames fit: peak 7,896 MiB of 8,151 MiB. The full `run_all.py --max-frames 60` (labeler on) ran with no errors in 2,028 s wall time, with peak 7,896 MiB sampled every 5 s. Its results differ from 40 frames (rows above). |
| Only one pair of captures | **Open** | Needs a new recording from the author. Why still open: I cannot record new clips from here. |
| Symmetric or flat scenes can flip the alignment | **Partly fixed** | Gravity prior and yaw sweep fix the case I tested. Other scenes are untested. |
| Synthetic clip artefacts (black bands) | **Fixed** | Measured: the generator now renders each 40-frame chunk in its own process (a fresh renderer). Both clips came out with 0 warnings (previously 343 of 720 before frames and 299 of 720 after frames flagged), 720 frames each at 960x540, checked with ffprobe. Before, the single renderer drifted over a long run; rebuilding it inside one process aborted Filament. Earlier note (superseded): Not fixed. The generator's own check flagged 343 of 720 before frames and 299 of 720 after frames after 12 tries each. Frames re-rendered alone pass, so the renderer drifts over a long run. Tried, no gain: rebuilding the renderer (Filament aborts when old renderers are destroyed, and keeping them hits the 32-timer limit), and reloading the scene in the same renderer (1,802 reloads, 221 flagged before frames by frame 553 against 266 at baseline). The generator is unchanged. Why still open: no reset found, and the synthetic clips are not used for real results. |
| MASt3R weights CC BY-NC-SA; YOLOv8 AGPL-3.0 | **By design** | Project is non-commercial. |
| Raw clips and reconstructions are not in the repo | **By design** | `data/`, `outputs/`, `models/` and `weights/` are git-ignored. |

## 9. Next steps

1. Score the object output and the voxel output against the changes I made on my clips, with `evaluate_changes.py`.
   I need the type and rough place of each change for that.
2. Done: `run_all.py --max-frames 60` with the labeler on ran in 2,028 s, peak 7,896 MiB (section 8). Next, score the 60-frame `bag` and 155-voxel region against your change list.
3. Check the OWLv2 labels against a hand-labelled set of real crops, so agreement becomes accuracy.
4. Record a second pair of captures with no changes, to measure repeatability beyond the control.
5. Measure the floor-frame scoring mode with ground truth in the floor frame.
