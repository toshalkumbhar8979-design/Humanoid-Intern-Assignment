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

## 2. Alignment: gravity prior, yaw sweep, then ICP

Two captures have arbitrary relative poses. FPFH+RANSAC alone gave a symmetric wrong solution (180 deg,
fitness 0.65) on the synthetic room, because the flat floor repeats itself.

1. **Gravity prior.** A RANSAC plane fit finds the floor in each reconstruction. The normal is signed using
   camera 0, since OpenCV cameras look with y pointing down. This removes two rotational degrees of freedom.
2. **Candidates.** Yaw in 30-degree steps about the floor normal, plus the FPFH+RANSAC solution.
3. **ICP from every candidate** (point-to-plane). I keep the best by fitness, then inlier RMSE.
4. **Trimmed refinement.** I drop the worst 15% of residuals and refit, so moved objects do not pull the
   static background.

On the synthetic pair with a known 25-degree yaw and 0.4 m shift, the method recovers the transform to 2.4 deg
and 4.4 cm. My target was 2 deg and 5 cm. I did not reach 2 deg because the flat, feature-poor floor lets ICP slide.
A voxel size of 0.025 m was much worse (17-31 deg), and trimming made no measurable difference here.

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

## 4. Object-level changes (YOLO + 3D + visibility check)

Object detection avoids the voxel problem. A moved chair's centre moves by about a metre, while surface noise
is a few centimetres, so comparing object centres is much more robust than comparing surface shells.

**Method (`scripts/detect_objects.py`).**

- YOLOv8n (ultralytics, COCO classes) runs on each of the reconstructed frames (40 per capture, the same frames
  as the pointmaps).
- Each box's central 40% is looked up in that frame's pointmap, using the points above the frame's median
  confidence. The median of those points is the object's 3D position in the shared floor frame, in metres.
- Observations of the same class within 0.3 m are merged into one object per capture. An object must appear in
  at least 3 frames.
- Same-class objects are matched across captures by Hungarian assignment on 3D distance. A match under 0.3 m is
  **stayed**; 0.3-4 m is a candidate **moved**.
- **Visibility check.** Each unmatched object, and the old and new spot of each candidate move, is projected into
  every frame of the other capture. The projection uses that frame's pose and focal length. I checked the convention
  by projecting each frame's own pointmap back onto its pixel grid: the median error is 0 px on the frames I tested.
  At the projected pixel, the other capture's depth decides the vote:
  - **empty**: the surface is behind the spot by more than 0.15 m, so the ray reaches the spot in free space;
  - **occupied**: a surface sits within 0.4 m in front of the spot, up to 0.15 m behind it;
  - **hidden**: a surface lies more than 0.4 m in front, so the spot cannot be seen.
- A spot is **empty** when at least 2 frames vote empty and none vote occupied. An unmatched before object is
  **removed** only if its spot is empty in the after capture; an unmatched after object is **added** only if its
  spot is empty in the before capture. A candidate move is **moved** only if both its old and new spots are empty.
- Everything else is **unconfirmed**. It is kept in the output for review and is not counted as a change.

**Null check.** I run the same stage with the no-change control as the "after" side. Every object should come out
as stayed or unconfirmed. Any moved, removed or added result is a false alarm.

**Results on my clips (min 3 frames, visibility check on):**

| | Stayed | Moved | Removed | Added | Unconfirmed |
|---|---|---|---|---|---|
| Real pair | 6 | 0 | 0 | 0 | 6 |
| Null (control) | 6 | 0 | 1 (chair) | 0 | 3 |

Before the visibility check, the same stage reported 1 moved, 4 added and 1 removed on the real pair, and
4 added/removed on the null. The check removed the null false alarms from 4 to 1, and moved the real-pair events
into the unconfirmed list. So the real-pair events are no longer reported as changes, because I could not confirm them.

The remaining real-pair cases, and why each stayed unconfirmed:

- **Bed** (candidate move, 0.36 m, 35 cm vertical): its new spot was occupied in the before capture (2 frames).
  This fits a mismatch rather than a move.
- **Backpack** (before only): visible in only 2 of 10 sampled after frames. In those frames a surface sits 4 cm
  behind its centre, so the spot is occupied by something else (probably a supporting surface or wall).
- **Cup, person** (after only): their spots are occupied in the before capture (8 and 7 frames).
- **Laptop, suitcase** (after only): their spots are not visible in the before capture at all.

The chair false alarm on the null is the clearest weakness. The control's reconstruction has no surface where
the chair stood, in all 4 frames that see that spot. Either the chair was not reconstructed in the control, or
the chair is too thin or dark for the depth to pick it up. An empty spot only shows the other reconstruction has
nothing there; it does not show the space is actually free. A point test of the object's centre cannot tell these apart.

**Why it can still fail.**

- YOLOv8n has COCO classes only. Objects outside the 80 classes are not seen at all.
- Box centres on thin or occluded objects can land on background, which biases the 3D position.
- The visibility test checks one point (the centre). It cannot see an object that sits on or against a surface,
  so the spot is often occupied by its own support.
- Only 40 pointmaps per capture exist, so an object seen in few frames has little evidence.

## 5. Semantics and language

For each voxel region, the box is projected into its best-view camera. That crop is labelled by OWLv2 (open
vocabulary, over a room-object list). The colour name comes from the mean surface colour of the change. The
sentence template uses metric size and displacement, plus a left/right and toward/away phrase from the camera's
viewpoint.

On the synthetic scene OWLv2 returned no label for the chair and labelled the box "laptop" at 0.15 confidence.
Both are expected at 128x72 resolution. I have not validated labels on real footage. My real clips produced no
voxel regions and no confirmed object changes, so there is nothing to describe yet. `descriptions/changes.md`
is empty for the real pair. That is the honest result, not a bug in the template.

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

- Unit tests (`tests/test_evaluate_changes.py`): 8/8 pass.
- End to end on synthetic detections: 4/4 matched at 0.00-0.11 m.
- The `--frame floor` mode runs but is not checked against ground truth.
- The scorer reads voxel-region output (`changes.json`). The object output (`objects/objects.json`) has its own
  format, and I have not added a scorer for it yet.

## 7. What the pipeline produces, and what it does not yet

The submission plan asks for three things: a shared 3D representation, moved/added/removed regions shown in 3D,
and a language description of each change. Here is where I stand on each:

| Required output | Produced | Status on my real clips |
|---|---|---|
| Shared 3D representation | `reconstructions/*.npz`, `*.ply`, aligned to the floor frame | Produced. Alignment fitness 0.65, inlier RMSE 2.2-2.4 cm. |
| Changes visualised in 3D | `visualizations/viewer.html`, `bev.png` | Produced for voxel regions only; the real pair has none. Object changes are not drawn yet. |
| Changes described in language | `descriptions/changes.md`, `changes.json` | Empty on the real pair. The object stage has no description step yet. |
| Object-level changes | `objects/objects.json`, `objects_null/objects.json` | Produced. 0 confirmed changes on the real pair, 1 false alarm on the null. |

## 8. Limitations: status

| Item | Status | Note |
|---|---|---|
| Voxel regions find 0 changes on my clips | **Open** | Control noise is at least as large as the real candidates at every voxel size and tolerance swept (section 3). |
| Object method has false alarms on an unchanged scene | **Partly fixed** | The visibility check cut the null from 4 false alarms to 1 (chair, 4 of 4 frames empty). Section 4. |
| Real-pair object changes not confirmed | **Open** | 0 confirmed on the real pair; 6 unconfirmed, each with its reason in section 4. |
| Point test on the object centre | **Open** | Cannot tell an empty spot from a spot the other reconstruction missed, and cannot see objects sitting on a surface. Next step: test the object's footprint. |
| Bed "move" (0.36 m) has a 35 cm vertical jump | **Open** | Now unconfirmed. Probably a mismatch, not a real move. |
| Floor residual (about 19% of points below the floor) | **Open** | Stability improved, residual not reduced. Likely depth warp. |
| Floor normal unstable across RANSAC runs (4-16 deg) | **Partly fixed** | Least-squares refit and median of five runs: five of six within about 1 deg; one run still about 10 deg off. |
| No way to score detections against what I changed | **Partly fixed** | `evaluate_changes.py` works for voxel output (tested on synthetic data). Object output has no scorer yet. |
| No graphs | **Fixed** | `make_graphs.py` writes three PNGs; checked through numbers only. |
| Object-level detection did not exist | **Fixed (experimental)** | `detect_objects.py` runs in `run_all`, with the visibility check and a null check. |
| Repository not on GitHub; submodule state dirty | **Fixed** | Pushed. MASt3R, DUSt3R and CroCo are pinned submodules, and a fresh recursive clone works. Each run regenerates `__pycache__` in the submodules; set `PYTHONDONTWRITEBYTECODE=1` to avoid it. |
| `--frame floor` scoring mode unchecked | **Open** | Runs, not compared with ground truth. |
| Descriptions file is empty on the real pair | **Open** | No confirmed voxel region or object change to describe. |
| Registration error about 2.4 deg / 4 cm on the synthetic pair | **Open** | Colour-ICP or a learned matcher is needed to reach 2 deg. |
| Objects under 50 voxels are dropped | **Open** | Reconstruction noise on my clips (up to 90 voxels) is larger than that cut-off. |
| YOLO COCO classes only | **Open** | Objects outside the 80 classes are missed. |
| Labels (OWLv2) not validated on real footage | **Open** | Validated only on the synthetic scene, at low resolution. |
| Occlusion handled only by the 'seen' count and the hidden vote | **Partly fixed** | Hidden spots give no vote; a spot seen from one frame still scores low. |
| 60-frame reconstruction exceeds 8 GB VRAM | **Open** | I use a 40-frame cap. |
| Only one pair of captures | **Open** | No repeat capture, so repeatability is untested beyond the control. |
| Symmetric or flat scenes can flip the alignment | **Partly fixed** | Gravity prior and yaw sweep fix the case I tested. Other scenes are untested. |
| Synthetic clip artefacts (black bands) | **Open** | Synthetic clips are used only for geometry checks, not for a full run. |
| MASt3R weights CC BY-NC-SA; YOLOv8 AGPL-3.0 | **By design** | Project is non-commercial. |
| Raw clips and reconstructions are not in the repo | **By design** | `data/` and `outputs/` are git-ignored. |

## 9. Next steps

1. Score the object output and the voxel output against the changes I made on my clips, with `evaluate_changes.py`.
   I need the type and rough place of each change for that.
2. Replace the centre point test with a footprint test: sample the object's extent, not only its centre, before
   calling its spot empty or occupied.
3. Add a scorer for the object output, so the object method can be compared with ground truth too.
4. Add a description step for object changes, so the object output also gets a language description.
5. Record a second pair of captures with no changes, to measure repeatability beyond the control.
6. Colour-ICP refinement to reach the 2 deg target on feature-poor scenes.
