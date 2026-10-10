# Design note

## 1. Shared 3D representation: MASt3R, not COLMAP + Gaussian Splatting

I chose MASt3R (metric checkpoint) for four reasons:

- It predicts dense pointmaps and camera poses from unordered phone frames. It needs no feature matching or
  bundle adjustment to get a usable reconstruction, and it works on textureless surfaces such as desks and walls.
- Each frame has a confident pointmap in a shared world frame, which the change detectors need.
- It runs in seconds per sequence on a laptop GPU, so anyone can rerun it.

COLMAP + 3DGS would look better, but photorealism does not improve change detection, so I left it as an
optional path.

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

On my clips the alignment fitness is 0.65 (before to after) and 0.53 (control to before), with inlier RMSE
of 2.2-2.4 cm. I have no ground-truth transform for my clips, so those numbers cannot show accuracy.

## 3. Voxel change regions: why they find zero changes on my clips

**What the detector does.** Each capture is carved into a voxel grid (5 cm by default). Surface points mark
occupied voxels. Camera rays mark the empty voxels in front of each surface as free. A voxel is **removed** if
it is occupied before, free after, and the after capture has no surface within `occ_tolerance_vox` voxels.
It is **added** by the reverse rule. Connected clusters of removed or added voxels become regions, and a removed
and an added region with matching size, shape and colour are paired as **moved**.

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

## 4. Object-level changes (YOLO + 3D)

Object detection avoids the voxel problem. A moved chair's centre moves by about a metre, while surface noise
is a few centimetres, so comparing object centres is much more robust than comparing surface shells.

**Method (`scripts/detect_objects.py`).**
- YOLOv8n (ultralytics, COCO classes) runs on each of the reconstructed frames (40 per capture, the same frames
  as the pointmaps).
- Each box's central 40% is looked up in that frame's pointmap, using the points above the frame's median confidence.
  The median of those points is the object's 3D position in the capture frame, in metres.
- Observations of the same class within 0.3 m are merged into one object per capture. An object must appear in at
  least 3 frames.
- Objects of the same class are matched across captures by Hungarian assignment on 3D distance. A match under
  0.3 m is **stayed**; 0.3-4 m is **moved**, with the displacement reported; an unmatched before object is
  **removed**; an unmatched after object is **added**.

**Null check.** I run the same stage with the no-change control as the "after" side. Every object should come out
as stayed. Anything else is a false alarm.

**Results on my clips (min 3 frames):**

| | Stayed | Moved | Added | Removed |
|---|---|---|---|---|
| Real pair | 6 | 1 (bed, 0.36 m) | 4 (cup 16 frames, laptop 4, person 3, suitcase 3) | 1 (backpack, 4 frames) |
| Null (control) | 6 | 0 | 2 (handbag, suitcase) | 2 (book, chair) |

The stayed-object displacements on the null reach about 0.23 m, so the 0.3 m stayed threshold is above that noise.
The null still produces four added/removed objects, all with 3 frames of support. That is the false-alarm rate
of this method on an unchanged scene, so with 3-frame support about a third of the real-pair add/remove events
could be noise too. The bed "move" has a 35 cm vertical jump, which points to a mismatch rather than a real move.
I have no ground truth to check the four real-pair add/remove events, so I am not claiming them as changes.

**Why it can still fail.**
- YOLOv8n has COCO classes only. Objects outside the 80 classes are not seen at all.
- Box centres on thin or occluded objects can land on background, which biases the 3D position.
- Only 40 pointmaps per capture exist, so an object seen in fewer frames has little support.
- The control shows that frame-to-frame detections flicker, so 3 frames is a weak support threshold.

## 5. Semantics and language

For each voxel region, the box is projected into its best-view camera. That crop is labelled by OWLv2 (open
vocabulary, over a room-object list). The colour name comes from the mean surface colour of the change. The
sentence template uses metric size and displacement, plus a left/right and toward/away phrase from the camera's
viewpoint. On the synthetic scene OWLv2 returned no label for the chair and labelled the box "laptop" at 0.15
confidence. Both are expected at 128x72 resolution. I have not validated labels on real footage, and my clips
produced no voxel regions to label.

## 6. Visualisation, graphs, floor and scoring

- **Bird's-eye map** (`bev.png`): the room is rotated so the floor is horizontal, using the floor fit in section 6.
  Changes are shown as coloured point sets with labelled convex hulls.
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

**Scoring (`scripts/evaluate_changes.py`).** It compares `changes.json` with a ground-truth file. Each entry has
a type (added, removed or moved). Positions are optional. Without them, an entry matches any unused detection of
that type, so I only need the type of each change I made. With positions, a detection must lie within a tolerance
(0.5 m by default) of every given position. Matching is greedy and one-to-one.
- Unit tests (`tests/test_evaluate_changes.py`): 8/8 pass.
- End to end on synthetic detections: 4/4 matched at 0.00-0.11 m.
- The `--frame floor` mode runs but is not checked against ground truth.
- The scorer is for voxel-region output. Object output (`objects.json`) has its own format, and I have not added a scorer for it yet.

## 7. Limitations: status

| Item | Status | Note |
|---|---|---|
| Voxel regions find 0 changes on my clips | **Open** | Control noise is at least as large as the real candidates at every voxel size and tolerance swept (section 3). |
| Object method has false alarms on an unchanged scene | **Open** | Null check gives 2 added and 2 removed objects, all with 3 frames of support. |
| Bed "move" (0.36 m) has a 35 cm vertical jump | **Open** | Probably a mismatch, not a real move. |
| Floor residual (about 19% of points below the floor) | **Open** | Stability improved, residual not reduced. Likely depth warp. |
| Floor normal unstable across RANSAC runs (4-16 deg) | **Partly fixed** | Least-squares refit and median of five runs: five of six within about 1 deg; one run still about 10 deg off. |
| No way to score detections against what I changed | **Fixed** | `evaluate_changes.py`, type-only or with positions. Tested on synthetic data. Object output not yet covered. |
| No graphs | **Fixed** | `make_graphs.py` writes three PNGs; checked through numbers only. |
| Object-level detection did not exist | **Fixed (experimental)** | `detect_objects.py` runs in `run_all`, with a null check. Its false-alarm rate is above zero (see above). |
| Repository not on GitHub; submodule state dirty | **Fixed** | Pushed. MASt3R, DUSt3R and CroCo are pinned submodules, and a fresh recursive clone works. Each run regenerates `__pycache__` in the submodules; clean it or set `PYTHONDONTWRITEBYTECODE=1`. |
| `--frame floor` scoring mode unchecked | **Open** | Runs, not compared with ground truth. |
| Descriptions file is empty | **Open** | Nothing survives the voxel threshold, so there is nothing to describe. |
| Registration error about 2.4 deg / 4 cm on the synthetic pair | **Open** | Colour-ICP or a learned matcher is needed to reach 2 deg. |
| Objects under 50 voxels are dropped | **Open** | Reconstruction noise on my clips (up to 90 voxels) is larger than that cut-off. |
| YOLO COCO classes only | **Open** | Objects outside the 80 classes are missed. |
| Labels (OWLv2) not validated on real footage | **Open** | Validated only on the synthetic scene, at low resolution. |
| Occlusion handled only by the 'seen' count | **Open** | A region seen from one camera scores lower but is still reported. |
| 60-frame reconstruction exceeds 8 GB VRAM | **Open** | I use a 40-frame cap. |
| Only one pair of captures | **Open** | No repeat capture, so repeatability is untested beyond the control. |
| Symmetric or flat scenes can flip the alignment | **Partly fixed** | Gravity prior and yaw sweep fix the case I tested. Other scenes are untested. |
| Synthetic video artefacts (black bands) | **Open** | Synthetic videos are not used for validation; I validate with ray-casting of the same scene. |
| MASt3R weights CC BY-NC-SA; YOLOv8 AGPL-3.0 | **By design** | Project is non-commercial. |
| Raw videos and reconstructions are not in the repo | **By design** | `data/` and `outputs/` are git-ignored. |

## 8. Next steps

1. Measure the changes I made on my clips (type and rough place), then score the object output and the voxel
   output against them with `evaluate_changes.py`.
2. Add a scorer for the object output, so the object method can be compared with ground truth too.
3. Record a second pair of captures with no changes, to measure repeatability beyond the control.
4. Replace the 3-frame support rule with a visibility check: an object counts as removed only if the other
   capture's pointmaps show empty space where it was.
5. Investigate the depth warp behind the floor residual, for example with joint reconstruction of both clips.
6. Colour-ICP refinement to reach the 2 deg target on feature-poor scenes.
