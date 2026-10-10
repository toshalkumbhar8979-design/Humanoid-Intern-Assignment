# Design note

## 1. Shared 3D representation: MASt3R, not COLMAP + Gaussian Splatting

The plan offers two paths. I chose MASt3R (metric checkpoint) because:

- It predicts dense pointmaps and camera poses from unordered phone frames, so it needs no feature
  matching or bundle adjustment to get a usable reconstruction, and it is robust on textureless surfaces
  such as desks and walls.
- Its output is directly usable for change detection. Each frame has a confident pointmap in a shared
  world frame, which I need for ray carving.
- It runs in seconds per sequence on a laptop GPU, so a reviewer can rerun it.

COLMAP + 3DGS would give better photorealism for the visualisation, but is heavier to run, and
photorealism does not improve the change detection. I'd add it as an optional path if time allows.

Trade-off: MASt3R's weights are CC BY-NC-SA, so the project is non-commercial.

## 2. Alignment: gravity prior, yaw sweep, then ICP

Two captures have arbitrary relative poses. Trying FPFH+RANSAC alone on the synthetic room gave a
symmetric wrong solution (180 deg error, fitness 0.65), because the large flat floor has repeated
geometry.

Current method:

1. **Gravity prior.** A RANSAC plane fit finds the floor in each reconstruction. The normal is signed
   using camera 0, because OpenCV cameras look with y pointing down. This removes two rotational degrees
   of freedom.
2. **Candidates.** Yaw in 30-degree steps about the floor normal, plus the FPFH+RANSAC solution.
3. **ICP from every candidate** (point-to-plane). Keep the best by fitness, then inlier RMSE.
4. **Trimmed refinement.** Drop the worst 15% of residuals and re-fit, so moved objects do not
   pull the static background.

Verification: on the synthetic pair with a known 25-degree yaw and 0.4 m shift, the method recovers
the transform to 2.4 deg and 4.4 cm. The target was 2 deg and 5 cm. I did not reach 2 deg, because
the room is a large flat floor with few features, so ICP can slide along it. I tried voxel 0.025 m,
which was much worse (17-31 deg). Trimming made no measurable difference here.

On the real pair, alignment fitness is 0.65 for before-to-after and 0.53 for control-to-before, with
inlier RMSE about 2.2-2.4 cm. The 2.4 deg error is the synthetic figure; I have no ground-truth
transform for the real clips.

Next step to close the gap: colour-ICP (a colour gradient term on the floor texture and objects) or
a learned matcher such as MASt3R's own correspondence head.

## 3. Geometric change detection in 3D

- Each capture is carved into a voxel grid (5 cm). Surface points mark voxels occupied. Camera rays
  to those points mark voxels along the way as free. Rays that hit nothing contribute free space
  through far points.
- A voxel is **removed** if it is occupied before, free after, and the after capture has no surface
  within 2 voxels (10 cm). It is **added** by the reverse rule.
- The 2-voxel tolerance is the key fix. Before it, a 2.4 deg registration error turned desk edges into
  large fake "added" and "removed" regions. The counts still looked right, so the count-based test
  passed for the wrong reason. The test now checks each added and removed region against its
  ground-truth position and size.
- Connected components are labelled on a dilated mask, because a surface sampled from points is a thin
  shell and opening would erase it. Clusters under 50 voxels are discarded.
- A removed region and an added region are paired as **moved** when their volumes are 0.5-2x, their
  centroids are 0.25-4 m apart, their sorted OBB extents match within 25%, and their mean colours are
  within 0.2. The colour gate stops a plant from pairing with a backpack.
- Each change gets an oriented bounding box (PCA), a volume, a confidence (median camera coverage), and
  the camera that sees it best.

Result on the synthetic scene: 4/4 kinds correct, moved distances 1.2 m and 0.9 m (GT 1.2 and 0.8).
The object-level check passes.

### Result on real phone captures: no reliable change

On the user's two clips, the full pipeline runs and reports **0 changes** at the default 5 cm voxels.
That is not a finding about the room; it is the detector saying the signal is below its noise floor. To
measure that noise floor I built a no-change control:

- The unchanged before-capture is re-reconstructed from every other frame (`make_control.py`), so it is
  an independent MASt3R reconstruction of the same room, aligned to the original.
- Running the detector between the two gives 17 false regions, the largest 90-91 voxels (about 11 L).
  The real pair gives 10 candidate regions, none larger than the control's largest.
- Free-space evidence was also tested as a filter (require several camera rays before a voxel counts as
  empty). It did not remove the false regions, so it was dropped. The noise comes from disagreement
  between the two reconstructions, not from single grazing rays.
- `detect_changes.py --null <control>` now only reports regions larger than the control's largest
  artifact. On the real pair, nothing survives that rule.

The no-change control is what makes this honest: without it, the earlier 3 cm run produced 10 regions on
the real pair, and the control shows the same sizes are noise.

A 60-frame reconstruction was tried to reduce the disagreement. It filled the 8 GB GPU and ran global
alignment about 13x slower, so the 40-frame setting stays.

## 4. Semantics and language

For each change, the box is projected into its best-view camera and that crop is labelled by OWLv2
(open vocabulary, over a room-object list). The colour name comes from the mean surface colour of the
change. The sentence template uses metric size and displacement, plus a left/right and
toward/away phrase from the camera's own viewpoint, so the language matches the geometry.

Validation: OWLv2 runs on the GPU. On the synthetic scene it returned no label for the chair (the crop
was missing or too small) and labelled the box "laptop" at 0.15 confidence. Both are expected at
128x72 resolution. I have not validated labels on real footage yet.

## 5. Visualisation and graphs

- **Bird's-eye map** (`bev.png`): the room is rotated so the floor is horizontal (z up) using the
  plane fit in section 6. Changes are shown as coloured point sets with convex-hull footprints, and labelled.
- **Interactive viewer** (`viewer.html`): plotly 3D, with colour point clouds and toggleable change
  regions with bounding boxes. It runs offline.
- **Graphs** (`scripts/make_graphs.py`, written by `run_all.py` to `outputs/graphs/`), none needing ground truth:
  - `top_down_difference.png`: floor-frame occupancy. Cells occupied only before are red, only after blue,
    both grey. On the real pair: 323 cells only before, 102 only after. Coverage differences contribute,
    so isolated cells are not evidence of change.
  - `noise_floor.png`: region sizes on the real pair against the no-change control, with the pipeline's
    threshold (91 voxels). On the real pair no region reaches the threshold.
  - `floor_heights.png`: heights of surface points above the fitted floor, per capture, with the 0 line.
- Verification: the graph script exits 0 on the real outputs and inside `run_all` (log: RUN_ALL_EXIT=0). I
  could not open the PNGs visually in this environment, so the figures were checked through their numbers,
  not by eye.
- Not implemented: an orbit GIF and novel-view rendering of the change overlay.

## 6. Floor estimation and scoring

**Floor choice.** `estimate_up` fits the top six planes and keeps the horizontal ones (normal within
53 deg of camera 0's y axis). The lowest of those is the floor. The largest plane was previously used and
was often a desk or wall. Each chosen plane is refined by least squares over all remaining points, and the
result is the median of five RANSAC runs. Plain RANSAC gave a 4-16 deg spread on the same points; the median
brings five of six runs within about 1 deg, with one run still about 10 deg off.

**Floor residual (open).** At the final code, 18.8% of before-capture points lie below the fitted floor,
30% of after-capture points and 15.6% of control points. Keeping only the top 15% by confidence, about 5% of
before points lie more than 5 cm below the floor. The floor normal also moves by about 15-25 deg as the
confidence cut changes, so the floor is not well determined on these reconstructions. The likely cause is
depth noise or non-planar warp, not a wrong plane choice. The bird's-eye heights stay approximate.

**Scoring (`scripts/evaluate_changes.py`).** Compares `changes.json` with a ground-truth file. Entries have a
type (added, removed or moved). Positions are optional: with none, an entry matches any unused detection of
that type, so the scorer needs only the types of the changes you made. With positions, a detection must lie
within a tolerance (0.5 m default) of every given position. The matching is greedy and one-to-one.
- Unit tests (`tests/test_evaluate_changes.py`): 8/8 pass.
- End to end on synthetic detections from `test_change_detection_gt.py`: 4/4 matched, at 0.00-0.11 m.
- `--frame floor` (bird's-eye coordinates) runs but is **not checked** against ground truth.
- Real-footage scoring has not been run, because no ground truth was supplied.

## 7. Limitations: status

| Item | Status | Note |
|---|---|---|
| Real-footage changes cannot be told from reconstruction noise | **Open** | Real pair: 10 candidates, none above the 91-voxel control threshold. Needs ground truth and/or a better reconstruction. |
| Floor residual (about 19% of points below the floor) | **Open** | Stability improved (below), but the residual is not reduced. Likely depth warp. |
| Floor normal unstable across RANSAC runs (4-16 deg) | **Partly fixed** | Least-squares refit and median of five runs: five of six runs within about 1 deg. One run is still about 10 deg off. |
| No way to score detections against what was actually changed | **Fixed** | `evaluate_changes.py` with type-only or positional scoring. Tested on synthetic data. |
| No graphs | **Fixed** | `make_graphs.py` and `run_all.py` write three PNGs. Not checked by eye. |
| Repository not on GitHub; submodule state dirty | **Fixed** | Pushed to `Humanoid-Project` (`main`, latest `6f1c46c`). MASt3R and its dependencies are pinned submodules; a fresh recursive clone works. `git status` is clean. |
| `--frame floor` scoring mode unchecked | **Open** | Runs, but has not been compared with ground truth. |
| Descriptions file is empty | **By design, open** | Nothing survives the noise floor, so there is nothing to describe. |
| Registration error of about 2.4 deg / 4 cm on the synthetic pair | **Open** | Colour-ICP or a learned matcher would be needed to reach the 2 deg target. |
| Objects under 50 voxels (about 0.1 L) are dropped | **Open** | On the real captures, reconstruction noise (up to 91 voxels) is larger than that cutoff. |
| Labels (OWLv2) not validated on real footage | **Open** | Validated only on the synthetic scene, at low resolution. |
| Occlusion handled only by the 'seen' count in the confidence | **Open** | A region seen from one camera scores lower but is still reported. |
| 60-frame reconstruction exceeds 8 GB VRAM | **Open** | 40-frame cap is used. |
| Only one real pair of captures | **Open** | No repeat captures, so repeatability is untested beyond the control. |
| Synthetic video artefacts (black bands) | **Open** | Synthetic videos are not used for validation. Validation uses ray-casting of the same scene. |
| Symmetric or flat scenes can flip the alignment | **Partly fixed** | Gravity prior and yaw sweep fix the tested case. Other scenes are untested. |
| MASt3R weights are CC BY-NC-SA | **By design** | Project is non-commercial. |
| Raw videos and reconstructions are not in the repo | **By design** | `data/` and `outputs/` are gitignored, so the clone cannot reproduce real-footage results alone. |

## 8. Next steps

1. Score the real clips against your ground truth. Only the types of the changes are needed:
   `python scripts/evaluate_changes.py outputs/change_maps/changes.json your_truth.json`.
2. Record a second pair of captures with no changes. This gives a repeatability check that is independent of
   the control reconstruction.
3. Investigate the depth warp behind the floor residual, for example with joint reconstruction of both clips.
4. Colour-ICP refinement to meet the 2 deg target on feature-poor scenes.
5. Multi-view label voting, plus a confidence score per label.
6. Orbit GIF and a novel-view render of the change overlay.
