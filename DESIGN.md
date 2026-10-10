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
- Running the detector between the two gives 17 false regions, the largest 91 voxels (about 11 L).
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

## 5. Visualisation

- **Bird's-eye map** (`bev.png`): the room is rotated so the floor is horizontal (z up) using the
  plane fit above. Changes are shown as coloured point sets with convex-hull footprints, and labelled.
- **Floor choice.** `estimate_up` now fits the top six planes and keeps the horizontal ones (normal within
  53 deg of camera 0's y axis). The lowest of those is the floor. The largest plane was previously used and
  was often a desk or wall. Diagnostics show the chosen plane is the lowest horizontal candidate, but about
  18% of before-capture points still lie more than 5 cm below it (1st percentile -0.25 m). Those points
  are off every plane, so the cause is depth noise or non-planar warp across the room. This residual is
  not fixed, and the bird's-eye heights should be read with that in mind.
- **Interactive viewer** (`viewer.html`): plotly 3D, with colour point clouds and toggleable change
  regions with bounding boxes. It runs offline.
- Not implemented: an orbit GIF and novel-view rendering of the change overlay.

## 6. Failure cases and limitations

- **Synthetic video artefacts.** `make_synthetic_data.py` (Open3D offscreen rendering) produces black
  bands on most frames, so the videos are not usable for a full run. I did not use them for validation.
  Validation used direct ray-casting of the same scene instead.
- **Symmetric or flat scenes** can flip the alignment (seen as 180 deg with FPFH alone). The gravity
  prior and yaw sweep fix this for the tested case. Real rooms with more structure should be safer.
- **Registration error** near 2 deg moves edges. Tolerance is handled by the 10 cm occupancy margin, which
  is tuned for a 5 cm voxel. A much more accurate alignment would allow a smaller margin.
- **Small or thin objects** (under about 50 voxels, or 0.1 L) are dropped. On the real captures, the
  reconstruction noise (up to 91 voxels) is larger than that, so the real changes are not separable yet.
- **Labels** depend on crop resolution. Phone footage at 1080p should help. Labels are per region from one
  view, so I'd add voting across several views next.
- **Floor frame residual.** About 18% of points lie below the fitted floor (see section 5). Height-based
  readings from the bird's-eye view are therefore approximate.
- **Occlusion** is handled only by the 'seen' count in the confidence. A region seen from only one camera
  scores lower but is still reported.

## 7. Next steps

1. Get ground truth for the real captures (what was moved, added or removed, and where) and check each
   detection against it. The detector can't yet tell a real change from reconstruction noise on these clips.
2. Reduce reconstruction disagreement: colour-ICP refinement to meet the 2 deg target on feature-poor scenes,
   and a denser, better-overlapping capture.
3. Multi-view label voting, plus a confidence score per label.
4. Orbit GIF and a novel-view render of the change overlay.
