# Research notes

Sources I read while fixing the open items. Web search was unavailable in this session, so every source below
is a primary document I opened directly (Open3D and Ultralytics documentation). Where I rely on a result I
measured myself, I say so.

## 1. Colour-aware ICP for the alignment (open: 2.4 deg on the synthetic pair)

Source: Open3D, "Colored point cloud registration" tutorial (Open3D 0.20 docs), which implements
Park, Zhou and Koltun, *Colored Point Cloud Registration Revisited*, ICCV 2017.

- Point-to-plane ICP can slide along a flat surface, because a plane constrains only the normal direction.
  The paper adds a photometric term: colour is compared on the tangent plane, which locks the tangential direction.
- The objective is `E = (1 - delta) * E_C + delta * E_G`, where `E_G` is the point-to-plane term and `E_C` the colour term.
- Multi-scale schedule from the tutorial: voxel radii `[0.04, 0.02, 0.01]` with iteration caps `[50, 30, 14]`.
  Each point cloud is downsampled at that radius and normals are estimated with radius `2 * radius`, max 30 neighbours.
- Call: `o3d.pipelines.registration.registration_colored_icp(source, target, radius, init, TransformationEstimationForColoredICP(), ICPConvergenceCriteria(...))`.
- The tutorial's own demo shows point-to-plane ICP with visibly misaligned textures on a planar region, which is the
  failure I saw on the flat synthetic floor.

How it is used here: a colour-ICP step after the existing point-to-plane refinement in `src/registration/align.py`.
Its result is measured against the known 25 deg / 0.4 m perturbation, not assumed to help.

## 2. Open-vocabulary object detection (open: COCO-only classes)

Source: Ultralytics, "YOLO-World Model" documentation (docs.ultralytics.com/models/yolo-world).

- `YOLOWorld("yolov8s-worldv2.pt")` loads the model. `model.set_classes([...])` sets a custom vocabulary without retraining.
- The `worldv2` weights are the ones the docs recommend, and they can be exported (ONNX, TensorRT). The `world` (v1) weights cannot.
- The docs suggest a `""` background class can help in some scenes, but the effect is scenario-dependent. I did not adopt it.
- Predictions use the same `results[0].boxes` interface as YOLOv8, so the existing lifting code in `detect_objects.py` is unchanged.

How it is used here: `detect_objects.py --detector yolo-world`, with a room vocabulary fixed before looking at results
(`ROOM_VOCAB`). Measured on the first 40 real before-frames: 20 classes detected at confidence 0.3 or higher,
including desk, laptop, mouse, backpack, mug and pillow, which the COCO model cannot name.

## 3. Visibility and free space (open: point test on object centres)

No external source used. This is my own design, documented in DESIGN.md section 4. The idea is standard ray-based
free-space reasoning: a spot is free when the ray from the camera reaches a surface behind it. I added a footprint
ring (centre plus 8 points at 0.15 m) so that one point landing on a support or wall does not decide the outcome.
The projection convention was checked by reprojecting each frame's own pointmap (0 px median error).

## 4. Still unresolved, and what would resolve them

- Repeat capture (repeatability): needs a new recording from the author. No source can replace it.
- Ground truth for the real pair: needs the author's list of changes. No source can replace it.
- Floor residual (about 19% of points below the floor): unresolved. The likely cause is depth warp in MASt3R
  pointmaps; I have no source that fixes it for this capture setup.
