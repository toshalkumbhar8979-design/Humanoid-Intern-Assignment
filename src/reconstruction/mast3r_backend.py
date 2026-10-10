"""Phase 2 - dense reconstruction with MASt3R (primary path from the plan).

MASt3R predicts a dense pointmap for every image pair; a global optimisation then
aligns all pairs into one world frame and recovers camera poses and focal lengths.
We use the metric checkpoint so the output scale is close to metres.

The MASt3R/DUSt3R code is vendored as a git checkout under third_party/mast3r
(see README, step 1). Model weights are fetched from Hugging Face on first use.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_MAST3R = ROOT / "third_party" / "mast3r"
if str(_MAST3R) not in sys.path:
    sys.path.insert(0, str(_MAST3R))
    sys.path.insert(0, str(_MAST3R / "dust3r"))

MODEL_NAME = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"


def load_model(device: str = "cuda"):
    from mast3r.model import AsymmetricMASt3R
    return AsymmetricMASt3R.from_pretrained(MODEL_NAME).to(device).eval()


def _lean_inference(pairs, model, device: str):
    """Same as dust3r.inference.inference (batch size 1), but drops MASt3R's descriptors.

    The global aligner never reads `desc`/`desc_conf` (about 19 MB per view at 512 px). Keeping them
    for every pair exhausted host RAM on a ~90-frame sequence.
    """
    import torch
    from dust3r.inference import loss_of_one_batch
    from dust3r.utils.device import collate_with_cat, to_cpu

    out = []
    with torch.no_grad():
        for pair in pairs:
            res = loss_of_one_batch(collate_with_cat([pair]), model, None, device)
            for k in ("pred1", "pred2"):
                res[k].pop("desc", None)
                res[k].pop("desc_conf", None)
            out.append(to_cpu(res))
    return collate_with_cat(out)


def reconstruct(frame_paths: list[Path], model=None, device: str = "cuda", size: int = 512,
                scene_graph: str = "swin-2", niter: int = 300, lr: float = 0.01) -> dict:
    """Reconstruct a sequence. Returns numpy arrays in one shared world frame.

    points[i]  (H, W, 3)  world-frame pointmap of frame i
    conf[i]    (H, W)     MASt3R confidence
    colors[i]  (H, W, 3)  RGB in [0, 1]
    c2w        (N, 4, 4)  camera-to-world poses
    focals     (N,)       focal lengths in pixels (at the working resolution)
    """
    from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
    from dust3r.image_pairs import make_pairs
    from dust3r.utils.image import load_images

    model = model or load_model(device)
    imgs = load_images([str(p) for p in frame_paths], size=size, verbose=False)
    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True)
    output = _lean_inference(pairs, model, device)
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    scene.compute_global_alignment(init="mst", niter=niter, schedule="cosine", lr=lr)

    pts = [p.detach().cpu().numpy() for p in scene.get_pts3d()]
    conf = [c.detach().cpu().numpy() for c in scene.get_conf()]
    colors = [np.asarray(im, dtype=np.float32) for im in scene.imgs]
    c2w = scene.get_im_poses().detach().cpu().numpy()
    focals = scene.get_focals().detach().cpu().numpy().reshape(-1)
    return dict(points=np.stack(pts), conf=np.stack(conf), colors=np.stack(colors), c2w=c2w, focals=focals)


def to_sequence(name: str, recon: dict, conf_quantile: float = 0.3, stride: int = 2):
    """Convert a reconstruction into the Sequence used by change detection.

    Confident pixels become surface points (with colours). Zero-confidence pixels are rays that
    hit nothing; they become far 'free space' points that only carve empty voxels.
    """
    from change_detection.voxel_change import Sequence
    from registration.align import surface_mask

    surf_all = surface_mask(recon["conf"], conf_quantile)
    s = (slice(None, None, stride), slice(None, None, stride))
    cam_points, cam_colors, cam_free = [], [], []
    for i in range(len(recon["points"])):
        pts = recon["points"][i][s]
        cols = recon["colors"][i][s]
        surf = surf_all[i][s]
        free = (recon["conf"][i] <= 0)[s]
        cam_points.append(pts[surf].reshape(-1, 3))
        cam_colors.append(cols[surf].reshape(-1, 3))
        cam_free.append(pts[free].reshape(-1, 3))
    return Sequence(name=name, cam_c2w=recon["c2w"], cam_points=cam_points, cam_colors=cam_colors,
                    cam_free_points=cam_free)
