"""Phase 5 - semantic labels and grounded natural-language descriptions of changes.

For every change region:
  1. Project its 3D oriented bounding box into the best camera view of its sequence.
  2. Crop that view (from the MASt3R colour frames) around the projected box.
  3. Label the crop with OWLv2 open-vocabulary detection over a room-object vocabulary.
  4. Name the dominant colour (from the reconstructed surface colours).
  5. Describe the change with metric size/displacement and a left/right/toward/away phrase
     taken from the camera's own viewpoint, so the language is tied to the geometry.
"""
from __future__ import annotations

import numpy as np

OWL_MODEL = "google/owlv2-base-patch16-ensemble"
VOCAB = ["a chair", "a backpack", "a potted plant", "a bottle", "a cardboard box", "a desk",
         "a stool", "a lamp", "a bag", "a laptop", "a mug", "a book", "a shoe", "a cup",
         "a table", "a box", "a plant pot", "a pair of shoes", "a jacket", "a water bottle"]

_COLOURS = {
    "black": (0.08, 0.08, 0.09), "white": (0.92, 0.92, 0.92), "grey": (0.5, 0.5, 0.5),
    "red": (0.75, 0.13, 0.13), "green": (0.18, 0.55, 0.22), "blue": (0.1, 0.35, 0.9),
    "brown": (0.55, 0.38, 0.22), "beige": (0.82, 0.75, 0.6), "yellow": (0.9, 0.8, 0.2),
}


def colour_name(rgb: np.ndarray | None) -> str | None:
    if rgb is None:
        return None
    names = list(_COLOURS)
    d = [np.linalg.norm(np.asarray(rgb) - np.asarray(_COLOURS[n])) for n in names]
    return names[int(np.argmin(d))]


def project_bbox(corners_world: np.ndarray, c2w: np.ndarray, focal: float, hw: tuple[int, int]):
    """Project world-space corners into an image; return (x0, y0, x1, y1) or None if not visible."""
    h, w = hw
    w2c = np.linalg.inv(c2w)
    cam = corners_world @ w2c[:3, :3].T + w2c[:3, 3]
    z = cam[:, 2]
    if np.all(z <= 0.05):
        return None
    z = np.clip(z, 0.05, None)
    u = focal * cam[:, 0] / z + w / 2
    v = focal * cam[:, 1] / z + h / 2
    x0, x1 = max(0, int(np.floor(u.min()))), min(w, int(np.ceil(u.max())))
    y0, y1 = max(0, int(np.floor(v.min()))), min(h, int(np.ceil(v.max())))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return x0, y0, x1, y1


def obb_corners(center: np.ndarray, extents: np.ndarray, axes: np.ndarray) -> np.ndarray:
    signs = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]) / 2
    return center + (signs * extents) @ axes


class OwlLabeler:
    """OWLv2 open-vocabulary labeller for image crops."""

    def __init__(self, vocab: list[str] = VOCAB, device: str = "cuda"):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        self.torch = torch
        self.vocab = vocab
        self.device = device
        self.processor = Owlv2Processor.from_pretrained(OWL_MODEL)
        self.model = Owlv2ForObjectDetection.from_pretrained(OWL_MODEL).to(device).eval()

    def __call__(self, crop_rgb: np.ndarray) -> tuple[str, float]:
        from PIL import Image
        torch = self.torch
        inputs = self.processor(text=[self.vocab], images=Image.fromarray(crop_rgb),
                                return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits[0]          # (patches, queries)
        scores = logits.max(0).values.sigmoid()
        i = int(scores.argmax())
        return self.vocab[i].removeprefix("a ").removeprefix("an "), float(scores[i])


def _side_phrase(displacement: np.ndarray, cam_c2w: np.ndarray) -> str:
    """Left/right and toward/away from a camera's viewpoint."""
    d_cam = cam_c2w[:3, :3].T @ displacement
    lateral, depth = d_cam[0], d_cam[2]
    side = "right" if lateral > 0 else "left"
    depth_word = "away from the camera" if depth > 0 else "toward the camera"
    if abs(lateral) < 0.5 * abs(depth):
        return depth_word
    return f"to the {side}, {depth_word}" if abs(depth) > 0.5 * abs(lateral) else f"to the {side}"


def describe(changes, recon_by_name: dict, labeler=None) -> list[dict]:
    """Return one record per change: label, colour, grounded sentence and the evidence used.

    recon_by_name: {"before": recon, "after": recon} where recon has c2w (aligned), focals, colors.
    """
    records = []
    for ch in changes:
        view = ch.best_view
        recon = recon_by_name[view["sequence"]]
        i = view["index"]
        label, score, crop_box = None, None, None
        if labeler is not None:
            corners = obb_corners(ch.obb_center, ch.obb_extents, ch.obb_axes)
            hw = recon["colors"].shape[1:3]
            box = project_bbox(corners, recon["c2w"][i], float(recon["focals"][i]), hw)
            if box is not None:
                x0, y0, x1, y1 = box
                img = (recon["colors"][i][y0:y1, x0:x1] * 255).astype(np.uint8)
                label, score = labeler(img)
                crop_box = box
        if label is None or score is None or score < 0.1:
            label, score = "object", 0.0
        adj = colour_name(ch.colour)
        noun = f"{adj} {label}" if adj else label
        length_cm = int(round(100 * float(ch.obb_extents[0]) / 5) * 5)

        if ch.kind == "moved":
            disp = float(ch.displacement_m)
            ref_cam = recon_by_name["before"]["c2w"][len(recon_by_name["before"]["c2w"]) // 2]
            direction = ch.centroid_after - ch.centroid_before
            phrase = _side_phrase(direction, ref_cam)
            text = f"The {noun} was moved about {disp:.1f} m, {phrase}."
        elif ch.kind == "added":
            text = f"A {noun} (about {length_cm} cm long) was added."
        else:
            text = f"A {noun} that was there before is no longer present."
        records.append({
            "id": ch.id, "kind": ch.kind, "label": label, "label_score": round(score, 3),
            "colour": adj, "text": text, "confidence": round(ch.confidence, 3),
            "displacement_m": None if ch.displacement_m is None else round(float(ch.displacement_m), 3),
            "extents_m": [round(float(x), 3) for x in ch.obb_extents],
            "volume_l": round(float(ch.volume_m3) * 1e3, 2),
            "centroid_before": None if ch.centroid_before is None else np.round(ch.centroid_before, 3).tolist(),
            "centroid_after": None if ch.centroid_after is None else np.round(ch.centroid_after, 3).tolist(),
            "view": view, "crop_box": crop_box,
        })
    return records
