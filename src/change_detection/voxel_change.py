"""Phase 4 - geometric change detection in a shared voxel grid.

Each sequence is a set of posed cameras plus per-camera 3D points (depth lifted
by MASt3R and expressed in the shared frame).  For every sequence we build

  occupied[v]  a surface point falls in voxel v (and its mean colour)
  free[v]      a camera ray reached v before hitting a surface (seen empty)

A voxel that is *occupied before* and *free after* was removed; *occupied after*
and *free before* was added.  Connected clusters of those voxels become change
regions.  A removed and an added region with matching size, shape and colour
are reported as one MOVED object with a displacement vector.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage


@dataclass
class Sequence:
    """Posed observations of one capture in the shared frame."""
    name: str
    cam_c2w: np.ndarray                    # (N, 4, 4) camera-to-world
    cam_points: list                       # N arrays (M_i, 3): surface points (world frame)
    cam_colors: list | None = None         # N arrays (M_i, 3) in [0, 1], optional
    # optional far points along rays that hit nothing: only used to carve free space
    cam_free_points: list | None = None


@dataclass
class ChangeConfig:
    voxel: float = 0.05                    # metres
    min_voxels: int = 50                   # discard clusters smaller than this (noise)
    pad: float = 0.15
    ray_step: float = 0.5                  # ray sampling step in voxels
    max_rays_per_cam: int = 4000
    move_min_dist: float = 0.25            # centroid distance that separates "moved" from "same place"
    move_max_dist: float = 4.0
    move_size_ratio: tuple = (0.5, 2.0)    # volume ratio of added / removed
    move_shape_tol: float = 0.25           # relative error of sorted OBB extents
    move_color_tol: float = 0.20           # max RGB distance of mean colours (if colour is available)
    occ_tolerance_vox: int = 2             # a surface within this many voxels in the other capture is 'the same'


@dataclass
class Change:
    id: int
    kind: str                              # "added" | "removed" | "moved"
    voxels: int
    volume_m3: float
    centroid_before: np.ndarray | None
    centroid_after: np.ndarray | None
    displacement_m: float | None
    obb_center: np.ndarray                 # OBB of the object at its current (after) location, or only location
    obb_extents: np.ndarray                # metres, sorted descending
    obb_axes: np.ndarray                   # (3, 3) rows = axes
    confidence: float
    colour: np.ndarray | None
    points: np.ndarray                     # (K, 3) voxel centres in world frame
    best_view: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- grids
class Grid:
    def __init__(self, points_all: np.ndarray, voxel: float, pad: float):
        self.voxel = voxel
        self.lo = points_all.min(0) - pad
        hi = points_all.max(0) + pad
        self.shape = np.ceil((hi - self.lo) / voxel).astype(int)

    def index(self, pts: np.ndarray):
        """Return (voxel indices of points inside the grid, mask of which points were kept)."""
        idx = np.floor((pts - self.lo) / self.voxel).astype(int)
        ok = np.all((idx >= 0) & (idx < self.shape), axis=1)
        return idx[ok], ok

    def centre(self, idx: np.ndarray) -> np.ndarray:
        return self.lo + (idx + 0.5) * self.voxel


@dataclass
class Carved:
    occ: np.ndarray        # bool
    free: np.ndarray       # bool
    seen: np.ndarray       # int: number of cameras that observed the voxel
    colour: np.ndarray     # (X, Y, Z, 3) mean surface colour (nan where unknown)


def _carve_rays(origin, pts, grid, cfg, free, seen):
    """Mark voxels along origin->pts (excluding the endpoint) as free for one camera."""
    vec = pts - origin
    dist = np.linalg.norm(vec, axis=1)
    n_samp = max(2, int(np.ceil(dist.max() / (grid.voxel * cfg.ray_step))))
    s = np.linspace(0.0, 1.0, n_samp)
    # stop 2 voxels short of the surface so the hit voxel is not counted as free
    keep = np.clip(dist - 2 * grid.voxel, 0, None) / np.maximum(dist, 1e-9)
    samples = origin + s[None, :, None] * (vec * keep[:, None])[:, None, :]
    idx, _ = grid.index(samples.reshape(-1, 3))
    if len(idx):
        uniq = np.unique(idx, axis=0)
        free[tuple(uniq.T)] = True
        seen[tuple(uniq.T)] += 1


def _carve(seq: Sequence, grid: Grid, cfg: ChangeConfig) -> Carved:
    occ = np.zeros(grid.shape, bool)
    free = np.zeros(grid.shape, bool)
    seen = np.zeros(grid.shape, np.int32)
    col_sum = np.zeros(tuple(grid.shape) + (3,), np.float64)
    col_cnt = np.zeros(grid.shape, np.int32)
    rng = np.random.default_rng(0)
    for i, c2w in enumerate(seq.cam_c2w):
        pts = seq.cam_points[i]
        if len(pts) == 0:
            continue
        cols = seq.cam_colors[i] if seq.cam_colors is not None else None
        if len(pts) > cfg.max_rays_per_cam:
            sel = rng.choice(len(pts), cfg.max_rays_per_cam, replace=False)
            pts = pts[sel]
            cols = None if cols is None else cols[sel]
        idx, ok = grid.index(pts)
        occ[tuple(idx.T)] = True
        if cols is not None:
            np.add.at(col_sum, tuple(idx.T), cols[ok])
            np.add.at(col_cnt, tuple(idx.T), 1)
        _carve_rays(c2w[:3, 3], pts, grid, cfg, free, seen)
        if seq.cam_free_points is not None and len(seq.cam_free_points[i]):
            far = seq.cam_free_points[i]
            if len(far) > cfg.max_rays_per_cam:
                far = far[rng.choice(len(far), cfg.max_rays_per_cam, replace=False)]
            _carve_rays(c2w[:3, 3], far, grid, cfg, free, seen)
    free &= ~occ
    colour = np.full(tuple(grid.shape) + (3,), np.nan)
    have = col_cnt > 0
    colour[have] = col_sum[have] / col_cnt[have][:, None]
    return Carved(occ=occ, free=free, seen=seen, colour=colour)


def _obb(points: np.ndarray):
    c = points.mean(0)
    cov = np.cov((points - c).T) if len(points) > 3 else np.eye(3)
    _, v = np.linalg.eigh(cov)
    axes = v[:, ::-1].T                     # rows = principal axes, largest first
    proj = (points - c) @ axes.T
    lo, hi = proj.min(0), proj.max(0)
    centre = c + axes.T @ ((lo + hi) / 2)
    return centre, hi - lo, axes


def _clusters(mask: np.ndarray, cfg: ChangeConfig):
    """Connected components of a surface-like mask.

    Occupancy from surface points is a thin shell, so an opening would erase real changes.
    We label the one-voxel-dilated mask (bridges shell gaps) but count only original voxels.
    """
    structure = np.ones((3, 3, 3))
    grown = ndimage.binary_dilation(mask, structure=structure)
    lab, n = ndimage.label(grown, structure=structure)
    if n == 0:
        return []
    counts = np.bincount(lab[mask].ravel(), minlength=n + 1)
    return [np.argwhere(mask & (lab == k)) for k in range(1, n + 1) if counts[k] >= cfg.min_voxels]


def _best_view(cam_c2w: np.ndarray, centre: np.ndarray, max_dist: float = 6.0) -> int:
    """Camera that looks most directly at the region (used to crop frames for semantics)."""
    cam_pos = cam_c2w[:, :3, 3]
    fwd = cam_c2w[:, :3, 2]                 # OpenCV convention: +z is forward
    d = centre - cam_pos
    dist = np.linalg.norm(d, axis=1)
    cos = np.sum(fwd * d, 1) / np.maximum(dist, 1e-9)
    score = np.where(dist < max_dist, cos, -np.inf)
    i = int(np.argmax(score))
    return i if np.isfinite(score[i]) else int(np.argmin(dist))


def _region(idx, grid: Grid, carved_other: Carved, carved_self: Carved, kind: str, view_seq: Sequence):
    centres = grid.centre(idx)
    centre, extents, axes = _obb(centres)
    seen = carved_other.seen[tuple(idx.T)] + carved_self.seen[tuple(idx.T)]
    col = carved_self.colour[tuple(idx.T)]
    col = col[~np.isnan(col).any(1)]
    return dict(
        kind=kind, voxels=len(idx), volume=len(idx) * grid.voxel ** 3,
        centre=centre, extents=extents, axes=axes, points=centres,
        conf=float(np.clip(np.median(seen) / 4.0, 0.0, 1.0)),
        colour=col.mean(0) if len(col) else None,
        view=_best_view(view_seq.cam_c2w, centre), view_seq=view_seq.name,
    )


# --------------------------------------------------------------------------- main
def detect_changes(before: Sequence, after: Sequence, cfg: ChangeConfig | None = None) -> list[Change]:
    cfg = cfg or ChangeConfig()
    pts_all = [np.concatenate(before.cam_points), np.concatenate(after.cam_points),
               before.cam_c2w[:, :3, 3], after.cam_c2w[:, :3, 3]]
    grid = Grid(np.concatenate(pts_all), cfg.voxel, cfg.pad)

    cb = _carve(before, grid, cfg)
    ca = _carve(after, grid, cfg)
    # A voxel only counts as removed/added if the other capture has no surface within a small
    # tolerance: residual registration error otherwise turns desk edges into fake changes.
    tol = cfg.occ_tolerance_vox
    near_a = ndimage.binary_dilation(ca.occ, iterations=tol) if tol > 0 else ca.occ
    near_b = ndimage.binary_dilation(cb.occ, iterations=tol) if tol > 0 else cb.occ
    removed_mask = cb.occ & ca.free & ~near_a
    added_mask = ca.occ & cb.free & ~near_b

    # a removed object was visible in *before*, an added one in *after*: view from where it was seen
    removed = [_region(i, grid, ca, cb, "removed", before) for i in _clusters(removed_mask, cfg)]
    added = [_region(i, grid, cb, ca, "added", after) for i in _clusters(added_mask, cfg)]
    changes = _pair(removed, added, cfg)
    for k, c in enumerate(changes):
        c.id = k
    return changes


def _pair(removed: list, added: list, cfg: ChangeConfig) -> list[Change]:
    """Greedily pair removed/added regions that look like the same object at a new place."""
    candidates = []
    for i, r in enumerate(removed):
        for j, a in enumerate(added):
            ratio = a["volume"] / r["volume"]
            if not (cfg.move_size_ratio[0] <= ratio <= cfg.move_size_ratio[1]):
                continue
            dist = float(np.linalg.norm(a["centre"] - r["centre"]))
            if not (cfg.move_min_dist <= dist <= cfg.move_max_dist):
                continue
            er, ea = np.sort(r["extents"]), np.sort(a["extents"])
            shape_err = float(np.abs(ea - er).sum() / max(er.sum(), 1e-9))
            if shape_err > cfg.move_shape_tol:
                continue
            if r["colour"] is not None and a["colour"] is not None:
                if np.linalg.norm(r["colour"] - a["colour"]) > cfg.move_color_tol:
                    continue
            candidates.append((shape_err, i, j, dist))
    candidates.sort()
    used_r, used_a, out = set(), set(), []
    for _, i, j, dist in candidates:
        if i in used_r or j in used_a:
            continue
        used_r.add(i)
        used_a.add(j)
        out.append(_make_change("moved", removed[i], added[j], dist))
    out += [_make_change("removed", r, None, None) for i, r in enumerate(removed) if i not in used_r]
    out += [_make_change("added", None, a, None) for j, a in enumerate(added) if j not in used_a]
    return out


def _make_change(kind: str, r: dict | None, a: dict | None, dist: float | None) -> Change:
    # the OBB describes the object where it is now (after) if it exists, else where it was
    own = a if a is not None else r
    pts = np.concatenate([x["points"] for x in (r, a) if x is not None])
    centre, extents, axes = _obb(own["points"])
    colours = [x["colour"] for x in (r, a) if x is not None and x["colour"] is not None]
    view = own["view"]
    return Change(
        id=-1, kind=kind,
        voxels=int(sum(x["voxels"] for x in (r, a) if x is not None)),
        volume_m3=float(own["volume"]),
        centroid_before=None if r is None else r["centre"],
        centroid_after=None if a is None else a["centre"],
        displacement_m=dist,
        obb_center=centre, obb_extents=np.sort(extents)[::-1], obb_axes=axes,
        confidence=float(np.mean([x["conf"] for x in (r, a) if x is not None])),
        colour=np.mean(colours, axis=0) if colours else None,
        points=pts,
        best_view={"sequence": own["view_seq"], "index": int(view)},
    )
