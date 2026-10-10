"""Visualisations for the change maps.

* floor_transform   - rotate the scene so the floor is horizontal with z up (RANSAC plane fit)
* transform_changes - apply the same rigid frame change to detected change regions
* bev_png           - bird's-eye change map (static figure for the README and report)
* viewer_html       - interactive 3D viewer (plotly, works offline in any browser)
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import open3d as o3d  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.spatial import ConvexHull  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from registration.align import estimate_up  # noqa: E402
from semantics.describe import obb_corners  # noqa: E402

KIND_RGB = {"removed": (0.90, 0.20, 0.20), "added": (0.20, 0.80, 0.25), "moved": (0.20, 0.45, 1.00)}
EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count("1") == 1]


def floor_transform(points: np.ndarray, cam0_c2w: np.ndarray):
    """Return (R, shift) with R @ p + shift putting the dominant plane (the floor) at z = 0, z up.

    OpenCV cameras have y pointing down, so 'up' is chosen with a negative dot product against the
    first camera's y axis. The plane is found with RANSAC on a random subsample.
    """
    up, floor_offset = estimate_up(points, cam0_c2w)
    R = Rotation.align_vectors([[0.0, 0.0, 1.0]], [up])[0].as_matrix()   # R @ up == z
    return R, -floor_offset


def _to_frame(x: np.ndarray, R: np.ndarray, shift: float) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64) @ R.T
    y[..., 2] += shift
    return y


def transform_changes(changes, R: np.ndarray, shift: float):
    out = []
    for c in changes:
        out.append(replace(
            c,
            points=_to_frame(c.points, R, shift),
            obb_center=_to_frame(c.obb_center, R, shift),
            obb_axes=c.obb_axes @ R.T,
            centroid_before=None if c.centroid_before is None else _to_frame(c.centroid_before, R, shift),
            centroid_after=None if c.centroid_after is None else _to_frame(c.centroid_after, R, shift),
        ))
    return out


def bev_png(before_pts: np.ndarray, after_pts: np.ndarray, changes, labels: dict, out_png: Path):
    """Bird's-eye (x, y) change map. Inputs must already be in the floor frame."""
    fig, ax = plt.subplots(figsize=(9, 7), dpi=160)
    ax.scatter(before_pts[:, 0], before_pts[:, 1], s=0.3, c="#8a9097", alpha=0.25, linewidths=0)
    ax.scatter(after_pts[:, 0], after_pts[:, 1], s=0.3, c="#b8c8dd", alpha=0.12, linewidths=0)
    for c in changes:
        col = KIND_RGB[c.kind]
        ax.scatter(c.points[:, 0], c.points[:, 1], s=4, color=col, linewidths=0)
        xy = obb_corners(c.obb_center, c.obb_extents, c.obb_axes)[:, :2]
        try:
            poly = xy[ConvexHull(xy).vertices]
        except Exception:
            poly = xy
        ax.fill(poly[:, 0], poly[:, 1], color=col, alpha=0.12)
        ax.plot(np.r_[poly[:, 0], poly[0, 0]], np.r_[poly[:, 1], poly[0, 1]], color=col, lw=1.5)
        cx, cy = poly.mean(0)
        ax.annotate(f"#{c.id} {labels.get(c.id, c.kind)}", (cx, cy), fontsize=8, ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85, lw=0))
    handles = [Patch(color=KIND_RGB[k], label=k) for k in KIND_RGB if any(c.kind == k for c in changes)]
    handles += [Patch(color="#8a9097", alpha=0.5, label="before scene"),
                Patch(color="#b8c8dd", alpha=0.5, label="after scene")]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("What changed in the room - bird's-eye view (floor-aligned)")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def viewer_html(before_pts, before_cols, after_pts, after_cols, changes, labels: dict, out_html: Path):
    """Interactive 3D viewer: colour point clouds + coloured change regions with bounding boxes."""
    import plotly.graph_objects as go

    def rgb(c):
        c = np.clip(np.asarray(c, dtype=np.float64), 0, 1)
        return [f"rgb({int(255 * r)},{int(255 * g)},{int(255 * b)})" for r, g, b in c]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=before_pts[:, 0], y=before_pts[:, 1], z=before_pts[:, 2], mode="markers",
                               marker=dict(size=1.3, color=rgb(before_cols), opacity=0.55),
                               name="before scene", legendgroup="scene"))
    fig.add_trace(go.Scatter3d(x=after_pts[:, 0], y=after_pts[:, 1], z=after_pts[:, 2], mode="markers",
                               marker=dict(size=1.3, color=rgb(after_cols), opacity=0.25),
                               name="after scene (aligned)", legendgroup="scene", visible="legendonly"))
    for c in changes:
        col = rgb([KIND_RGB[c.kind]])[0]
        name = f"#{c.id} {labels.get(c.id, c.kind)} ({c.kind})"
        fig.add_trace(go.Scatter3d(x=c.points[:, 0], y=c.points[:, 1], z=c.points[:, 2], mode="markers",
                                   marker=dict(size=2.5, color=col), name=name, legendgroup=name))
        corners = obb_corners(c.obb_center, c.obb_extents, c.obb_axes)
        xs, ys, zs = [], [], []
        for i, j in EDGES:
            xs += [corners[i, 0], corners[j, 0], None]
            ys += [corners[i, 1], corners[j, 1], None]
            zs += [corners[i, 2], corners[j, 2], None]
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=col, width=6),
                                   name=name, legendgroup=name, showlegend=False))
    fig.update_layout(
        title="What changed in the room - interactive view (click legend entries to toggle)",
        scene=dict(aspectmode="data", xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="height (m)"),
        margin=dict(l=0, r=0, t=50, b=0), legend=dict(itemsizing="constant"),
    )
    fig.write_html(str(out_html), include_plotlyjs=True, full_html=True)
