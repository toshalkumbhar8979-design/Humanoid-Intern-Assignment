"""Unit tests for scripts/detect_objects.py (rules and text only, no model weights needed).

Run:  python tests/test_detect_objects.py      (or: pytest tests/test_detect_objects.py)
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import detect_objects as do  # noqa: E402


def test_small_vertical_change_is_allowed():
    assert do.vertical_reason(0.10, 0.15) is None
    assert do.vertical_reason(-0.15, 0.15) is None


def test_large_vertical_jump_is_rejected_with_reason():
    reason = do.vertical_reason(0.36, 0.15)
    assert reason is not None and "vertical jump +0.36 m" in reason


def test_large_drop_is_a_legitimate_move():
    # ObChange ground truth has a real move down 0.22 m between surfaces (table to lower shelf);
    # the old symmetric rule rejected it. Only rises are rejected now.
    assert do.vertical_reason(-0.22, 0.15) is None


def test_one_frame_cannot_declare_a_spot_occupied():
    # mixed evidence or a single vote is unknown; only clean agreement decides
    assert do.vote_status(3, 1, 5, 2) == "unknown"
    assert do.vote_status(0, 1, 5, 2) == "unknown"
    assert do.vote_status(1, 1, 2, 2) == "unknown"
    assert do.vote_status(3, 0, 5, 2) == "empty"
    assert do.vote_status(0, 3, 5, 2) == "occupied"


def test_status_needs_the_minimum_seen():
    assert do.vote_status(2, 0, 1, 2, min_seen=2) == "unknown"
    assert do.vote_status(2, 0, 2, 2, min_seen=2) == "empty"


TO_FLOOR = lambda p: [p[0], p[1], -p[2]]   # height = -z: the camera looks down its +z axis


def _fake_capture(n_frames=4, behind=0.12, drop=0.12, spot_depth=2.0):
    """A capture looking at world X = [0, 0, 0]; the surface at X's pixel is `behind` metres past X
    along the ray and `drop` metres below it (height -z). behind = drop = 0 means an object's own
    surface sits at the spot. A large negative `behind` is an occluder in front of the spot.

    The two signals feed the rule directly (depth from Z, height from points); they decouple where a
    real surface would tie them (e.g. a sloped support can sit at the spot's depth but below it).
    """
    H = W = 16
    Z = np.full((n_frames, H, W), np.nan)
    P = np.zeros((n_frames, H, W, 3))
    c2w = np.tile(np.eye(4), (n_frames, 1, 1))
    for i in range(n_frames):
        c2w[i, 2, 3] = -spot_depth                        # camera at z = -2, looking +z (down)
        Z[i, :, :] = spot_depth + behind                  # camera depth of the surface
        P[i, :, :, 2] = drop                              # world z; height = -z -> hit sits `drop` below X
    return {"c2w": c2w, "focals": np.full(n_frames, 8.0), "points": P}, Z


def test_bare_support_at_the_spot_reads_empty():
    # object gone: the ray passes the spot and lands on the support behind and below it
    cap, Z = _fake_capture(behind=0.12, drop=0.12)
    s = do.spot_status(np.zeros(3), cap, Z, margin=0.15, radius=0.4, min_votes=2,
                       to_floor=TO_FLOOR, rise_tol=0.03, gap=0.03)
    assert s["status"] == "empty", s


def test_support_below_the_spot_reads_empty():
    # grazing view: the hit is at the spot's depth but clearly below it -> bare support
    cap, Z = _fake_capture(behind=0.0, drop=0.12)
    s = do.spot_status(np.zeros(3), cap, Z, margin=0.15, radius=0.4, min_votes=2,
                       to_floor=TO_FLOOR, rise_tol=0.03, gap=0.03)
    assert s["status"] == "empty", s


def test_object_standing_at_the_spot_reads_occupied():
    # object still there: its own surface is at the spot's depth and level with it
    cap, Z = _fake_capture(behind=0.0, drop=0.0)
    s = do.spot_status(np.zeros(3), cap, Z, margin=0.15, radius=0.4, min_votes=2,
                       to_floor=TO_FLOOR, rise_tol=0.03, gap=0.03)
    assert s["status"] == "occupied", s


def test_occluder_in_front_gives_no_vote():
    # a surface 0.5 m in front of the spot hides it: no vote at all
    cap, Z = _fake_capture(behind=-0.5, drop=0.0)
    s = do.spot_status(np.zeros(3), cap, Z, margin=0.15, radius=0.4, min_votes=2,
                       to_floor=TO_FLOOR, rise_tol=0.03, gap=0.03)
    assert s["seen"] == 0 and s["status"] == "unknown", s


def test_footprint_ring_is_horizontal_in_the_floor_frame():
    # with a floor normal along +z in the recon frame, ring samples share the centre's z
    class A:
        footprint, margin, radius, min_votes, min_seen = 0.15, 0.15, 0.4, 2, 2
    cap, Z = _fake_capture(behind=0.12, drop=0.12)
    calls = []
    orig = do.spot_status
    def spy(X, *args, **kwargs):
        calls.append(X.copy())
        return {"status": "empty", "empty": 1, "occupied": 0, "seen": 1}
    do.spot_status = spy
    try:
        do.footprint_status(np.zeros(3), cap, Z, A, normal=np.array([0.0, 0.0, 1.0]))
    finally:
        do.spot_status = orig
    zs = [p[2] for p in calls[1:]]            # the 8 ring samples
    assert max(zs) - min(zs) < 1e-9, calls    # one height: the ring is horizontal...
    assert abs(zs[0] - 0.04) < 1e-9, calls    # ...and lifted above the support contact plane
    radii = [float(np.linalg.norm(p[:2])) for p in calls[1:]]
    assert all(abs(r - 0.15) < 1e-9 for r in radii)


def test_to_floor_xyz_maps_recon_frame_to_z_up():
    # raw frame rotated 90 deg about x (raw +y is floor up); heights must come out on the floor axis
    R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    p = do.to_floor_xyz([0.0, 2.0, 1.0], R, -0.5)
    assert np.allclose(p, [0.0, -1.0, 1.5])


def test_duplicate_labels_on_one_object_merge_to_one_cluster():
    # observed on ObChange: one object fired 'book' and 'box' at the same point and became two
    # records; objects cannot occupy the same volume, so the shorter track is dropped
    objs = [
        {"cls": "book", "xyz": np.zeros(3), "frames": 16, "conf": 0.6},
        {"cls": "box", "xyz": np.array([0.001, 0.0, 0.0]), "frames": 14, "conf": 0.7},
        {"cls": "cup", "xyz": np.array([0.25, 0.0, 0.0]), "frames": 5, "conf": 0.7},
    ]
    out = do.merge_duplicates(objs)
    assert len(out) == 2
    assert {(o["cls"], o["frames"]) for o in out} == {("book", 16), ("cup", 5)}


def test_a_hit_on_a_moved_object_does_not_count_as_occupied():
    # the plate was added 0.15 m from where the sugar box used to sit; the box's body vetoed the
    # addition until hits on objects that moved away were attributed to them instead
    cap, Z = _fake_capture(behind=0.0, drop=0.0)          # a surface at the spot: 'occupied' raw
    moved_pt = np.zeros(3)                                # the surface belongs to an object that left
    s = do.spot_status(np.zeros(3), cap, Z, margin=0.15, radius=0.4, min_votes=2,
                       to_floor=TO_FLOOR, rise_tol=0.03, gap=0.03, free_pts=[moved_pt])
    assert s["status"] == "empty", s
    s2 = do.spot_status(np.zeros(3), cap, Z, margin=0.15, radius=0.4, min_votes=2,
                        to_floor=TO_FLOOR, rise_tol=0.03, gap=0.03,
                        free_pts=[moved_pt + 1.0])        # ... but not one from a far-away object
    assert s2["status"] == "occupied", s2


class _Args:
    same, max_move, max_vertical = 0.3, 4.0, 0.15
    verify_pairs, attr = 0.02, 0.12
    footprint, margin, radius, min_votes, min_seen = 0.08, 0.15, 0.4, 2, 2
    rise_tol, gap = 0.03, 0.03


def _classify_with(script, before, after):
    """Run classify with a scripted footprint verdict per world point."""
    orig = do.footprint_status
    do.footprint_status = lambda X, other, Z, args, to_floor=None, normal=None, free_pts=(): \
        dict(script(tuple(np.round(np.asarray(X, float), 3))))
    try:
        return do.classify(before, after, {}, {}, None, None, _Args)
    finally:
        do.footprint_status = orig


def _obj(cls, xyz, frames=5):
    return {"cls": cls, "xyz": np.asarray(xyz, float), "frames": frames, "conf": 0.9}


def _spot(status, occ=0, emp=9):
    return {"status": status, "empty_samples": emp, "occupied_samples": occ, "unknown_samples": 9 - occ - emp,
            "samples": 9}


def test_pair_below_same_is_verified_not_assumed():
    # both spots occupied -> the object really stayed, even 0.15 m of pairing noise apart
    b, a = _obj("cup", [0, 0, 0]), _obj("cup", [0.15, 0, 0])
    out = _classify_with(lambda p: _spot("occupied", occ=7, emp=0), [b], [a])
    assert [r["kind"] for r in out] == ["stayed"], out


def test_false_pair_splits_into_stayed_and_added():
    # observed on ObChange: an added object 0.15 m from a stayed same-label neighbour was matched to
    # it and vanished as 'stayed'. The neighbour kept its spot (old occupied) and the new spot was
    # bare (strong empty), so the pair is two objects: one stayed, one added.
    b, a = _obj("cup", [0, 0, 0]), _obj("cup", [0.15, 0, 0])
    out = _classify_with(lambda p: _spot("empty") if p[0] > 0.05 else _spot("occupied", occ=7, emp=0), [b], [a])
    kinds = sorted(r["kind"] for r in out)
    assert kinds == ["added", "stayed"], out
    added = next(r for r in out if r["kind"] == "added")
    assert added["split_from_pair"] and added["side"] == "after only"


def test_weak_evidence_does_not_split_a_pair():
    # one occupied sample in the new spot is enough to keep the pair unconfirmed, not split it
    b, a = _obj("cup", [0, 0, 0]), _obj("cup", [0.15, 0, 0])
    out = _classify_with(lambda p: _spot("empty", occ=1, emp=8) if p[0] > 0.05 else _spot("occupied", occ=7, emp=0),
                         [b], [a])
    assert [r["kind"] for r in out] == ["unconfirmed"], out


def test_ambiguous_small_pair_is_not_called_a_move():
    # both spots empty with sub-threshold displacement reads the same for one small move and for a
    # removal plus an appearance wrongly paired; no control can tell them apart, so it stays
    # unconfirmed (ObChange showed a false 'move' exactly this way)
    b, a = _obj("cup", [0, 0, 0]), _obj("cup", [0.15, 0, 0])
    out = _classify_with(lambda p: _spot("empty"), [b], [a])
    assert [r["kind"] for r in out] == ["unconfirmed"], out
    assert "cannot tell apart" in out[0]["reason"]


def test_spot_lift_must_exceed_the_drop_tolerance():
    # at the support contact the bare table reads 'occupied' at grazing angles; the lifted point makes
    # the height signal decisive. This is the geometry bug behind every rejected tabletop addition.
    class A:
        footprint, margin, radius, min_votes, min_seen = 0.08, 0.15, 0.4, 2, 2
        rise_tol, gap, attr, spot_lift = 0.03, 0.03, 0.12, 0.04
    assert A.spot_lift > A.rise_tol
    cap, Z = _fake_capture(behind=0.0, drop=0.0)     # support at the spot's depth (grazing view)
    # at the contact plane the height signal cannot see the support -> occupied
    s0 = do.spot_status(np.zeros(3), cap, Z, margin=0.15, radius=0.4, min_votes=2,
                        to_floor=TO_FLOOR, rise_tol=0.03, gap=0.03)
    assert s0["status"] == "occupied", s0
    # the same rays against the lifted footprint: the support is now clearly below -> empty
    s1 = do.footprint_status(np.zeros(3), cap, Z, A, to_floor=TO_FLOOR,
                             normal=np.array([0.0, 0.0, -1.0]))   # floor up is -z in the fake frame
    assert s1["status"] == "empty", s1


def test_pairs_below_the_noise_band_are_stayed_without_tests():
    # the author's no-change control drifts up to 0.21 m; at that scale testing manufactures moves
    # out of noise, so pairs under --verify-pairs are stayed outright
    b, a = _obj("cup", [0, 0, 0]), _obj("cup", [0.015, 0, 0])
    calls = []
    orig = do.footprint_status
    def spy(X, *args, **kwargs):
        calls.append(1)
        return _spot("empty")
    do.footprint_status = spy
    try:
        out = do.classify([b], [a], {}, {}, None, None, _Args)
    finally:
        do.footprint_status = orig
    assert [r["kind"] for r in out] == ["stayed"] and not calls, (out, calls)


def test_verify_pairs_off_keeps_every_small_pair_stayed():
    # --verify-pairs 0 turns the check off entirely (run_all measures the control noise with it)
    b, a = _obj("cup", [0, 0, 0]), _obj("cup", [0.25, 0, 0])
    args = type("A", (), {**vars(_Args)})()
    args.verify_pairs = 0.0
    calls = []
    orig = do.footprint_status
    do.footprint_status = lambda *ka, **kw: calls.append(1) or _spot("empty")
    try:
        out = do.classify([b], [a], {}, {}, None, None, args)
    finally:
        do.footprint_status = orig
    assert [r["kind"] for r in out] == ["stayed"] and not calls, (out, calls)


def test_objects_md_states_the_vertical_reason():
    changes = [{
        "cls": "bed", "kind": "unconfirmed", "side": "matched pair", "displacement_m": 0.365,
        "vertical_m": 0.36, "before_xyz": [0, 0, 0], "after_xyz": [0.3, 0, 0.36],
        "frames_before": 3, "frames_after": 3,
        "old_spot": {"status": "occupied"}, "new_spot": {"status": "occupied"},
        "reason": do.vertical_reason(0.36, 0.15),
    }]
    text = do.describe_objects(changes)
    assert "bed (matched pair): candidate move, old spot occupied, new spot occupied; vertical jump" in text


def _pair(reason, b=(0.49, 0.45, 0.03), a=(0.35, 0.53, 0.03)):
    return {"cls": "cup", "kind": "unconfirmed", "side": "matched pair",
            "displacement_m": 0.157, "vertical_m": 0.002,
            "before_xyz": list(b), "after_xyz": list(a),
            "frames_before": 6, "frames_after": 9, "reason": reason}


AMBIG = ("both spots look empty: one small move or a removal plus an appearance, "
         "which the gate cannot tell apart")


def test_vanish_label_splits_an_ambiguous_pair_into_removal_and_appearance():
    gone = {"cls": "mug", "kind": "removed", "side": "before only",
            "before_xyz": [0.49, 0.45, 0.03], "after_xyz": None,
            "frames_before": 4, "frames_after": 0}
    out = do.fold_records(do.split_on_vanish([_pair(AMBIG), gone]))
    kinds = sorted(r["kind"] for r in out)
    assert kinds == ["added", "removed"], out
    removed = next(r for r in out if r["kind"] == "removed")
    added = next(r for r in out if r["kind"] == "added")
    assert removed["before_xyz"] == [0.49, 0.45, 0.03] and removed["after_xyz"] is None
    assert added["after_xyz"] == [0.35, 0.53, 0.03] and added["before_xyz"] is None


def test_ambiguous_pair_stays_unconfirmed_without_a_vanished_label():
    # no co-located record at all: nothing decides the ambiguity, keep the honest verdict
    assert do.split_on_vanish([_pair(AMBIG)])[0]["kind"] == "unconfirmed"
    # a co-located label that is still seen after does not prove anything left
    still = {"cls": "mug", "kind": "stayed", "side": "before only",
             "before_xyz": [0.49, 0.45, 0.03], "after_xyz": None,
             "frames_before": 4, "frames_after": 3}
    assert do.split_on_vanish([_pair(AMBIG), still])[0]["kind"] == "unconfirmed"


def test_duplicate_label_folds_into_the_moved_object():
    # the same box seen as 'book' and 'box' must not become a move plus a phantom addition
    moved = {"cls": "book", "kind": "moved", "side": "matched pair",
             "displacement_m": 0.665, "vertical_m": -0.01,
             "before_xyz": [0.35, 0.40, 0.02], "after_xyz": [-0.31, 0.44, 0.01],
             "frames_before": 4, "frames_after": 16}
    dup = {"cls": "box", "kind": "added", "side": "after only",
           "before_xyz": None, "after_xyz": [-0.31, 0.44, 0.01],
           "frames_before": 0, "frames_after": 14}
    out = do.fold_records([moved, dup])
    assert len(out) == 1 and out[0]["kind"] == "moved" and out[0]["cls"] == "book"
    assert out[0]["labels"] == ["box"], out


def test_two_labels_at_one_spot_merge_into_one_addition():
    a = {"cls": "chair", "kind": "added", "side": "after only",
         "before_xyz": None, "after_xyz": [0.16, 0.33, 0.01],
         "frames_before": 0, "frames_after": 6}
    b = {"cls": "mouse", "kind": "added", "side": "after only",
         "before_xyz": None, "after_xyz": [0.11, 0.38, 0.01],
         "frames_before": 0, "frames_after": 3}
    out = do.fold_records([a, b])
    assert len(out) == 1 and out[0]["kind"] == "added"
    assert out[0]["cls"] == "chair" and out[0]["labels"] == ["mouse"], out


def test_distinct_objects_are_left_alone():
    # 0.23 m apart: two real neighbours (an added box next to an added pitcher)
    a = {"cls": "box", "kind": "added", "side": "after only",
         "before_xyz": None, "after_xyz": [0.12, 0.57, 0.03], "frames_before": 0, "frames_after": 14}
    b = {"cls": "pitcher", "kind": "added", "side": "after only",
         "before_xyz": None, "after_xyz": [0.35, 0.53, 0.03], "frames_before": 0, "frames_after": 9}
    out = do.fold_records([a, b])
    assert len(out) == 2, out


def test_scene_structure_is_not_an_object_change():
    desk = {"cls": "desk", "kind": "added", "side": "after only",
            "before_xyz": None, "after_xyz": [-0.11, 0.36, 0.01],
            "frames_before": 0, "frames_after": 5}
    out = do.demote_scene_structure([desk])
    assert out[0]["kind"] == "unconfirmed" and "scene structure" in out[0]["reason"], out
    # a real tabletop class passes through untouched
    mug = {"cls": "mug", "kind": "added", "side": "after only",
           "before_xyz": None, "after_xyz": [0.12, 0.57, 0.03],
           "frames_before": 0, "frames_after": 14}
    assert do.demote_scene_structure([mug])[0]["kind"] == "added"


def _votes(e, o, u):
    return {"status": "empty" if o == 0 and e else "unknown", "empty_samples": e,
            "occupied_samples": o, "unknown_samples": u, "samples": e + o + u}


def test_one_contrary_hit_is_not_proof_of_absence():
    # the author's control false alarms all carry 1-2 occupied samples in an 'empty' spot
    r = {"cls": "bed", "kind": "moved", "side": "matched pair", "displacement_m": 1.0,
         "old_spot": _votes(6, 1, 2), "new_spot": _votes(7, 0, 2)}
    out = do.require_strong_evidence([r])
    assert out[0]["kind"] == "unconfirmed" and "too weak" in out[0]["reason"], out


def test_a_mostly_unseen_spot_is_not_declared_empty():
    r = {"cls": "book", "kind": "added", "side": "after only",
         "spot_in_before": _votes(3, 0, 6)}
    out = do.require_strong_evidence([r])
    assert out[0]["kind"] == "unconfirmed" and "unseen" in out[0]["reason"], out


def test_strong_spot_evidence_confirms():
    # the shape of every measured ObChange true positive: 8-9 empty, 0 contrary, <=1 unseen
    moved = {"cls": "book", "kind": "moved", "side": "matched pair",
             "old_spot": _votes(9, 0, 0), "new_spot": _votes(9, 0, 0)}
    added = {"cls": "mug", "kind": "added", "side": "after only",
             "spot_in_before": _votes(8, 0, 1)}
    removed = {"cls": "cup", "kind": "removed", "side": "before only",
               "spot_in_after": _votes(9, 0, 0)}
    out = do.require_strong_evidence([moved, added, removed])
    assert [r["kind"] for r in out] == ["moved", "added", "removed"], out


def test_weak_split_half_stays_unconfirmed():
    # a split's appearance half must clear the same bar as any other addition
    r = {"cls": "chair", "kind": "added", "side": "after only", "split_from_pair": True,
         "spot_in_before": _votes(6, 0, 3)}
    out = do.require_strong_evidence([r])
    assert out[0]["kind"] == "unconfirmed", out


def _region(kind="added", cent=(0.93, -0.40, 0.05), ext=(0.96, 0.43, 0.22), vol=19.4):
    return {"id": 0, "kind": kind, "volume_l": vol, "obb_extents_m": list(ext),
            "centroid_before": None if kind == "added" else list(cent),
            "centroid_after": list(cent) if kind == "added" else None}


TO_FLOOR_ID = lambda p: list(p)


def test_candidate_inside_a_voxel_region_is_confirmed_by_two_methods():
    bag = {"cls": "bag", "kind": "unconfirmed", "side": "after only",
           "before_xyz": None, "after_xyz": [0.88, 0.01, 0.16], "frames_before": 0, "frames_after": 3}
    out = do.corroborate([bag], [_region()], TO_FLOOR_ID)
    assert out[0]["kind"] == "added" and "independent voxel" in out[0]["reason"], out


def test_one_region_is_one_physical_change_labels_fold_in():
    bag = {"cls": "bag", "kind": "unconfirmed", "side": "after only",
           "before_xyz": None, "after_xyz": [0.88, 0.01, 0.16], "frames_before": 0, "frames_after": 3}
    cup = {"cls": "cup", "kind": "unconfirmed", "side": "after only",
           "before_xyz": None, "after_xyz": [0.96, -0.42, 0.19], "frames_before": 0, "frames_after": 3}
    fridge = {"cls": "refrigerator", "kind": "unconfirmed", "side": "after only",
              "before_xyz": None, "after_xyz": [0.78, -0.13, -0.05], "frames_before": 0, "frames_after": 6}
    out = do.corroborate([bag, cup, fridge], [_region()], TO_FLOOR_ID)
    assert len([r for r in out if r["kind"] == "added"]) == 1, out
    one = next(r for r in out if r["kind"] == "added")
    assert one["cls"] == "refrigerator" and sorted(one["labels"]) == ["bag", "cup"], one


def test_corroboration_needs_a_matching_region_kind():
    # an after-only candidate is not a removal, no matter where the region sits
    bag = {"cls": "bag", "kind": "unconfirmed", "side": "after only",
           "before_xyz": None, "after_xyz": [0.88, 0.01, 0.16], "frames_before": 0, "frames_after": 3}
    out = do.corroborate([bag], [_region(kind="removed")], TO_FLOOR_ID)
    assert out[0]["kind"] == "unconfirmed", out


def test_scene_structure_is_not_corroborated_into_existence():
    desk = {"cls": "desk", "kind": "unconfirmed", "side": "after only",
            "before_xyz": None, "after_xyz": [0.9, 0.0, 0.1], "frames_before": 0, "frames_after": 5}
    out = do.corroborate([desk], [_region()], TO_FLOOR_ID)
    assert out[0]["kind"] == "unconfirmed", out


def test_no_regions_no_promotions():
    bag = {"cls": "bag", "kind": "unconfirmed", "side": "after only",
           "before_xyz": None, "after_xyz": [0.88, 0.01, 0.16], "frames_before": 0, "frames_after": 3}
    assert do.corroborate([bag], [], TO_FLOOR_ID)[0]["kind"] == "unconfirmed"


def test_objects_md_states_the_corroboration_not_a_faked_spot_verdict():
    rec = {"cls": "bag", "kind": "added", "side": "after only",
           "before_xyz": None, "after_xyz": [0.88, 0.01, 0.16], "frames_before": 0, "frames_after": 3,
           "reason": "spot evidence alone is too weak, but an independent voxel change region agrees"}
    text = do.describe_objects([rec])
    assert "added bag: seen at" in text and "independent voxel" in text
    assert "the spot is empty in the before capture" not in text, text


def test_frame_extraction_replaces_stale_frames():
    # leftover frames from an earlier extraction must not pollute the next run's sample
    import tempfile
    sys.path.insert(0, str(ROOT / "scripts"))
    import extract_frames as ef
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        (out / "frame_0000.png").write_bytes(b"old")
        (out / "frame_0001.png").write_bytes(b"old")
        (out / "manifest.json").write_text("{}")
        (out / "keep_me.txt").write_text("unrelated")
        ef.clear_frames(out)
        assert not list(out.glob("frame_*.png")) and not (out / "manifest.json").exists()
        assert (out / "keep_me.txt").exists()   # only its own artifacts are touched


def test_capture_noise_comes_from_the_band_or_the_run_itself():
    # run_all passes 1.5x the control noise as --verify-pairs; 0 means "measure this run's stayed pairs"
    stayed = [{"kind": "stayed", "displacement_m": 0.21}, {"kind": "stayed", "displacement_m": 0.05}]
    assert abs(do.capture_noise(stayed, 0.3135) - 0.209) < 1e-6
    assert abs(do.capture_noise(stayed, 0.0) - 0.21) < 1e-6
    assert do.capture_noise([], 0.0) == 0.0


def _confirmed_removed():
    return [{"cls": "pillow", "kind": "removed", "side": "before only",
             "before_xyz": [-0.5, -0.2, 0.2], "after_xyz": None,
             "frames_before": 3, "frames_after": 0,
             "spot_in_after": {"status": "empty", "empty_samples": 7,
                              "occupied_samples": 0, "unknown_samples": 2, "samples": 9}}]


def test_spot_verdict_is_not_proof_above_the_resolution():
    # measured failure: the no-change control itself produced this 'removed pillow' at 0.21 m noise
    out = do.require_provable_evidence(_confirmed_removed(), noise_m=0.21, rise_tol=0.03)
    assert out[0]["kind"] == "unconfirmed", out
    assert "reconstruction noise" in out[0]["reason"] and "7 empty, 0 contrary, 2 unseen" in out[0]["reason"]


def test_spot_proof_stands_below_the_resolution():
    # ObChange's control drifts 0.01 m: there the spot verdict is proof (measured precision 1.00)
    out = do.require_provable_evidence(_confirmed_removed(), noise_m=0.01, rise_tol=0.03)
    assert out[0]["kind"] == "removed", out


def test_corroboration_re_promotes_after_the_noise_gate():
    # the noise gate demotes to candidate; an independent voxel region of the same kind confirms
    rec = _confirmed_removed()[0]
    rec["cls"] = "bag"
    rec["kind"] = "added"
    rec["side"] = "after only"
    rec["before_xyz"], rec["after_xyz"] = None, [0.8, -0.35, 0.01]
    rec["frames_before"], rec["frames_after"] = 0, 7
    rec["spot_in_before"] = rec.pop("spot_in_after")
    out = do.require_provable_evidence([rec], noise_m=0.21, rise_tol=0.03)
    assert out[0]["kind"] == "unconfirmed"
    regions = [{"id": 0, "kind": "added", "volume_l": 19.4,
                "centroid_after": [0.8, -0.35, 0.0], "obb_extents_m": [0.5, 0.4, 0.3]}]
    out = do.corroborate(out, regions, to_floor=lambda xyz: np.asarray(xyz, dtype=float))
    assert out[0]["kind"] == "added", out
    assert "independent voxel change region" in out[0]["reason"]


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"PASS {len(tests)} tests")
