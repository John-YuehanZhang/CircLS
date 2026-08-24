"""Rotation triggers of the sequential-PPM planner (auto_rotate, rule table v3).

Triggers, all realised by REAL protocols (rotate_90 = SWAP network,
litinski = 5-step deformation) — except birth reorientation, which is a FREE
declaration flip for targets first-used at the step (no protocol, no ticks)
and is always preferred when it routes:

* k-parity repair (rotate_90, flips the conjugation bit) — covered by
  tests/test_sequential_ppm_ls.py::test_role_switch_sequence_builds_full_distance;
* blocked face (litinski, keeps the conjugation bit): the measured logical must
  run PARALLEL to the seam, so a patch whose X̄ faces the wrong way and whose
  legal faces are unusable must rotate orientation-only;
* bus-length saving >= rotate_saving_threshold (litinski), default threshold 1.
"""
import contextlib
import io

import pytest

from circls.core.sequential_ppm_ls import PPMStep, SequentialPPMExperiment
from circls.core.multi_patch_coupler import (
    BentLayoutError)
from circls.core.routed_multi_patch_ls import PatchSpec, origin_of
from lightstim.noise.config import NoiseConfig
import circls.core.multi_patch_coupler as _mpc

pytestmark = pytest.mark.smoke

D = 3
NP = NoiseConfig(p_1q=1e-3, p_2q=1e-3, p_meas=1e-3, p_reset=1e-3, p_idle=1e-3)


def _spec(nm, a, b, o):
    return PatchSpec(nm, origin_of(a, b, D, seam=True), D, o)


def _build(exp):
    with contextlib.redirect_stdout(io.StringIO()):
        return exp.build()


def _check(c, d=D):
    det, obs = c.compile_detector_sampler(seed=0).sample(
        512, separate_observables=True)
    assert not det.any(), "detector fired at p=0"
    assert not obs.any(), "observable not deterministic at p=0"


def _dist(exp):
    noisy = exp.builder.build_noisy_circuit(noise_params=NP,
                                            noise_model='circuit_level')
    noisy.detector_error_model(decompose_errors=True)
    return len(noisy.shortest_graphlike_error())


def test_blocked_face_inserts_litinski(monkeypatch):
    # rotation-machinery coverage: the seam-table dispatch would
    # legally solve this type-split with a WALL (zero rotations);
    # disable it here so the test keeps exercising the rotation path
    monkeypatch.setattr(_mpc, 'TABLE_WALL_DISPATCH', False)
    # Q (X̄ horizontal) must join P through the E/W cell (2,1): measuring X
    # through a vertical seam needs X̄ VERTICAL, so Q's orientation is wrong and
    # its conjugation bit must NOT change — at step 1 the only legal move is one
    # litinski on Q (bar cell (1,0) is free; the planner's step-1 candidate list
    # is a singleton).  Q is PRE-REGISTERED by an anchor step (a first-use Q
    # would take the free birth reorientation instead); flipping first-use P
    # does not help, so no birth flip fires.
    px = [_spec("Q", 1, 1, "X_horizontal"),
          _spec("P", 3, 1, "X_vertical"),
          # A anchors X_vertical: its E/W anchor seams are hook-safe; an
          # X_horizontal A above Q would seam A's SOUTH face (the unsafe
          # table cell) and rule ② would insert an extra litinski on A
          _spec("A", 1, 3, "X_vertical")]
    seq = [PPMStep([("Q", "X"), ("A", "X")]),
           PPMStep([("Q", "X"), ("P", "X")], route=[(2, 1)])]
    exp = SequentialPPMExperiment(
        px, seq, initial_states={"Q": "X", "P": "X", "A": "X"},
        final_measure_states={"Q": "X", "P": "X", "A": "X"},
        rounds=D, rounds_init=1, auto_rotate=True)
    c = _build(exp)
    assert exp.birth_reorientations == []
    # since the ±x transpose variants (2026-08-02) an X̄-vertical patch can
    # litinski too: at step 0 rotating A wins the candidate search (1-cell
    # corridor + threshold 1 beats the longer un-rotated route), so the plan
    # front-loads one litinski on A before the forced step-1 litinski on Q
    assert exp.rotation_log == [(0, "A", "litinski"), (1, "Q", "litinski")]
    _check(c)
    assert _dist(exp) == D


def test_saving_threshold_default_rotates():
    # Q (X̄ horizontal) and P (X̄ vertical) at distance: the un-rotated layout
    # routes with a 5-cell snaking corridor, the rotated one with a 3-cell
    # straight bus (probed). Default threshold 1 -> the 2-cell saving rotates.
    px = [_spec("Q", 1, 1, "X_horizontal"),
          _spec("P", 5, 1, "X_vertical")]
    seq = [PPMStep([("Q", "X"), ("P", "X")])]
    exp = SequentialPPMExperiment(
        px, seq, initial_states={"Q": "X", "P": "X"},
        final_measure_states={"Q": "X", "P": "X"},
        rounds=D, rounds_init=1, auto_rotate=True)
    c = _build(exp)
    assert exp.rotation_log == [(0, "Q", "litinski")]
    _check(c)
    assert _dist(exp) == D


def test_saving_threshold_high_keeps_long_bus():
    # Same layout with an unreachable threshold: no rotation, the 5-cell
    # snaking corridor is used instead, and the circuit still verifies.
    px = [_spec("Q", 1, 1, "X_horizontal"),
          _spec("P", 5, 1, "X_vertical")]
    seq = [PPMStep([("Q", "X"), ("P", "X")])]
    exp = SequentialPPMExperiment(
        px, seq, initial_states={"Q": "X", "P": "X"},
        final_measure_states={"Q": "X", "P": "X"},
        rounds=D, rounds_init=1, auto_rotate=True,
        rotate_saving_threshold=99)
    c = _build(exp)
    assert exp.rotation_log == []
    _check(c)
    assert _dist(exp) == D


def test_blocked_face_inserts_litinski_d5(monkeypatch):
    # rotation-machinery coverage: the seam-table dispatch would
    # legally solve this type-split with a WALL (zero rotations);
    # disable it here so the test keeps exercising the rotation path
    monkeypatch.setattr(_mpc, 'TABLE_WALL_DISPATCH', False)
    # d=5 variant of the blocked-face trigger (slow; runs only on request).
    d = 5
    px = [PatchSpec("Q", origin_of(1, 1, d, seam=True), d, "X_horizontal"),
          PatchSpec("P", origin_of(3, 1, d, seam=True), d, "X_vertical"),
          PatchSpec("A", origin_of(1, 3, d, seam=True), d, "X_vertical")]
    seq = [PPMStep([("Q", "X"), ("A", "X")]),
           PPMStep([("Q", "X"), ("P", "X")], route=[(2, 1)])]
    exp = SequentialPPMExperiment(
        px, seq, initial_states={"Q": "X", "P": "X", "A": "X"},
        final_measure_states={"Q": "X", "P": "X", "A": "X"},
        rounds=d, rounds_init=1, auto_rotate=True)
    c = _build(exp)
    # step-0 A litinski = the ±x-variant optimum, same as the d=3 test above
    assert exp.rotation_log == [(0, "A", "litinski"), (1, "Q", "litinski")]
    _check(c, d)
    noisy = exp.builder.build_noisy_circuit(noise_params=NP,
                                            noise_model='circuit_level')
    noisy.detector_error_model(decompose_errors=True)
    assert len(noisy.shortest_graphlike_error()) == d


def test_rotation_kind_litinski_matches_auto_on_blocked(monkeypatch):
    # rotation-machinery coverage: the seam-table dispatch would
    # legally solve this type-split with a WALL (zero rotations);
    # disable it here so the test keeps exercising the rotation path
    monkeypatch.setattr(_mpc, 'TABLE_WALL_DISPATCH', False)
    # forcing rotation_kind='litinski' on the blocked-face case picks the same
    # plan the auto mode does (step-0 A litinski + forced step-1 Q litinski)
    px = [_spec("Q", 1, 1, "X_horizontal"),
          _spec("P", 3, 1, "X_vertical"),
          # A anchors X_vertical: its E/W anchor seams are hook-safe; an
          # X_horizontal A above Q would seam A's SOUTH face (the unsafe
          # table cell) and rule ② would insert an extra litinski on A
          _spec("A", 1, 3, "X_vertical")]
    seq = [PPMStep([("Q", "X"), ("A", "X")]),
           PPMStep([("Q", "X"), ("P", "X")], route=[(2, 1)])]
    exp = SequentialPPMExperiment(
        px, seq, initial_states={"Q": "X", "P": "X", "A": "X"},
        final_measure_states={"Q": "X", "P": "X", "A": "X"},
        rounds=D, rounds_init=1, auto_rotate=True, rotation_kind='litinski')
    c = _build(exp)
    assert exp.rotation_log == [(0, "A", "litinski"), (1, "Q", "litinski")]
    _check(c)
    assert _dist(exp) == D


def test_strict_births_first_use_pair():
    # STRICT BIRTHS (design decision 2026-07-31): first-use orientations are
    # NOT free knobs — declared correctly the pair builds with zero
    # rotations; declared wrong with rotate_90-only (which cannot reach
    # the flipped-standard state) the planner fails loudly.
    seq = [PPMStep([("Q", "X"), ("P", "Z")], route=[(2, 1)])]
    px_ok = [_spec("Q", 1, 1, "X_vertical"),
             _spec("P", 3, 1, "X_horizontal")]
    exp = SequentialPPMExperiment(
        px_ok, seq, initial_states={"Q": "X", "P": "Z"},
        final_measure_states={"Q": "X", "P": "Z"},
        rounds=D, rounds_init=1, auto_rotate=True, rotation_kind='rotate_90')
    c = _build(exp)
    assert exp.rotation_log == []
    assert exp.birth_reorientations == []
    _check(c)
    assert _dist(exp) == D
    px_bad = [_spec("Q", 1, 1, "X_horizontal"),
              _spec("P", 3, 1, "X_vertical")]
    exp2 = SequentialPPMExperiment(
        px_bad, seq, initial_states={"Q": "X", "P": "Z"},
        final_measure_states={"Q": "X", "P": "Z"},
        rounds=D, rounds_init=1, auto_rotate=True, rotation_kind='rotate_90')
    with pytest.raises(BentLayoutError, match="no feasible"):
        _build(exp2)


def test_rotation_kind_rotate90_group_registered_pair(monkeypatch):
    # rotation-machinery coverage: the seam-table dispatch would
    # legally solve this type-split with a WALL (zero rotations);
    # disable it here so the test keeps exercising the rotation path
    monkeypatch.setattr(_mpc, 'TABLE_WALL_DISPATCH', False)
    # a PRE-REGISTERED same-letter pair blocked through the E/W cell: both
    # X-horizontal, X(x)X needs vertical seams — the rotate_90 GROUP serves
    # (both rotate, colours swap together, the corridor is the standard
    # construction under a global colour swap)
    px = [_spec("Q", 1, 1, "X_horizontal"), _spec("P", 3, 1, "X_horizontal"),
          _spec("A", 1, 3, "X_vertical"), _spec("B", 3, 3, "X_vertical")]
    seq = [PPMStep([("Q", "X"), ("A", "X")]),
           PPMStep([("B", "X"), ("P", "X")]),
           PPMStep([("Q", "X"), ("P", "X")], route=[(2, 1)])]
    exp = SequentialPPMExperiment(
        px, seq,
        initial_states={"Q": "X", "P": "X", "A": "X", "B": "X"},
        final_measure_states={"Q": "X", "P": "X", "A": "X", "B": "X"},
        rounds=D, rounds_init=1, auto_rotate=True, rotation_kind='rotate_90')
    c = _build(exp)
    assert exp.birth_reorientations == []
    assert exp.rotation_log == [(2, "Q", "rotate_90"), (2, "P", "rotate_90")]
    _check(c)


def test_litinski_only_colour_switch_served_by_constructs():
    # a mid-sequence letter/colour mismatch used to demand rotate_90 (the
    # only colour-swapping rotation) and raised under litinski-only.
    # STRICT BIRTHS + the full seam table serve it with CONSTRUCTS (#6
    # wall / #7 column) instead — the sequence builds, no rotation at all.
    px = [_spec("Q", 0, 0, "X_vertical"),
          _spec("A", 2, 0, "X_horizontal"),
          _spec("B", 4, 0, "X_horizontal")]
    seq = [PPMStep([("Q", "X"), ("A", "Z")]),
           PPMStep([("B", "X"), ("A", "Z")]),
           PPMStep([("Q", "Z"), ("B", "X")])]
    exp = SequentialPPMExperiment(
        px, seq, initial_states={"Q": "X", "A": "Z", "B": "X"},
        final_measure_states={"Q": "Z", "A": "Z", "B": "X"},
        rounds=D, rounds_init=1, auto_rotate=True, rotation_kind='litinski')
    c = _build(exp)
    # the cost law picks litinskis for the blocked faces; since the
    # chirality-law wall envelope (2026-08-02) A's native-#6 wall hosts
    # directly on the X̄-horizontal target, so only B rotates — the build
    # succeeds where the old role-switch machinery raised
    assert exp.rotation_log == [(1, "B", "litinski")]
    _check(c)


def test_rotation_kind_rotate90_group_solves_pure_type_blocked():
    # BOTH patches X̄ horizontal, pure-type XX forced through the E/W cell:
    # after the all-targets rotate_90 group both are colour-swapped (weight-2
    # positions unchanged — still a plain-merge row of the rule table), and
    # the corridor they need is the STANDARD one under a global colour swap.
    # _route_result routes/verifies the colour-conjugate pre-image (X̄
    # horizontal, ZZ) and emits conjugate_layout of it.
    px = [_spec("Q", 1, 1, "X_horizontal"),
          _spec("P", 3, 1, "X_horizontal")]
    seq = [PPMStep([("Q", "X"), ("P", "X")], route=[(2, 1)])]
    exp = SequentialPPMExperiment(
        px, seq, initial_states={"Q": "X", "P": "X"},
        final_measure_states={"Q": "X", "P": "X"},
        rounds=D, rounds_init=1, auto_rotate=True, rotation_kind='rotate_90')
    c = _build(exp)
    # STRICT BIRTHS: no free flips — the rotate_90 group is now two REAL
    # rotations, adopted because 2-cell saving >= threshold x 2
    assert exp.birth_reorientations == []
    assert exp.rotation_log == [(0, "Q", "rotate_90"), (0, "P", "rotate_90")]
    _check(c)
    assert _dist(exp) == D


def test_conjugate_layout_symmetry():
    # colour swapping is an exact relabeling: double swap is the identity and
    # a verified layout stays verified after one swap
    from circls.core.multi_patch_coupler import (
        route_and_build, conjugate_layout)
    px = [_spec("Q", 1, 1, "X_horizontal"),
          _spec("P", 3, 1, "X_horizontal")]
    r = route_and_build(px, [("Q", "Z"), ("P", "Z")], seam=True,
                        route=[(2, 1)])
    assert r.status == 'ok'
    sw = conjugate_layout(r.layout)
    back = conjugate_layout(sw)
    assert back.checks == r.layout.checks
    assert back.logicals == r.layout.logicals
    assert back.domains == r.layout.domains
    assert back.bus == r.layout.bus
    assert all(sw.verify().values())
