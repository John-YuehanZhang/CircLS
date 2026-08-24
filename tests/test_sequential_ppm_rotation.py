"""Between-PPM 90° rotation (``SequentialPPMExperiment(auto_rotate=True)``).

A shared patch that is measured through an E/W seam in PPM 0 and through a N/S
seam in PPM 1 needs OPPOSITE orientations for the two seams (the measured
logical must run PARALLEL to each seam).  A single declared orientation cannot
satisfy both, so the un-rotated sequence is a majority<->minority role switch
that the router rejects (perpendicular seam / distance-1 fallback).

``auto_rotate=True`` detects that PPM 1 does not route in the current
orientation but DOES route with the patch rotated, physically rotates the live
patch 90° between the PPMs, and native-locks PPM 1's coupler to the rotated
(conjugate-native) checkerboard.  The joint reaches the FULL code distance d.
"""
import contextlib
import io

import pytest

from circls.core.sequential_ppm_ls import PPMStep, SequentialPPMExperiment
from circls.core.routed_multi_patch_ls import PatchSpec, origin_of
from circls.core.multi_patch_coupler import BentLayoutError
from lightstim.noise.config import NoiseConfig

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from conftest import assert_valid_circuit, assert_noiseless, assert_dem_valid

pytestmark = pytest.mark.smoke

D = 3
NP = NoiseConfig(p_1q=1e-3, p_2q=1e-3, p_meas=1e-3, p_reset=1e-3, p_idle=1e-3)


def _spec(nm, a, b, o):
    return PatchSpec(nm, origin_of(a, b, D, seam=True), D, o)


def _build(exp):
    with contextlib.redirect_stdout(io.StringIO()):
        return exp.build()


def _dist(c):
    c.detector_error_model(decompose_errors=True)   # detectors deterministic
    return len(c.shortest_graphlike_error())


def _geometry():
    # Q (centre), L (west), Bt (below), C (anchor for Bt).  Cells: Q=(2,2),
    # L=(0,2), Bt=(2,0), C=(4,1).
    #   PPM 0  M(Q^Z, L^Z):  Q-L are E/W neighbours -> vertical (E/W) seam;
    #                        measuring Z through an E/W seam needs X_horizontal. Q is
    #                        majority (Z == bus Z).
    #   PPM 1  M(Bt^X, C^X): anchors Bt as declared/registered — with Bt
    #                        first-used in the conflicted step, the birth
    #                        freedoms (colour/orientation) would serve the
    #                        switch with ZERO rotations instead.
    #   PPM 2  M(Q^Z, Bt^X): needs {Z, X} disagree between REGISTERED
    #                        targets -> the colour repair rotates Q
    #                        (rotate_90, colours swap, need flips to X).
    px = [_spec("Q", 2, 2, "X_horizontal"),
          _spec("L", 0, 2, "X_horizontal"),
          _spec("Bt", 2, 0, "X_horizontal"),
          _spec("C", 4, 1, "X_vertical")]
    seq = [PPMStep([("Q", "Z"), ("L", "Z")]),
           PPMStep([("Bt", "X"), ("C", "X")]),
           PPMStep([("Q", "Z"), ("Bt", "X")])]
    # each patch prepared and read out in a single basis -> bare logical observables
    init = {"Q": "X", "L": "Z", "Bt": "X", "C": "X"}
    meas = {"Q": "X", "L": "Z", "Bt": "X", "C": "X"}
    return px, seq, init, meas


def _exp(px, seq, init, meas, **kw):
    return SequentialPPMExperiment(
        px, seq, initial_states=init, final_measure_states=meas,
        rounds=D, rounds_init=1, **kw)


def test_rotation_between_ppms_full_distance():
    px, seq, init, meas = _geometry()
    exp = _exp(px, seq, init, meas, noise_params=NP, auto_rotate=True, rotation_kind='auto')
    c = _build(exp)
    assert exp.rotation_count == 1
    assert exp.rotations == [(2, "Q")]          # Q rotated before PPM 2
    assert c.num_observables == 2
    assert _dist(c) == D                         # full joint distance


def test_rotation_between_ppms_noiseless_valid():
    px, seq, init, meas = _geometry()
    exp = _exp(px, seq, init, meas, noise_params=None, auto_rotate=True, rotation_kind='auto')
    c = _build(exp)
    assert_valid_circuit(c)
    assert_noiseless(c)                          # detectors deterministic at p=0
    assert_dem_valid(c)
    assert c.num_observables == 2
    assert exp.rotation_count == 1


def test_without_rotation_wall_serves():
    # Same sequence, auto_rotate OFF: the mismatch needs a native-#6 wall
    # on an X_horizontal target — hosted since the chirality-law envelope
    # (2026-08-02), so the sequence builds with zero rotations (under
    # 'auto' the planner still prefers the shorter rotate_90 corridor by
    # score; the loud-failure path is covered by
    # test_sequential_ppm_ls.test_boxed_target_without_auto_rotate_raises).
    px, seq, init, meas = _geometry()
    exp = _exp(px, seq, init, meas, noise_params=NP, auto_rotate=False)
    c = _build(exp)
    assert exp.rotation_count == 0
    assert c.num_observables == 2
    assert _dist(c) == D
