"""RoutedMultiPatchLSExperiment — the atomic-operation driven seam-column LS.

Cross-validation of the IR stack (QECSystem + RotatedRoutedMultiPatchCoupler +
CircuitBuilder + SyndromeTracker) against the physics established with the
hand-written phased engine: full code distance d on the state channel, the
conjugate-convention minority registration (no morph), and the automatically
folded teleportation observable (the banked joint outcome m enters the
observable through the tracker's decomposition — no hand-written folding).
"""
import contextlib
import io

import pytest

from lightstim.noise.config import NoiseConfig
from circls.core.routed_multi_patch_ls import (
    RoutedMultiPatchLSExperiment, PatchSpec, origin_of)

pytestmark = pytest.mark.smoke

D = 3
NP = NoiseConfig(p_1q=1e-3, p_2q=1e-3, p_meas=1e-3, p_reset=1e-3, p_idle=1e-3)


def _spec(nm, a, b, o):
    return PatchSpec(nm, origin_of(a, b, D, seam=True), D, o)


def _build(px, IT, init, meas, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        exp = RoutedMultiPatchLSExperiment(px, IT, initial_states=init,
                                           measure_states=meas,
                                           rounds_init=1, rounds=D,
                                           noise_params=NP, **kw)
        c = exp.build()
    return c


def _w(c):
    c.detector_error_model(decompose_errors=True)   # detectors deterministic
    return len(c.shortest_graphlike_error())


def test_state_channel_full_distance():
    c = _build([_spec("Q1", 0, 0, "X_horizontal"),
                _spec("Q2", 2, 0, "X_horizontal")],
               [("Q1", "Z"), ("Q2", "Z")],
               {"Q1": "X", "Q2": "Z"}, {"Q1": "X", "Q2": "Z"})
    assert c.num_observables == 1
    assert _w(c) == D


def test_mixed_minority_full_distance():
    # XZ joint: Q1 is minority -> registered in the conjugate convention
    # (transposed geometry, records typeswapped); the state channel keeps
    # both bare observables at full distance
    c = _build([_spec("Q1", 0, 0, "X_vertical"),
                _spec("Q2", 2, 0, "X_horizontal")],
               [("Q1", "X"), ("Q2", "Z")],
               {"Q1": "X", "Q2": "Z"}, {"Q1": "X", "Q2": "Z"})
    assert c.num_observables == 2
    assert _w(c) == D


def test_teleport_observable_auto_folded():
    # init (X,Z), measure (Z,X): X2's readout alone is random — the tracker
    # folds the banked chain records into the observable automatically
    c = _build([_spec("Q1", 0, 0, "X_vertical"),
                _spec("Q2", 2, 0, "X_vertical")],
               [("Q1", "X"), ("Q2", "X")],
               {"Q1": "X", "Q2": "Z"}, {"Q1": "Z", "Q2": "X"})
    assert c.num_observables == 1
    assert _w(c) == D


def test_t_junction_mixed():
    c = _build([_spec("Q1", 0, 0, "X_vertical"),
                _spec("Q2", 1, 1, "X_vertical"),
                _spec("Q3", 2, 0, "X_horizontal")],
               [("Q1", "X"), ("Q2", "Z"), ("Q3", "Z")],
               {"Q1": "X", "Q2": "Z", "Q3": "Z"},
               {"Q1": "X", "Q2": "Z", "Q3": "Z"})
    assert c.num_observables == 3
    assert _w(c) == D


def test_no_morph_layer_full_distance():
    # the minority patch registers the conjugate-convention construction
    # (typeswapped records) from round 0 and initial/measure bases are
    # literal, so the protocol emits NO transversal-H layer anywhere
    px = [_spec("Q1", 0, 0, "X_vertical"),
          _spec("Q2", 2, 0, "X_horizontal")]
    IT = [("Q1", "X"), ("Q2", "Z")]
    for init, meas, n_obs in [
            ({"Q1": "X", "Q2": "X"}, {"Q1": "Z", "Q2": "Z"}, 1),   # teleport
            ({"Q1": "X", "Q2": "Z"}, {"Q1": "X", "Q2": "Z"}, 2)]:  # state
        c = _build(px, IT, init, meas)
        assert c.num_observables == n_obs
        assert _w(c) == D
        assert not any(inst.name == "H" and len(inst.targets_copy()) == D * D
                       for inst in c.flattened())      # no transversal-H morph


def test_t_junction_teleport_folded():
    c = _build([_spec("Q1", 0, 1, "X_vertical"),
                _spec("Q2", 4, 1, "X_horizontal"),
                _spec("Q3", 2, 3, "X_vertical"),
                _spec("Q4", 2, 0, "X_vertical")],
               [("Q1", "X"), ("Q2", "Z"), ("Q3", "Z"), ("Q4", "Z")],
               {"Q1": "X", "Q2": "X", "Q3": "X", "Q4": "X"},
               {"Q1": "Z", "Q2": "Z", "Q3": "Z", "Q4": "Z"},
               route=[(1, 1), (2, 1), (3, 1), (2, 2)])
    assert c.num_observables == 1
    assert _w(c) == D
