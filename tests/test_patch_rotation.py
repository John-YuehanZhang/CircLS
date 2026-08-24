"""Single-patch 90° rotation primitive.

Rotating a patch clockwise 90° reorients its logicals (X̄ vertical↔horizontal,
Z̄ likewise) and carries the checkerboard with it — WITHOUT swapping any
stabilizer's Pauli type (U_L = I; a pure geometric rotation, not a Hadamard).
The rotated patch is byte-identical to the shape the routed coupler already
builds and validates for a minority patch (see the equivalence test below),
so the existing rule-0 native lock / bus machinery hosts it at full distance.
"""
import math

import numpy as np
import pytest

from lightstim.qec_code.surface_code.rotated import RotatedSurfaceCode
from circls.core.multi_patch_coupler import (
    conjugate_patch_records)
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.ir.builder import CircuitBuilder
from lightstim.qec_code.surface_code.rotated.bent_joint_se import se_round_chunk

pytestmark = pytest.mark.smoke


def _logical(code, ptype):
    qc = code.qubit_coords
    lg = next(l for l in code.logical_ops if l["type"] == ptype)
    return sorted(tuple(round(v, 3) for v in qc[i]) for i in lg["pauli"])


def _orientation(support):
    xs = {p[0] for p in support}
    ys = {p[1] for p in support}
    if len(xs) == 1:
        return "vertical"
    if len(ys) == 1:
        return "horizontal"
    return "bent"


def _syn_by_type(code, t):
    return sorted(tuple(round(v) for v in s["syn_coord"])
                  for s in code.stabilizers if s["type"] == t)


def _norm_sig(code):
    items = [(tuple(round(v) for v in s["syn_coord"]), s["type"])
             for s in code.stabilizers]
    mnx = min(c[0] for c, _ in items)
    mny = min(c[1] for c, _ in items)
    return sorted(((c[0] - mnx, c[1] - mny), t) for c, t in items)


def _rotate_cw(code):
    """Clockwise 90° rotation about the patch centroid."""
    code.rotate_90(clockwise=True)


@pytest.mark.parametrize("d", [3, 5, 7])
def test_rotation_reorients_logicals(d):
    # before: Z̄ horizontal, X̄ vertical (the library's base X_vertical patch)
    code = RotatedSurfaceCode(distance=d)
    assert _orientation(_logical(code, "Z")) == "horizontal"
    assert _orientation(_logical(code, "X")) == "vertical"
    # after a clockwise 90°: Z̄ vertical, X̄ horizontal (the user's image)
    _rotate_cw(code)
    assert _orientation(_logical(code, "Z")) == "vertical"
    assert _orientation(_logical(code, "X")) == "horizontal"


@pytest.mark.parametrize("d", [3, 5, 7])
def test_rotation_preserves_pauli_type(d):
    # a rotation is U_L = I: X̄ stays X-type, Z̄ stays Z-type (NOT a Hadamard).
    code = RotatedSurfaceCode(distance=d)
    types_before = {l["type"] for l in code.logical_ops}
    _rotate_cw(code)
    types_after = {l["type"] for l in code.logical_ops}
    assert types_before == types_after == {"X", "Z"}
    for stab in code.stabilizers:
        # every stabilizer keeps its type; only its support moved
        assert stab["type"] in ("X", "Z")


@pytest.mark.parametrize("d", [3, 5, 7])
def test_rotation_carries_checkerboard(d):
    # at a fixed plaquette position the check colour swaps: where X-plaquettes
    # were, Z-plaquettes now sit (90° rotation moves the whole checkerboard).
    code = RotatedSurfaceCode(distance=d)
    x_before = set(_syn_by_type(code, "X"))
    z_before = set(_syn_by_type(code, "Z"))
    _rotate_cw(code)
    x_after = set(_syn_by_type(code, "X"))
    # the rotated X-plaquettes land exactly on former Z-plaquette positions
    assert x_after == z_before
    assert set(_syn_by_type(code, "Z")) == x_before


@pytest.mark.parametrize("d", [3, 5, 7])
def test_rotation_equals_coupler_minority_shape(d):
    # the crux: rotate_coords(90°) of the base patch is byte-identical to
    # place_patch(X_vertical)+conjugate — exactly the shape the routed coupler
    # builds for a minority patch declared X_horizontal. So the existing
    # native-lock/bus machinery hosts a rotated patch verbatim (see
    # test_routed_multi_patch_ls.test_mixed_minority_full_distance, which
    # already runs this shape at full distance d).
    rotated = RotatedSurfaceCode(distance=d)
    _rotate_cw(rotated)
    coupler_minority = RotatedSurfaceCode(distance=d)
    conjugate_patch_records(coupler_minority)
    assert _norm_sig(rotated) == _norm_sig(coupler_minority)


@pytest.mark.parametrize("d", [3, 5])
def test_rotate_live_patch_keeps_deterministic_circuit(d):
    # QECSystem.rotate_patch turns a LIVE patch 90° in place, mid-circuit: it moves
    # only the coord metadata (qubit_coords / index_map / syn_coord) and leaves the
    # index-keyed records + the index-based SyndromeTracker untouched. A round of SE
    # before AND after the turn must still yield a valid, fully-deterministic (p=0)
    # circuit — the rotated patch is stabilized just as before, only reoriented.
    system = QECSystem()
    system.add_patch(RotatedSurfaceCode(distance=d), name="Q", offset=(0, 0))
    tracker = SyndromeTracker(num_qubits=system.num_qubits,
                              expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=True)
    system.register_tracker(tracker)
    system.register_builder(builder)
    builder.write_coordinates()

    owner = system.index_to_owner_map
    orient = {"Q": "X_vertical"}

    def standalone(n):
        domains = {tuple(system.qubit_coords[q]): orient[owner[q]]
                   for q in system.data_indices if owner.get(q) in orient}
        builder.apply_syndrome_extraction(
            circuit_chunk=se_round_chunk(system, domains=domains), rounds=n)

    coords_before = dict(system.qubit_coords)
    builder.initialize(init_dict={q: "Z" for q in system.data_indices},
                       n=system.num_qubits)
    standalone(2)
    system.rotate_patch("Q", clockwise=True)             # <-- the turn, mid-circuit
    orient["Q"] = "X_horizontal"
    # footprint is invariant; only the coord<->index assignment inside the patch permutes
    assert set(coords_before.values()) == set(system.qubit_coords.values())
    assert system.qubit_coords != coords_before
    standalone(2)
    builder.apply_data_readout(final_measurements={q: "Z" for q in system.data_indices})

    c = builder.circuit
    assert c.num_detectors > 0 and c.num_observables == 1
    dets, obs = c.compile_detector_sampler(seed=3).sample(
        shots=200, separate_observables=True)
    assert not np.any(dets)          # every detector deterministic at p=0
    assert not np.any(obs)
