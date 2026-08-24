"""A stabilizer's identity includes its ANCILLA (QECSystem registration).

Registration deduplicates stabilizer records on
``(type, support, syn_idx)`` so the same physical check keeps its uid — and
its detector history — when it changes owner.  Dropping the ancilla from that
key aliases two DIFFERENT checks onto one uid: a corridor's weight-2 seam
check and a patch's identical-support boundary lobe live on different
ancillas, and the two registration paths then corrupt each other's records
(``add_patch`` overwrites the live record, ``_apply_patch_geometry`` inherits
the stale one).  Either way one ancilla ends up carrying two ACTIVE checks,
which the compact-7 validator reports as

    fixed compact-7 schedule: slot 4 on qubit (89, 35) is used by both
    X-check@(90, 36) and Z-check@(90, 36)

(seen while building the bv_n70 replay, 2026-08-14).
"""
import pytest

from lightstim.ir.qec_system import QECSystem
from lightstim.ir.coupler import LogicalCouplerPatch
from lightstim.qec_code.surface_code.rotated.code_patch import RotatedSurfaceCode
from lightstim.qec_code.surface_code.rotated.diagonal_se import (
    DiagonalSurfaceCodeExtractionBlock)

_FLIP = {'X': 'Z', 'Z': 'X'}


def _recoloured(code):
    """Colour swap in place (circls.core.multi_patch_coupler.conjugate_patch_records)."""
    for rec in code.stabilizers + code.logical_ops:
        rec['type'] = _FLIP.get(rec['type'], rec['type'])
        rec['pauli'] = {q: _FLIP.get(P, P) for q, P in rec['pauli'].items()}
    code.syndrome_indices_x, code.syndrome_indices_z = (
        code.syndrome_indices_z, code.syndrome_indices_x)
    return code


def _corridor_check(system, name, typ, support, syn_coord):
    """Register a coupler-style patch holding ONE coord-keyed check — the record
    shape a routed coupler emits for a seam check that lands on an ancilla the
    patch construction does not use.  It owns no qubits of its own."""
    cpl = LogicalCouplerPatch(name=name)
    cpl.stabilizers.append({'pauli': {c: typ for c in support},
                            'type': typ, 'syn_coord': syn_coord})
    system.add_patch(cpl, name=name, is_active=False)
    return cpl


def _ancillas_of_active_checks(system):
    out = []
    for uid in sorted(system.active_stabilizer_indices):
        s = system.stabilizers[uid]
        if s.get('syn_idx') is not None:
            out.append(system.qubit_coords[s['syn_idx']])
    return out


def test_coupler_check_does_not_steal_a_live_patch_check():
    """add_patch: a corridor check with the same (type, support) as a LIVE patch
    check but a different ancilla must NOT reuse the patch record."""
    system = QECSystem()
    system.add_patch(RotatedSurfaceCode(distance=3), name='q')
    uid_lobe = next(uid for uid, s in enumerate(system.stabilizers)
                    if s['type'] == 'X' and s['syn_coord'] == (2, 0))
    n_before = len(system.stabilizers)

    # same weight-2 X support, corridor-facing ancilla (2, 2) instead of (2, 0)
    _corridor_check(system, 'cpl', 'X', [(1, 1), (3, 1)], (2, 2))

    lobe = system.stabilizers[uid_lobe]
    assert lobe['patch_name'] == 'q'
    assert lobe['syn_coord'] == (2, 0)          # not moved onto (2, 2)
    assert len(system.stabilizers) == n_before + 1   # corridor got its own uid
    anc = _ancillas_of_active_checks(system)
    assert len(anc) == len(set(anc)), f"two active checks on one ancilla: {anc}"


def test_recoloured_patch_does_not_inherit_a_corridor_ancilla():
    """_apply_patch_geometry (grow/move_corners/shrink): a re-registered check
    must not inherit a stale corridor record living on another ancilla."""
    system = QECSystem()
    system.add_patch(RotatedSurfaceCode(distance=3), name='q')
    # corridor check on the patch's BULK ancilla (2, 2); at this point the patch
    # has no Z check on that pair, so it takes a fresh uid
    _corridor_check(system, 'cpl', 'Z', [(1, 1), (3, 1)], (2, 2))
    stale_uid = len(system.stabilizers) - 1

    # the recoloured geometry HAS a Z weight-2 on {(1,1),(3,1)} — at its own
    # ancilla (2, 0), not at (2, 2)
    system.move_corners('q', _recoloured(RotatedSurfaceCode(distance=3)),
                        offset=(0, 0))

    stale = system.stabilizers[stale_uid]
    assert stale['patch_name'] == 'cpl'
    assert stale_uid not in system.active_stabilizer_indices
    anc = _ancillas_of_active_checks(system)
    assert len(anc) == len(set(anc)), f"two active checks on one ancilla: {anc}"
    DiagonalSurfaceCodeExtractionBlock(system)   # compact-7 validator must pass


def test_same_check_same_ancilla_keeps_its_uid():
    """The dedup contract itself: a deformation that re-registers a check with
    the same type, support AND ancilla keeps the uid (detector continuity)."""
    system = QECSystem()
    system.add_patch(RotatedSurfaceCode(distance=3), name='q')
    bulk = next(uid for uid, s in enumerate(system.stabilizers)
                if s['type'] == 'Z' and s['syn_coord'] == (2, 2))
    support = tuple(sorted(system.stabilizers[bulk]['data_indices']))

    # move_corners with the SAME geometry: every check re-registers unchanged
    system.move_corners('q', RotatedSurfaceCode(distance=3), offset=(0, 0))

    same = [uid for uid, s in enumerate(system.stabilizers)
            if s['type'] == 'Z' and tuple(sorted(s['data_indices'])) == support
            and s['syn_coord'] == (2, 2)]
    assert same == [bulk]
    assert bulk in system.active_stabilizer_indices


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
