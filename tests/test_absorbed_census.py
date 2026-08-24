"""Absorbed-DOF census: one ledger, derived count, logical equivalence.

absorbed_ops is the single census ledger — every path that absorbs a
logical DOF records the OPERATOR; the count is always derived as the
ledger's own GF(2) rank, with logical-equivalence deduplication at
INSERTION time (record_absorbed_op reduces against [stabilizers ∪
ledger] and skips dependent representatives; the count is deliberately
NOT quotiented by the bank — see num_absorbed_dof).  There is
deliberately no separately maintained integer (the old
`_absorbed_logical_dofs` counter demanded perfect increment/decrement
pairing from every path and drifted silently when one forgot).  The
ledger is OPERATOR-ONLY: absorbed_ops.records is never written — banked
parity retrieval returns with the liveness layer (scope ruling
2026-08-17).  Geometry-independent, like test_tracker_closure_contract.
"""
import numpy as np
import pytest

from lightstim.ir.tracker import SyndromeTracker


def _z_row(n, qubits):
    row = np.zeros(2 * n, dtype=np.uint8)
    for q in qubits:
        row[n + q] = 1
    return row

# NOTE (CircLS port): the upstream file also carries tests for the
# reset-over-banked guard, the partial-resolve fail-loud, and the
# UNMEASURED-sentinel observable guard.  Those are dropped here: the
# first and third are the not-yet-ported defensive-guard batch, and the
# second is deliberately NOT ported (CircLS re-prices half-read
# remainders instead of raising; see the mid-measurement absorb sites).


def test_census_dedups_representatives_at_insertion():
    # Reviewer-mandated regression: two representatives differing by a
    # stabilizer are ONE logical relation, not two.  The dedup happens at
    # insertion: the second representative reduces to zero against the
    # bank + ledger and is skipped.
    n = 4
    tracker = SyndromeTracker(n, 0)
    tracker.stabilizers.matrix = _z_row(n, [0, 1]).reshape(1, -1)
    tracker.stabilizers.records = [[0]]
    rep_a = _z_row(n, [2, 3])
    rep_b = (_z_row(n, [2, 3]) + _z_row(n, [0, 1])) % 2  # rep_a * stabilizer

    assert tracker.record_absorbed_op(rep_a) is True
    assert tracker.record_absorbed_op(rep_b) is False
    assert tracker.absorbed_ops.count == 1
    assert tracker.num_absorbed_dof() == 1


def test_group_member_relation_is_not_banked():
    # An operator already expressed by the current stabilizer rows holds
    # no NEW logical DOF at insertion time and must not enter the ledger.
    n = 4
    tracker = SyndromeTracker(n, 0)
    tracker.stabilizers.matrix = _z_row(n, [0, 1]).reshape(1, -1)
    tracker.stabilizers.records = [[0]]

    assert tracker.record_absorbed_op(_z_row(n, [0, 1])) is False
    assert tracker.num_absorbed_dof() == 0


def test_insertion_reduces_against_the_ledger_too():
    # The reduction basis is [stabilizers ∪ ledger]: a second relation
    # overlapping an already-banked one stores only its new content, and
    # the census still counts two DOFs.
    n = 3
    tracker = SyndromeTracker(n, 0)
    assert tracker.record_absorbed_op(_z_row(n, [0, 1])) is True
    assert tracker.record_absorbed_op(_z_row(n, [0, 2])) is True
    assert (tracker.absorbed_ops.matrix[1] == _z_row(n, [1, 2])).all()
    assert tracker.num_absorbed_dof() == 2


def test_banked_relation_survives_its_own_closure_row():
    # After a merge, the joint's CLOSURE row (same relation, carrying the
    # merge records) legitimately enters the stabilizer bank.  The banked
    # DOF must keep counting — the census is the ledger's own rank, never
    # quotiented by the bank (regression for the measurement-block round
    # over post-PPM state, where the quotient version tripped the alarm).
    n = 4
    tracker = SyndromeTracker(n, 1)
    assert tracker.record_absorbed_op(_z_row(n, [0, 1])) is True
    # the closure row appears in the bank AFTER the relation was banked
    tracker.stabilizers.matrix = _z_row(n, [0, 1]).reshape(1, -1)
    tracker.stabilizers.records = [[3]]

    assert tracker.num_absorbed_dof() == 1
    tracker.validate_logical_count(context="closure row coexists")


def test_alarm_fires_when_a_relation_is_lost():
    n = 4
    tracker = SyndromeTracker(n, 1)  # budget: one logical DOF somewhere
    tracker.absorbed_ops.matrix = _z_row(n, [2, 3]).reshape(1, -1)
    tracker.validate_logical_count(context="healthy state")  # no raise

    # A path that loses the relation without accounting for it must trip
    # the alarm at the next validation — the ledger IS the count, so the
    # two can no longer drift apart silently.
    tracker.absorbed_ops.matrix = np.zeros((0, 2 * n), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="absorbed logical DOFs"):
        tracker.validate_logical_count(context="after losing the relation")


def test_block_absorb_records_the_operator():
    n = 4
    tracker = SyndromeTracker(n, 0)
    consumed = _z_row(n, [1, 2])
    tracker._gauge_logical_vectors = [np.array([1], dtype=np.uint8)]

    tracker._record_measurement_logical_effects(
        set(), old_logicals_current_frame=consumed.reshape(1, -1))

    assert tracker.absorbed_ops.count == 1
    assert (tracker.absorbed_ops.matrix[0] == consumed).all()
    assert tracker.num_absorbed_dof() == 1


def test_block_absorb_skips_surviving_logicals():
    n = 4
    tracker = SyndromeTracker(n, 0)
    tracker._gauge_logical_vectors = [np.array([1], dtype=np.uint8)]

    tracker._record_measurement_logical_effects(
        {0}, old_logicals_current_frame=_z_row(n, [1, 2]).reshape(1, -1))

    assert tracker.absorbed_ops.count == 0
    assert tracker.num_absorbed_dof() == 0


def test_block_absorb_requires_the_frame():
    tracker = SyndromeTracker(4, 0)
    tracker._gauge_logical_vectors = [np.array([1], dtype=np.uint8)]

    with pytest.raises(ValueError, match="cannot be recorded"):
        tracker._record_measurement_logical_effects(set())


def test_corridor_fold_reexpresses_off_the_bus():
    # Corridor readout re-expresses a banked relation off the bus via a
    # stabilizer row: the patch-side representative survives the bus
    # readout (the relation is only defined MOD the stabilizer group).
    import stim
    n = 3
    tracker = SyndromeTracker(n, 0)
    tracker.stabilizers.matrix = _z_row(n, [0, 1]).reshape(1, -1)
    tracker.stabilizers.records = [[7]]
    tracker.absorbed_ops.matrix = _z_row(n, [1]).reshape(1, -1)
    fp = np.vstack([_z_row(n, [1]), _z_row(n, [2])])

    tracker.process_data_measurement(
        stim.Circuit(), fp, {i: (i, 0) for i in range(n)},
        resolve_absorbed=False)

    assert (tracker.absorbed_ops.matrix[0] == _z_row(n, [0])).all()
    assert tracker.num_absorbed_dof() == 1


def test_corridor_residual_folds_against_readout():
    # Bus support the stabilizer group cannot cancel is folded against the
    # readout's own measured Paulis.  (The old code hard-zeroed the
    # columns, silently changing the relation instead of its
    # representative.)
    import stim
    n = 3
    tracker = SyndromeTracker(n, 0)
    tracker.absorbed_ops.matrix = _z_row(n, [0, 2]).reshape(1, -1)
    fp = np.vstack([_z_row(n, [0]), _z_row(n, [1])])   # bus = {0, 1}

    tracker.process_data_measurement(
        stim.Circuit(), fp, {i: (i, 0) for i in range(n)},
        resolve_absorbed=False)

    assert (tracker.absorbed_ops.matrix[0] == _z_row(n, [2])).all()
    assert tracker.num_absorbed_dof() == 1


def test_gauge_absorb_banks_the_consumed_operator():
    # A gauge measurement consuming the SECOND of two logicals banks that
    # logical's operator (and only it) while the surviving first logical
    # stays out of the ledger.
    n = 3
    tracker = SyndromeTracker(n, 0)
    tracker._gauge_logical_vectors = [np.array([0, 1], dtype=np.uint8)]
    old = np.vstack([_z_row(n, [0]), _z_row(n, [1])])

    tracker._record_measurement_logical_effects(
        {0}, old_logicals_current_frame=old)

    assert tracker.absorbed_ops.count == 1
    assert (tracker.absorbed_ops.matrix[0] == _z_row(n, [1])).all()
    assert tracker.num_absorbed_dof() == 1


def test_real_reset_path_ignores_disjoint_resets():
    # ancilla-style resets on qubits outside every banked relation pass
    # through untouched (the per-round hot path stays cheap and silent)
    n = 4
    tracker = SyndromeTracker(n, 1)
    tracker.record_absorbed_op(_z_row(n, [2, 3]))

    reset = np.zeros((2, 2 * n), dtype=np.uint8)
    reset[0, n + 0] = 1
    reset[1, n + 1] = 1
    tracker.process_resets(reset)
    assert tracker.num_absorbed_dof() == 1


def test_insertion_stores_the_canonical_residue():
    # Upstream review round 3, item 2 (the exact counterexample):
    # stabilizer Z0, inserted relation Z0*Z1 — the stored row must be the
    # irreducible residue Z1.  (The upstream test's second half exercises
    # the reset-over-banked guard, which is not ported; see file header.)
    n = 2
    tracker = SyndromeTracker(n, 0)
    tracker.stabilizers.matrix = _z_row(n, [0]).reshape(1, -1)
    tracker.stabilizers.records = [[7]]

    assert tracker.record_absorbed_op(_z_row(n, [0, 1])) is True
    assert (tracker.absorbed_ops.matrix[0] == _z_row(n, [1])).all()
    assert tracker.num_absorbed_dof() == 1
    # records stay positionally aligned with the matrix (CircLS deviation:
    # retire/resolve filter them together)
    assert len(tracker.absorbed_ops.records) == tracker.absorbed_ops.count
