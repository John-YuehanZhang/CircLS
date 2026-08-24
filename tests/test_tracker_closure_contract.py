"""Terminal-readout closure contract, independent of any patch/PPM geometry.

The tracker's stabilizer bank is not a canonical basis: row updates during
measurement processing multiply rows together, so a bank row can be a
product of local checks whose Pauli support exceeds any single physical
check.  Its terminal closure is legitimate syndrome information and must be
emitted like every other determined row's — suppressing it by a support
weight threshold silently costs decoder information (paired-noise MWPM LER
is ~30% worse without the closure; PR #68 review, blocker #1).

This test drives SyndromeTracker directly with a deliberately non-canonical
bank so the contract is pinned at the tracker layer, not via any lattice
surgery construction.
"""
import numpy as np
import stim

from lightstim.ir.tracker import SyndromeTracker


def test_high_weight_product_row_still_emits_closure_detector():
    n = 6
    # Physical history on |0..0>: MPP C2 = Z3Z4Z5 (rec 0), then
    # MPP C1 = Z0Z1Z2Z3 (rec 1), then transversal Z readout (recs 2..7).
    # Every parity below is deterministic.
    circuit = stim.Circuit("""
        R 0 1 2 3 4 5
        MPP Z3*Z4*Z5
        MPP Z0*Z1*Z2*Z3
        M 0 1 2 3 4 5
    """)

    tracker = SyndromeTracker(n, 0)
    tracker.total_measurements = 2  # the two MPPs above

    # Non-canonical bank: row A = C2 (support 3, record [0]); row B = C1*C2
    # = Z0 Z1 Z2 Z4 Z5 (support 5 — above any single check), records [0, 1].
    row_a = np.zeros(2 * n, dtype=np.uint8)
    row_a[[n + 3, n + 4, n + 5]] = 1
    row_b = np.zeros(2 * n, dtype=np.uint8)
    row_b[[n + 0, n + 1, n + 2, n + 4, n + 5]] = 1
    tracker.stabilizers.matrix = np.vstack([row_a, row_b])
    tracker.stabilizers.records = [[0], [0, 1]]

    final_paulis = np.zeros((n, 2 * n), dtype=np.uint8)
    for q in range(n):
        final_paulis[q, n + q] = 1  # Z readout, order matches the M above

    tracker.process_data_measurement(
        circuit,
        final_paulis,
        idx_to_coord_map={q: (float(q), 0.0) for q in range(n)},
    )

    detectors = [i for i in circuit.flattened() if i.name == "DETECTOR"]
    assert len(detectors) == 2, (
        "every determined stabilizer bank row must emit its closure "
        f"detector; got {len(detectors)}"
    )
    # Row A closes over 1 historical + 3 final records; row B (the
    # high-support product) over 2 historical + 5 final records.
    assert sorted(len(i.targets_copy()) for i in detectors) == [4, 7]

    # The assembled history is physically consistent: silent at p=0.
    assert not circuit.compile_detector_sampler(seed=0).sample(1024).any()
def test_sentinel_tagged_gauge_row_emits_no_detector():
    """The sentinel-record branch (kept by design, review follow-up: 'keep
    its semantics covered separately'): a stabilizer row whose records
    carry UNMEASURED_STAB_RECORD is an unwatched gauge direction's
    close-out — its only content is init-parity == readout-parity with no
    banked init record, so emitting it would attach an unbanked,
    per-shot-random parity.  The terminal readout must skip exactly these
    rows, by PROVENANCE (the sentinel), never by support weight."""
    import numpy as np
    import stim
    from lightstim.ir.tracker import SyndromeTracker, UNMEASURED_STAB_RECORD

    n = 4
    circuit = stim.Circuit("""
        R 0 1 2 3
        MPP Z0*Z1
        M 0 1 2 3
    """)
    tracker = SyndromeTracker(n, 0)
    tracker.total_measurements = 1

    watched = np.zeros(2 * n, dtype=np.uint8)
    watched[[n + 0, n + 1]] = 1
    unwatched = np.zeros(2 * n, dtype=np.uint8)
    unwatched[[n + 2, n + 3]] = 1
    tracker.stabilizers.matrix = np.vstack([watched, unwatched])
    tracker.stabilizers.records = [[0], [UNMEASURED_STAB_RECORD]]

    final_paulis = np.zeros((n, 2 * n), dtype=np.uint8)
    for q in range(n):
        final_paulis[q, n + q] = 1
    tracker.process_data_measurement(
        circuit, final_paulis,
        idx_to_coord_map={q: (float(q), 0.0) for q in range(n)})

    detectors = [i for i in circuit.flattened() if i.name == "DETECTOR"]
    assert len(detectors) == 1, (
        f"only the record-carrying row may close out; got {len(detectors)}")
    recs = {t.value + circuit.num_measurements
            for t in detectors[0].targets_copy()}
    assert recs == {0, 1, 2}, (
        f"the emitted closure must be the WATCHED row's (MPP rec 0 + finals "
        f"of qubits 0/1); got absolute records {sorted(recs)}")


def test_gauge_row_records_sentinel_on_the_producing_path():
    """The sentinel must be written on the PRODUCING path
    (process_mid_measurement), not only injected by hand as in
    test_sentinel_tagged_gauge_row_emits_no_detector above.

    Regression for a dead-write aliasing bug: when standing logicals are
    present, process_mid_measurement reorders the records into a fresh list
    (logicals-before-stabs) and used to tag the unwatched-gauge row by
    rebinding an element of THAT copy; the write-back reads only the
    ORIGINAL full_records, so the sentinel was lost exactly when
    num_logs > 0 and the gauge row reached readout records-less — its
    forbidden close-out parity then leaked out as a DETECTOR.  Here the
    budget-full gauge filing runs WITH a logical standing, so the row must
    still carry UNMEASURED_STAB_RECORD afterwards."""
    from lightstim.ir.tracker import SyndromeTracker, UNMEASURED_STAB_RECORD

    def zrow(nq, qubits):
        row = np.zeros(2 * nq, dtype=np.uint8)
        for q in qubits:
            row[nq + q] = 1
        return row

    n = 3
    tracker = SyndromeTracker(n, 1)          # expected_num_logicals = 1
    tracker.total_measurements = 1           # Z1 already measured (rec 0)
    tracker.stabilizers.matrix = np.vstack([zrow(n, [0]), zrow(n, [1])])
    tracker.stabilizers.records = [[], [0]]  # Z0 records-less, Z1 rec 0
    tracker.logicals.matrix = zrow(n, [2]).reshape(1, -1)
    tracker.logicals.records = [[]]          # a logical is standing (num_logs>0)

    # the circuit already carries the history and the re-measured Z1 (rec 1);
    # the budget is full (the one expected logical is Z2), so re-measuring Z1
    # files the records-less Z0 as an unwatched gauge direction.
    circuit = stim.Circuit()
    circuit.append("MPP", [stim.target_z(1)])   # rec 0 (history)
    circuit.append("MPP", [stim.target_z(1)])   # rec 1 (this round)
    tracker.process_mid_measurement(
        circuit=circuit,
        back_propagated_paulis=zrow(n, [1]).reshape(1, -1),
        syn_coords=[(1.0, 0.0)],
    )

    gauge_rows = [
        rec for row, rec in zip(tracker.stabilizers.matrix,
                                tracker.stabilizers.records)
        if sorted(set(np.where(row[:n])[0].tolist())
                  | set(np.where(row[n:])[0].tolist())) == [0]
    ]
    assert gauge_rows, "the records-less Z0 direction should be filed as a gauge row"
    assert gauge_rows[0] == [UNMEASURED_STAB_RECORD], (
        "the budget-full gauge row filed with a logical standing must carry "
        f"the sentinel on the producing path; got {gauge_rows[0]} (dead-write "
        "regression: sentinel written to the reordered copy, not full_records)")


def test_terminal_observable_uses_the_central_allocator():
    """Observable IDs come from tracker.allocate_observable() alone, never
    from circuit.num_observables (the historical collision source): with
    two IDs pre-reserved on the tracker and ZERO observables in the
    circuit, the terminal readout's observable must land on index 2."""
    import numpy as np
    import stim
    from lightstim.ir.tracker import SyndromeTracker

    n = 4
    circuit = stim.Circuit("""
        R 0 1 2 3
        MPP Z0*Z1
        M 0 1 2 3
    """)
    tracker = SyndromeTracker(n, 1)
    tracker.total_measurements = 1
    assert tracker.allocate_observable() == 0
    assert tracker.allocate_observable() == 1

    logical = np.zeros(2 * n, dtype=np.uint8)
    logical[[n + 0, n + 1]] = 1
    tracker.logicals.matrix = logical.reshape(1, -1)
    tracker.logicals.records = [[0]]

    final_paulis = np.zeros((n, 2 * n), dtype=np.uint8)
    for q in range(n):
        final_paulis[q, n + q] = 1
    tracker.process_data_measurement(
        circuit, final_paulis,
        idx_to_coord_map={q: (float(q), 0.0) for q in range(n)})

    obs_ids = [int(i.gate_args_copy()[0]) for i in circuit.flattened()
               if i.name == 'OBSERVABLE_INCLUDE']
    assert obs_ids == [2], (
        f"terminal readout must allocate from the tracker (expected id 2), "
        f"got {obs_ids} - id 0 would mean circuit.num_observables leaked "
        f"back in as an allocation source")
    assert tracker.total_observables == 3


def _build_bell_flagging_memory(reserved_ids: int, rounds: int = 5):
    """Color-code Bell-flagging memory via the raw builder, with
    `reserved_ids` observable IDs pre-reserved on the tracker before any
    circuit is emitted (the review's compression-collision reproduction)."""
    from lightstim.ir.builder import CircuitBuilder
    from lightstim.ir.qec_system import QECSystem
    from lightstim.ir.tracker import SyndromeTracker
    from lightstim.qec_code.color_code import (
        ColorCode, ColorCodeBellFlaggingBlock)

    system = QECSystem()
    system.add_patch(ColorCode(distance=3), name="memory")
    num_qubits = len(system.qubit_coords)
    tracker = SyndromeTracker(num_qubits=num_qubits,
                              expected_num_logicals=system.num_logicals)
    for expected in range(reserved_ids):
        assert tracker.allocate_observable() == expected
    builder = CircuitBuilder(tracker=tracker, system_config=system,
                             if_detector=True)
    builder.write_coordinates()
    se_block = ColorCodeBellFlaggingBlock(system)
    data_indices = [system.index_map[c] for c in system.data_coords]
    data_basis = {q: "Z" for q in data_indices}
    block_init = set(getattr(se_block, "data_qubits_initialized_by_block",
                             ()))
    builder.initialize(
        init_dict={q: b for q, b in data_basis.items()
                   if q not in block_init},
        n=num_qubits)
    system.active_qubit_indices.update(data_indices)
    builder.apply_syndrome_extraction(
        circuit_chunk=se_block.circuit,
        rounds=rounds,
        z_only=False,
        measurement_blocks=getattr(se_block, "measurement_blocks", None),
    )
    builder.apply_data_readout(final_measurements=dict(data_basis),
                               z_only=False)
    return builder.circuit


def test_steady_round_compression_respects_reserved_observables():
    """Review round 3 merge blocker: _try_compress_steady_rounds guarded on
    circuit.num_observables only, so with tracker IDs pre-reserved (and the
    circuit still observable-free) the compressed body accumulated into the
    hardcoded ID 0 — an already-reserved observable — while the terminal
    readout correctly allocated ID 2.  No generated OBSERVABLE_INCLUDE may
    target a reserved ID, and the built circuit must stay deterministic."""
    circuit = _build_bell_flagging_memory(reserved_ids=2)

    obs_ids = [int(i.gate_args_copy()[0]) for i in circuit.flattened()
               if i.name == 'OBSERVABLE_INCLUDE']
    assert obs_ids and all(i == 2 for i in obs_ids), (
        f"OBSERVABLE_INCLUDE ids {sorted(set(obs_ids))}: contributions to a "
        f"reserved ID silently fold a second logical result into someone "
        f"else's observable (invisible to p=0 sampling)")

    det, obs = circuit.compile_detector_sampler(seed=0).sample(
        64, separate_observables=True)
    assert not det.any(), "declined compression must stay deterministic"
    assert not obs.any()


def test_steady_round_compression_still_fires_without_reservations():
    """The guard must not over-trigger: with no tracker-side reservations
    the compression keeps firing (REPEAT block present) and every
    OBSERVABLE_INCLUDE legally targets the sole logical's ID 0."""
    import stim

    circuit = _build_bell_flagging_memory(reserved_ids=0)
    assert any(isinstance(i, stim.CircuitRepeatBlock) for i in circuit), (
        "compression stopped firing for the reservation-free path — the "
        "guard is broader than the review's minimal fix")
    obs_ids = [int(i.gate_args_copy()[0]) for i in circuit.flattened()
               if i.name == 'OBSERVABLE_INCLUDE']
    assert obs_ids and all(i == 0 for i in obs_ids)


def test_canonicalization_sentinel_row_first_measurement_emits_detector():
    """The [-1] sentinel has TWO producers with opposite needs, and the
    mid-circuit exact-match path must serve the canonicalization one:
    stabilizer_canonicalization files "code stabilizer, deterministic from
    init, just not measured YET" rows as [[-1]], and that row's FIRST
    measurement is a legitimate deterministic check.  Regression for a
    blanket suppress that keyed on the sentinel alone and silently dropped
    these detectors (cross_ls d=3: 59 -> 49 detectors, graphlike distance
    2 -> 1)."""
    from lightstim.ir.tracker import SyndromeTracker, UNMEASURED_STAB_RECORD

    n = 2
    row = np.zeros(2 * n, dtype=np.uint8)
    row[[n + 0, n + 1]] = 1                      # Z0 Z1
    tracker = SyndromeTracker(n, 0)
    tracker.stabilizers.matrix = row.reshape(1, -1)
    tracker.stabilizers.records = [[UNMEASURED_STAB_RECORD]]

    circuit = stim.Circuit()
    circuit.append("MPP", [stim.target_z(0), stim.target_combiner(),
                           stim.target_z(1)])
    tracker.process_mid_measurement(
        circuit=circuit,
        back_propagated_paulis=row.reshape(1, -1),
        syn_coords=[(0.0, 0.0)],
    )
    detectors = [i for i in circuit.flattened() if i.name == "DETECTOR"]
    assert len(detectors) == 1, (
        "first measurement of a canonicalization-filed ([-1]) stabilizer "
        f"row must emit its deterministic detector; got {len(detectors)}")
