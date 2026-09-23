"""Resource aggregator: hand-computed oracle cases + own-circuit invariants.

Every number in the hand cases is derived on paper in the comments — these are
oracle tests, not change detectors.
"""
import contextlib
import io

import pytest
import stim

from circls.metrics import circuit_stats, experiment_stats, timed_build
from circls.core.sequential_ppm_ls import PPMStep, SequentialPPMExperiment
from circls.core.routed_multi_patch_ls import PatchSpec, origin_of

pytestmark = pytest.mark.smoke

D = 3


# ── hand-computed stim circuits ───────────────────────────────────────────────

def test_hand_circuit_counts():
    # Moments:  L0[R 0 1]  L1[CX 0 1]  L2[M 1]  L3[R 2]  L4[CX 2 0]  L5[M 0 2]
    # Measurement layers: L2, L5  → 2 rounds (round1 = L0..L2, round2 = L3..L5).
    # q0 touched L0..L5 → rounds 1..2 → 2;  q1 L0..L2 → 1;  q2 L3..L5 → 1.
    # qubit 7 exists only in QUBIT_COORDS → allocated, never active.
    c = stim.Circuit("""
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 0) 1
        QUBIT_COORDS(2, 0) 2
        QUBIT_COORDS(9, 9) 7
        R 0 1
        TICK
        CX 0 1
        TICK
        M 1
        TICK
        R 2
        TICK
        CX 2 0
        TICK
        M 0 2
    """)
    s = circuit_stats(c)
    assert s.qubits_allocated == 4          # {0,1,2,7}
    assert s.qubits_active == 3             # {0,1,2}
    assert s.measurement_layers == 2
    assert s.ticks == 5
    assert s.qubit_rounds == 2 + 1 + 1
    assert s.live_rounds == {0: (1, 2), 1: (1, 1), 2: (2, 2)}
    assert s.num_detectors == 0 and s.num_observables == 0


def test_repeat_block_flattened():
    # R0 | (H0 | M0) ×3  → 3 measurement layers; q0 live rounds 1..3 → 3.
    c = stim.Circuit("""
        R 0
        TICK
        REPEAT 3 {
            H 0
            TICK
            M 0
            TICK
        }
    """)
    s = circuit_stats(c)
    assert s.measurement_layers == 3
    assert s.qubits_allocated == s.qubits_active == 1
    assert s.qubit_rounds == 3
    assert s.ticks == 7                     # 1 + 2×3 TICKs after flattening


def test_noise_channels_do_not_activate():
    # qubit 5 is only ever hit by a noise channel: allocated, not active.
    c = stim.Circuit("""
        R 0
        TICK
        X_ERROR(0.01) 5
        DEPOLARIZE1(0.01) 0
        M 0
    """)
    s = circuit_stats(c)
    assert s.qubits_allocated == 2          # {0,5}
    assert s.qubits_active == 1             # {0}
    assert s.measurement_layers == 1
    assert s.qubit_rounds == 1


def test_measurement_and_reset_both_activate():
    # MR counts as measurement layer; a reset-only qubit is active too.
    c = stim.Circuit("""
        RX 1
        TICK
        MR 0
    """)
    s = circuit_stats(c)
    assert s.qubits_active == 2
    assert s.measurement_layers == 1


# ── invariants on a real compiled circuit of ours ─────────────────────────────

def _own_experiment():
    px = [PatchSpec("Q1", origin_of(0, 0, D, seam=True), D, "X_horizontal"),
          PatchSpec("Q2", origin_of(2, 0, D, seam=True), D, "X_horizontal")]
    return SequentialPPMExperiment(
        px, [PPMStep([("Q1", "Z"), ("Q2", "Z")])],
        initial_states={"Q1": "Z", "Q2": "Z"},
        final_measure_states={"Q1": "Z", "Q2": "Z"},
        rounds=D, rounds_init=1)


def _own_circuit():
    with contextlib.redirect_stdout(io.StringIO()):
        return _own_experiment().build()


def test_own_circuit_invariants():
    c = _own_circuit()
    s = circuit_stats(c)
    # S4 == S5 self-check: our compiler must not declare qubits nobody uses.
    assert s.qubits_active == s.qubits_allocated
    assert s.ticks == c.num_ticks
    assert s.measurement_layers > 0
    # live ranges well-formed and within [1, rounds]
    for q, (lo, hi) in s.live_rounds.items():
        assert 1 <= lo <= hi <= s.measurement_layers
    # V2 bounded by the full space-time rectangle
    assert 0 < s.qubit_rounds <= s.qubits_active * s.measurement_layers


def test_stats_do_not_mutate_circuit():
    c = _own_circuit()
    before = str(c)
    circuit_stats(c)
    assert str(c) == before


def test_reuse_gap_not_billed():
    # q0 is measured out in round 1 and RE-RESET in round 3 (a retired cell
    # reused later): holding = {1} ∪ {3} = 2 rounds, while the naive span is 3.
    # q1 is worked on throughout (a gate follows its round-2 measurement, so
    # its round-3 readout is real): 3.  Layers L0..L5, measurement layers
    # L1/L3/L5.
    c = stim.Circuit("""
        R 0 1
        TICK
        M 0
        TICK
        H 1
        TICK
        M 1
        TICK
        R 0
        H 1
        TICK
        M 0 1
    """)
    s = circuit_stats(c)
    assert s.measurement_layers == 3
    assert s.occupancy[0] == ((1, 1), (3, 3))
    assert s.occupancy[1] == ((1, 3),)
    assert s.qubit_rounds == 2 + 3
    assert s.qubit_rounds_span == 3 + 3


def test_remeasure_without_reset_not_billed():
    # q0 is measured out at L1 and measured AGAIN at L5 with no reset and no
    # gate in between (a redundant readout of an already measured-out cell):
    # the second measurement holds nothing, so q0 is billed round 1 only.
    # q1 is worked on throughout: rounds 1..3.  Measurement layers L1/L3/L5.
    c = stim.Circuit("""
        R 0 1
        TICK
        M 0
        TICK
        H 1
        TICK
        M 1
        TICK
        H 1
        TICK
        M 0 1
    """)
    s = circuit_stats(c)
    assert s.measurement_layers == 3
    assert s.occupancy[0] == ((1, 1),)
    assert s.live_rounds[0] == (1, 1)
    assert s.occupancy[1] == ((1, 3),)
    assert s.qubit_rounds == 1 + 3
    assert s.qubit_rounds_span == 1 + 3


def test_gate_after_measurement_reopens_hold():
    # A gate on a measured-out qubit puts it back to work: the later
    # measurement is a real readout and the hold runs L0..L3 → rounds 1..2.
    c = stim.Circuit("""
        R 0
        TICK
        M 0
        TICK
        H 0
        TICK
        M 0
    """)
    s = circuit_stats(c)
    assert s.measurement_layers == 2
    assert s.occupancy[0] == ((1, 2),)
    assert s.qubit_rounds == 2


def test_product_measurements_do_not_close_hold():
    # MPP / MZZ are not destructive: repeated product measurements with no
    # reset keep both qubits held (rounds 1..3 each), and the final M is billed.
    c = stim.Circuit("""
        R 0 1
        TICK
        MPP X0*X1
        TICK
        MZZ 0 1
        TICK
        M 0 1
    """)
    s = circuit_stats(c)
    assert s.measurement_layers == 3
    assert s.occupancy[0] == s.occupancy[1] == ((1, 3),)
    assert s.qubit_rounds == 3 + 3


def test_mr_chain_unchanged_by_close_rule():
    # Ancilla measure+reset chains stay contiguous: MR closes and reopens in
    # the same moment, so q0 is billed every round, 1..3.
    c = stim.Circuit("""
        R 0
        TICK
        MR 0
        TICK
        MR 0
        TICK
        M 0
    """)
    s = circuit_stats(c)
    assert s.measurement_layers == 3
    assert s.live_rounds[0] == (1, 3)
    assert s.qubit_rounds == 3


# ── layer 2: hand-derived oracle on the 2-patch adjacent-cell ZZ case ─────────

def test_experiment_stats_two_patch_oracle():
    # Geometry: Q1 on cell (0,0), Q2 on cell (2,0), routed corridor on (1,0).
    # Timeline (measurement layers): baseline SE (rounds_init=1) → r1;
    # merged SE ×3 → r2..r4; corridor readout → r5; final readout → r6.
    # Tile occupancy: patch cells r1..r6 (6 each); corridor r2..r5 (its data
    # are reset in the same moment as r1's measurement — holding starts r2).
    # tile_rounds = 6+6+4 = 16 → V1 = 16/3.  L1 = 6+6 = 12 (one cell each).
    # L2 = 12 / (peak 3 × rounds 6) = 2/3.  Nothing retires mid-run → L4 None.
    exp = _own_experiment()
    with contextlib.redirect_stdout(io.StringIO()):
        c, secs = timed_build(exp)
    es = experiment_stats(exp, c, compile_seconds=secs)
    assert es.circuit.measurement_layers == 6
    assert es.tiles_bbox == 3               # tiles (0,0)..(2,0): 3×1 box
    assert es.tiles_touched == 3
    assert es.tiles_peak == 3
    assert es.logical_timesteps == pytest.approx(2.0)      # 6 layers / d=3
    assert es.tile_rounds == 16
    assert es.volume_blocks == pytest.approx(16 / 3)
    assert es.bbox_volume_blocks == pytest.approx(3 * 6 / 3)
    assert es.live_tile_rounds == 12
    assert es.utilization == pytest.approx(12 / 18)
    assert es.reuse_rate is None
    assert es.patch_live == {"Q1": (1, 6), "Q2": (1, 6)}
    assert es.tile_occupancy[(1, 0)] == (2, 3, 4, 5)
    assert es.compile_seconds == secs > 0
    # V1↔V2 density: ~one patch's worth of qubits per tile·round
    assert D**2 <= es.qubits_per_tile_round <= 2 * (D + 1)**2
    # S4 == S5 gate is enforced inside experiment_stats (it did not raise)
    assert es.circuit.qubits_active == es.circuit.qubits_allocated


# ── layer 2: liveness ablation — retiring Q1 must shrink the accounts ─────────

def _three_patch_exp(liveness):
    px = [PatchSpec("Q1", origin_of(0, 0, D, seam=True), D, "X_horizontal"),
          PatchSpec("Q2", origin_of(2, 0, D, seam=True), D, "X_horizontal"),
          PatchSpec("Q3", origin_of(4, 0, D, seam=True), D, "X_horizontal")]
    seq = [PPMStep([("Q1", "Z"), ("Q2", "Z")]),
           PPMStep([("Q2", "Z"), ("Q3", "Z")])]
    states = {nm: "Z" for nm in ("Q1", "Q2", "Q3")}
    kw = dict(liveness=liveness)
    if liveness:
        kw["keep_patches"] = {"Q2", "Q3"}
    return SequentialPPMExperiment(
        px, seq, initial_states=states, final_measure_states=states,
        rounds=D, rounds_init=1, **kw)


def test_liveness_shrinks_volume_and_liverange():
    stats = {}
    for flag in (False, True):
        exp = _three_patch_exp(flag)
        with contextlib.redirect_stdout(io.StringIO()):
            c = exp.build()
        stats[flag] = experiment_stats(exp, c)
    off, on = stats[False], stats[True]
    # same program, same placement: footprint unchanged
    assert on.tiles_touched == off.tiles_touched
    assert on.tiles_bbox == off.tiles_bbox   # S1 is ablation-invariant by design
    # retirement ends Q1's live range before the final round …
    assert on.patch_live["Q1"][1] < on.circuit.measurement_layers
    assert off.patch_live["Q1"][1] == off.circuit.measurement_layers
    # … and shrinks both the occupancy volume and the live-range account
    assert on.tile_rounds < off.tile_rounds
    assert on.live_tile_rounds < off.live_tile_rounds
    # Q1's freed cell is never reused here (step 2's corridor is elsewhere)
    assert on.reuse_rate == 0.0
    assert off.reuse_rate is None          # nothing retires without liveness


# ── cross-tool oracle: tqec circuits, numbers verified two independent ways ───
# Fixtures generated 2026-08-04 with tqec v0.2.0 (gallery memory / cnot, k=1,
# fixed-bulk convention).  Expected values were measured twice independently:
# by the metrics-survey probe scripts (metrics_survey/tqec_probes/probe*.py)
# and by circuit_stats itself; both agree.  rounds = bbox_z × d (tqec law),
# alloc − active = idle boundary qubits tqec declares but never gates.

@pytest.mark.parametrize("fixture, alloc, active, rounds, ticks, v2", [
    ("tqec_memory_k1.stim", 25, 17, 3, 20, 51),    # V2 = 17×3: all live throughout
    ("tqec_cnot_k1.stim", 81, 65, 12, 83, 552),    # V2 < 65×12: staggered live ranges
])
def test_tqec_fixture_oracle(fixture, alloc, active, rounds, ticks, v2):
    import pathlib
    path = pathlib.Path(__file__).parent / "fixtures" / fixture
    s = circuit_stats(stim.Circuit(path.read_text()))
    assert s.qubits_allocated == alloc
    assert s.qubits_active == active
    assert s.measurement_layers == rounds
    assert s.ticks == ticks
    assert s.qubit_rounds == v2
