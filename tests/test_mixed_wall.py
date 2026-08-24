"""Mixed-wall joint measurement M(X̄2 ⊗ Z̄3) — the odd-cycle resolver (FIG4).

Two d×d patches in DIFFERENT conventions face each other across a 2-row
ancilla-only gap: q3 (textbook, Z̄ horizontal) below, q2 (paper-q2, X̄ horizontal)
above.  The joint is measured through a thin X↔Z domain wall: Fig 39-style WIDE
weight-4 mixed checks (feet on the two facing data rows, ancilla mid-gap, no new
data qubits) — the wall that a GF(2) exhaustive search proves UNIQUE:
alternating-orientation dominoes + one end lobe, both facing boundary-lobe sets
retired.  The wall terminates on open boundary at both ends: no twist.

Circuit level: the FIXED compact-7 diagonal schedule extended to mixed/wide
checks (sign-based corner classification; CX for X-feet, CZ for Z-feet; K&F
relay dominoes use the two alternating Fig 4(c) variants, flag-at-hi = '+' /
syndrome-at-hi = '-') keeps every ladder-hook residual on a diagonal or mixed
pair.  Measured result:
FULL graphlike (= matchable) distance at d = 3 and 5 for both the eigenstate
init (2 observables) and the teleport init (1 observable with the joint outcome
m folded in) — no flag qubits and no correlation-aware decoder needed, unlike
the N/Z-scheduled stretched stabilizers of K&F §VI (whose halved distance we
reproduced before switching the merged rounds to the diagonal block).
"""
import pytest

from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.ir.builder import CircuitBuilder
from lightstim.qec_code.surface_code.rotated.diagonal_se import (
    DiagonalSurfaceCodeExtractionBlock)
from lightstim.qec_code.surface_code.rotated.rule_based_patch import RuleBasedRectPatch
from lightstim.qec_code.surface_code.rotated.litinski_layouts import rect_boundary
from lightstim.noise.config import NoiseConfig
from circls.core.joint_merge import (
    RotatedSeamWallCoupler, WallSpec)

pytestmark = pytest.mark.smoke

NP = NoiseConfig(p_1q=1e-3, p_2q=1e-3, p_meas=1e-3, p_reset=1e-3, p_idle=1e-3)


def wall_spec(d):
    """The unique mixed wall at the TRUE seam-grid spacing (GF(2) exhaustive,
    scratch mixed_wall_seam — same alternating dominoes + end lobe as the
    original 3-cell-gap search; the wall algebra is spacing-invariant), in the
    K&F relay realization: every check carries a 3-qubit apparatus
    (flag aux A on the CZ/Z-feet side, syndrome aux B on the CX/X-feet side,
    shared seam-relay S), all couplings nearest-neighbour (±1, ±1).
    Returns [(syn_coord_B, pauli_dict, kf_meta)]."""
    bot, top, seam = 2 * d - 1, 2 * d + 3, 2 * d + 1
    lo, hi = 2 * d, 2 * d + 2
    checks = []
    for c in range(2, 2 * d, 2):
        plus = (c % 4 == 2)
        t, b = ('X', 'Z') if plus else ('Z', 'X')
        pauli = {(c - 1, top): t, (c + 1, top): t,
                 (c - 1, bot): b, (c + 1, bot): b}
        # A must be diagonal-adjacent to the Z feet, B to the X feet
        A, B = ((c, lo), (c, hi)) if plus else ((c, hi), (c, lo))
        # K&F Fig 4 variant keying under the fixed table: Z-feet-on-top
        # dominoes (A at hi) anchor at slot 1 ('+'), X-on-top (B at hi) at
        # slot 3 ('-') — the top row's SW bulk corner pins slot 4 on every
        # top foot, which rules out the other two A/B-offset combinations.
        checks.append((B, pauli, {'flag': A, 'shared': (c - 1, seam),
                                  'orient': '-' if plus else '+'}))
    e = 2 * d - 1
    checks.append(((2 * d, hi), {(e, top): 'X', (e, bot): 'Z'},
                   {'flag': (2 * d, lo), 'shared': (e, seam),
                    'orient': '-'}))
    return checks


def build_case(d, init_q2, init_q3, rounds=None):
    rounds = rounds or d
    oy = 2 * d + 2                      # TRUE seam-grid pitch (origin_of)
    system = QECSystem()
    system.add_patch(RuleBasedRectPatch(distance_x=d, distance_z=d,
                                        forced=rect_boundary(d, d, 'Z', 'X'),
                                        phase=0),
                     offset=(0, 0), name="q3")
    system.add_patch(RuleBasedRectPatch(distance_x=d, distance_z=d,
                                        forced=rect_boundary(d, d, 'X', 'Z'),
                                        phase=0),
                     offset=(0, oy), name="q2")
    facing = {(c, 2 * d) for c in range(2, 2 * d, 2)} | \
             {(c, 2 * d + 2) for c in range(2, 2 * d, 2)}
    system.register_coupler(
        RotatedSeamWallCoupler(), patch_names=["q3", "q2"], name="wall",
        spec=WallSpec.from_records(wall_spec(d), facing),
        occupied=frozenset(system.index_map))

    tracker = SyndromeTracker(num_qubits=system.num_qubits,
                              expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker=tracker, system_config=system,
                             if_detector=True)
    system.register_tracker(tracker)
    system.register_builder(builder)
    builder.write_coordinates()

    owner = system.index_to_owner_map
    q2_data = [q for q in sorted(system.data_indices) if owner.get(q) == "q2"]
    q3_data = [q for q in sorted(system.data_indices) if owner.get(q) == "q3"]
    init = {q: init_q2 for q in q2_data}
    init.update({q: init_q3 for q in q3_data})
    builder.initialize(init, n=system.num_qubits)
    builder.apply_syndrome_extraction(
        DiagonalSurfaceCodeExtractionBlock(system).circuit, rounds=rounds)

    # merge: the production coupler pauses the facing lobes (spec.retire)
    builder.activate_coupler("wall")

    builder.apply_syndrome_extraction(
        DiagonalSurfaceCodeExtractionBlock(system).circuit, rounds=rounds)

    # split-by-readout (two_patch_ls precedent)
    meas = {q: init_q2 for q in q2_data}
    meas.update({q: init_q3 for q in q3_data})
    builder.apply_data_readout(final_measurements=meas)
    return builder


def memory_baseline(d):
    """Plain d×d memory of comparable depth (2d diagonal rounds) — the LER
    reference the mixed wall is compared against (see show_fig.ipynb Example 7)."""
    system = QECSystem()
    system.add_patch(RuleBasedRectPatch(distance_x=d, distance_z=d,
                                        forced=rect_boundary(d, d, 'Z', 'X'),
                                        phase=0),
                     offset=(0, 0), name="q")
    tracker = SyndromeTracker(num_qubits=system.num_qubits,
                              expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker=tracker, system_config=system,
                             if_detector=True)
    system.register_tracker(tracker)
    system.register_builder(builder)
    builder.write_coordinates()
    builder.initialize({q: 'Z' for q in sorted(system.data_indices)},
                       n=system.num_qubits)
    builder.apply_syndrome_extraction(
        DiagonalSurfaceCodeExtractionBlock(system).circuit, rounds=2 * d)
    builder.apply_data_readout({q: 'Z' for q in sorted(system.data_indices)})
    return builder


def _check(d, init_q2, init_q3, n_obs):
    builder = build_case(d, init_q2, init_q3)
    c = builder.circuit
    assert c.num_observables == n_obs
    det, obs = c.compile_detector_sampler(seed=0).sample(
        512, separate_observables=True)
    assert not det.any(), f"d={d}: detector fired at p=0"
    assert not obs.any(), f"d={d}: observable not deterministic at p=0"
    noisy = builder.build_noisy_circuit(noise_params=NP,
                                       noise_model='circuit_level')
    noisy.detector_error_model(decompose_errors=True)
    assert len(noisy.shortest_graphlike_error()) == d


@pytest.mark.parametrize("init,n_obs", [(('X', 'Z'), 2), (('Z', 'X'), 1)])
def test_mixed_wall_d3(init, n_obs):
    _check(3, init[0], init[1], n_obs)


@pytest.mark.parametrize("init,n_obs", [(('X', 'Z'), 2), (('Z', 'X'), 1)])
def test_mixed_wall_d5(init, n_obs):
    _check(5, init[0], init[1], n_obs)
