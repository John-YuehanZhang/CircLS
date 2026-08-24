"""Horizontal mixed wall — the 90°-transposed twin of test_mixed_wall (FIG4a/c).

Two d×d patches SIDE BY SIDE at the true seam pitch: ql (X̄ horizontal,
Z̄ vertical) on the left, qr (Z̄ horizontal, X̄ vertical) on the right, joined
through a vertical 3-column aux band by HORIZONTAL stretched dominoes (feet
left and right of the band).  The transposition is an exact symmetry of the
wall ALGEBRA, but NOT of the fixed compact-7 schedule (Z owns slots 1-4,
X owns 4-7, and the corner tables are not x<->y symmetric), so this exercises
the genuinely different horizontal branch of the K&F Fig 4 table: variant
keying flag-at-RIGHT = '+' (slot-1 anchor) / syndrome-at-RIGHT = '-'
(slot-3 anchor) — the right column's SW bulk corner pins slot 4 on every
right foot, mirroring the vertical wall's top-row rule — and foot subslots
keyed by VARIANT ('-' couples both sides north first, '+' south first,
Fig 4c's tl=4,bl=5 / tr=5,br=6).
"""
import pytest

from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.ir.builder import CircuitBuilder
from lightstim.qec_code.surface_code.rotated.rule_based_patch import (
    RuleBasedRectPatch)
from lightstim.qec_code.surface_code.rotated.litinski_layouts import (
    rect_boundary)
from lightstim.qec_code.surface_code.rotated.diagonal_se import (
    DiagonalSurfaceCodeExtractionBlock, _kf_events)
from lightstim.noise.config import NoiseConfig
from circls.core.joint_merge import (
    RotatedSeamWallCoupler, WallSpec)

pytestmark = pytest.mark.smoke

NP = NoiseConfig(p_1q=1e-3, p_2q=1e-3, p_meas=1e-3, p_reset=1e-3, p_idle=1e-3)


def horizontal_wall_spec(d):
    """test_mixed_wall.wall_spec with every coordinate transposed (x<->y):
    dominoes at even ROWS, S relay one row south, feet on the two facing data
    COLUMNS.  A stays diagonal-adjacent to the Z feet; the fixed-table variant
    keying transposes to A-at-hi(right) = '+' / B-at-hi(right) = '-'.
    Returns [(syn_coord_B, pauli_dict, kf_meta)]."""
    lt, rt, seam = 2 * d - 1, 2 * d + 3, 2 * d + 1
    lo, hi = 2 * d, 2 * d + 2
    checks = []
    for r in range(2, 2 * d, 2):
        plus = (r % 4 == 2)
        t, b = ('X', 'Z') if plus else ('Z', 'X')   # t = RIGHT feet (was top)
        pauli = {(rt, r - 1): t, (rt, r + 1): t,
                 (lt, r - 1): b, (lt, r + 1): b}
        A, B = ((lo, r), (hi, r)) if plus else ((hi, r), (lo, r))
        checks.append((B, pauli, {'flag': A, 'shared': (seam, r - 1),
                                  'orient': '-' if plus else '+'}))
    e = 2 * d - 1
    checks.append(((hi, 2 * d), {(rt, e): 'X', (lt, e): 'Z'},
                   {'flag': (lo, 2 * d), 'shared': (seam, e),
                    'orient': '-'}))
    return checks


def build_case_h(d, init_qr, init_ql, rounds=None):
    rounds = rounds or d
    ox = 2 * d + 2                      # TRUE seam-grid pitch, now along x
    system = QECSystem()
    system.add_patch(RuleBasedRectPatch(distance_x=d, distance_z=d,
                                        forced=rect_boundary(d, d, 'X', 'Z'),
                                        phase=0),
                     offset=(0, 0), name="ql")
    system.add_patch(RuleBasedRectPatch(distance_x=d, distance_z=d,
                                        forced=rect_boundary(d, d, 'Z', 'X'),
                                        phase=0),
                     offset=(ox, 0), name="qr")
    facing = {(2 * d, r) for r in range(0, 2 * d + 1, 2)} | \
             {(2 * d + 2, r) for r in range(0, 2 * d + 1, 2)}
    system.register_coupler(
        RotatedSeamWallCoupler(), patch_names=["ql", "qr"], name="wall",
        spec=WallSpec.from_records(horizontal_wall_spec(d), facing),
        occupied=frozenset(system.index_map))

    tracker = SyndromeTracker(num_qubits=system.num_qubits,
                              expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker=tracker, system_config=system,
                             if_detector=True)
    system.register_tracker(tracker)
    system.register_builder(builder)
    builder.write_coordinates()

    owner = system.index_to_owner_map
    qr = [q for q in sorted(system.data_indices) if owner.get(q) == "qr"]
    ql = [q for q in sorted(system.data_indices) if owner.get(q) == "ql"]
    init = {q: init_qr for q in qr}
    init.update({q: init_ql for q in ql})
    builder.initialize(init, n=system.num_qubits)
    builder.apply_syndrome_extraction(
        DiagonalSurfaceCodeExtractionBlock(system).circuit, rounds=rounds)

    # merge: the production coupler pauses the facing lobes (spec.retire)
    builder.activate_coupler("wall")

    builder.apply_syndrome_extraction(
        DiagonalSurfaceCodeExtractionBlock(system).circuit, rounds=rounds)

    meas = {q: init_qr for q in qr}
    meas.update({q: init_ql for q in ql})
    builder.apply_data_readout(final_measurements=meas)
    return builder


def test_horizontal_domino_slots_match_fig4():
    """The emitted foot slots ARE the paper's horizontal numbers (1-indexed):
    '-' domino (Fig 4c, A left): A tl=4,bl=5 / B tr=5,br=6;
    '+' domino (A right): A br=2,tr=3 / B bl=3,tl=4."""
    d = 3
    builder = build_case_h(d, 'X', 'Z', rounds=1)   # build_case activates wall
    system = builder.system
    se = DiagonalSurfaceCodeExtractionBlock(system)
    _, kf_checks = se._gather()
    foot_slots = {}                       # coord -> 1-indexed slot
    by_B = {}
    for ch in kf_checks:
        by_B[ch['B']] = ch
        for slot, gate, (u, v) in _kf_events(ch):
            for q in (u, v):
                if q not in (ch['A'], ch['S'], ch['B']):
                    foot_slots[(ch['B'], q)] = slot + 1
    lt, rt = 2 * d - 1, 2 * d + 3                   # 5, 9
    # '-' domino at row 2: A=(6,2) left, B=(8,2) right, north first
    assert foot_slots[((8, 2), (lt, 3))] == 4       # A north (tl)
    assert foot_slots[((8, 2), (lt, 1))] == 5       # A south (bl)
    assert foot_slots[((8, 2), (rt, 3))] == 5       # B north (tr)
    assert foot_slots[((8, 2), (rt, 1))] == 6       # B south (br)
    # '+' domino at row 4: A=(8,4) right, B=(6,4) left, south first
    assert foot_slots[((6, 4), (rt, 3))] == 2       # A south (br)
    assert foot_slots[((6, 4), (rt, 5))] == 3       # A north (tr)
    assert foot_slots[((6, 4), (lt, 3))] == 3       # B south (bl)
    assert foot_slots[((6, 4), (lt, 5))] == 4       # B north (tl)


def _check(d, init_qr, init_ql, n_obs):
    builder = build_case_h(d, init_qr, init_ql)
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
def test_mixed_wall_horizontal_d3(init, n_obs):
    _check(3, init[0], init[1], n_obs)


@pytest.mark.parametrize("init,n_obs", [(('X', 'Z'), 2), (('Z', 'X'), 1)])
def test_mixed_wall_horizontal_d5(init, n_obs):
    _check(5, init[0], init[1], n_obs)
