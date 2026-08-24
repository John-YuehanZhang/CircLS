"""Circuit-level verification of the user's type rules (weight-2 layout
defines the patch type; same-type/mixed measurement x same type/different
type):

  row 2: same-type measurement, one textbook one conjugate -> the FIG1
         same-type wall (alternating all-Z / all-X dominoes + uniform end lobe)
  row 3: mixed measurement, one textbook one conjugate -> the mixed wall; the
         arrangement in test_mixed_wall is one of the four; here the MIRROR
         arrangement (both patches with the other weight-2 layout, realized
         as bulk phase 1) whose unique wall is the mirrored one.

Both walls use the K&F nearest-neighbour relay realization (flag aux A +
syndrome aux B + shared seam relay S per check), with the geometric aux
split of DiagonalSurfaceCodeExtractionBlock.  GF(2) ground truth:
scratch verify_rule.py / all_pairs_mixed.py (unique walls, true distance d).
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


def same_type_wall_spec(d):
    """FIG1 wall (bottom rect('X','Z') phase 0, top rect('X','Z') phase 1):
    alternating all-Z / all-X dominoes + all-X end lobe at column 1
    (GF(2)-unique, verify_rule.py).  End lobe on the LEFT, so dominoes take
    the seam relay at c+1 and the lobe takes column 1."""
    bot, top, seam = 2 * d - 1, 2 * d + 3, 2 * d + 1
    lo, hi = 2 * d, 2 * d + 2
    checks = []
    for i, c in enumerate(range(2, 2 * d, 2)):
        t = 'Z' if i % 2 == 0 else 'X'
        pauli = {(c - 1, top): t, (c + 1, top): t,
                 (c - 1, bot): t, (c + 1, bot): t}
        # fixed-table variant keying (see wall_spec in test_mixed_wall):
        # the '+' (slot-1) variant needs the FLAG on the hi row, the '-'
        # (slot-3) variant the SYNDROME there; alternate along the wall so
        # shared feet stay Sec. II-consistent under the K&F Fig 4 slots
        A, B = ((c, hi), (c, lo)) if i % 2 == 0 else ((c, lo), (c, hi))
        checks.append((B, pauli,
                       {'flag': A, 'shared': (c + 1, seam),
                        'orient': '+' if i % 2 == 0 else '-'}))
    # end lobe shares two anticommuting feet with D2: offset +2 ('-') keeps
    # the Sec. II relative order consistent (D2 first on both shared feet)
    checks.append(((0, hi), {(1, top): 'X', (1, bot): 'X'},
                   {'flag': (0, lo), 'shared': (1, seam), 'orient': '-'}))
    return checks


def mirror_mixed_wall_spec(d):
    """Mirror mixed wall (bottom rect('Z','X') phase 1, top rect('X','Z')
    phase 1): dominoes top-Z/bottom-X at c=2 alternating, end lobe X/Z at
    column 1 (GF(2)-unique, all_pairs_mixed.py phases (1,1))."""
    bot, top, seam = 2 * d - 1, 2 * d + 3, 2 * d + 1
    lo, hi = 2 * d, 2 * d + 2
    checks = []
    for i, c in enumerate(range(2, 2 * d, 2)):
        tt, bb = ('Z', 'X') if i % 2 == 0 else ('X', 'Z')
        pauli = {(c - 1, top): tt, (c + 1, top): tt,
                 (c - 1, bot): bb, (c + 1, bot): bb}
        # flag aux A on the Z-feet side; under the FIXED compact-7 table the
        # side choice is forced (top feet can never take slot 4, so only
        # A-at-hi+'+' and B-at-hi+'-' schedule) and this keying satisfies it
        A, B = ((c, hi), (c, lo)) if tt == 'Z' else ((c, lo), (c, hi))
        checks.append(((B[0], B[1]), pauli,
                       {'flag': A, 'shared': (c + 1, seam),
                        'orient': '+' if i % 2 == 0 else '-'}))
    checks.append(((0, hi), {(1, top): 'X', (1, bot): 'Z'},
                   {'flag': (0, lo), 'shared': (1, seam), 'orient': '-'}))
    return checks


def build_wall_case(d, bot_layout, top_layout, wall_checks,
                    init_bot, init_top, meas_bot, meas_top, rounds=None):
    """Two patches at seam pitch + a KF-relay wall between them.
    ``*_layout`` = (lr, tb, phase); init/meas per patch basis."""
    rounds = rounds or d
    oy = 2 * d + 2
    system = QECSystem()
    system.add_patch(
        RuleBasedRectPatch(distance_x=d, distance_z=d,
                           forced=rect_boundary(d, d, *bot_layout[:2],
                                                bot_layout[2]),
                           phase=bot_layout[2]),
        offset=(0, 0), name="qb")
    system.add_patch(
        RuleBasedRectPatch(distance_x=d, distance_z=d,
                           forced=rect_boundary(d, d, *top_layout[:2],
                                                top_layout[2]),
                           phase=top_layout[2]),
        offset=(0, oy), name="qt")
    facing = {(c, 2 * d) for c in range(0, 2 * d + 1, 2)} | \
             {(c, 2 * d + 2) for c in range(0, 2 * d + 1, 2)}
    system.register_coupler(
        RotatedSeamWallCoupler(), patch_names=["qb", "qt"], name="wall",
        spec=WallSpec.from_records(wall_checks, facing),
        occupied=frozenset(system.index_map))

    tracker = SyndromeTracker(num_qubits=system.num_qubits,
                              expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker=tracker, system_config=system,
                             if_detector=True)
    system.register_tracker(tracker)
    system.register_builder(builder)
    builder.write_coordinates()

    owner = system.index_to_owner_map
    qb = [q for q in sorted(system.data_indices) if owner.get(q) == "qb"]
    qt = [q for q in sorted(system.data_indices) if owner.get(q) == "qt"]
    init = {q: init_bot for q in qb}
    init.update({q: init_top for q in qt})
    builder.initialize(init, n=system.num_qubits)
    builder.apply_syndrome_extraction(
        DiagonalSurfaceCodeExtractionBlock(system).circuit, rounds=rounds)

    # merge: the production coupler pauses the facing lobes (spec.retire)
    builder.activate_coupler("wall")

    builder.apply_syndrome_extraction(
        DiagonalSurfaceCodeExtractionBlock(system).circuit, rounds=rounds)

    meas = {q: meas_bot for q in qb}
    meas.update({q: meas_top for q in qt})
    builder.apply_data_readout(final_measurements=meas)
    return builder


def build_same_type_case(d, init_bot='Z', init_top='X'):
    """Same-type measurement X̄⊗X̄, one textbook one conjugate (bulk phase 0
    vs 1, both X̄ horizontal)."""
    return build_wall_case(d, ('X', 'Z', 0), ('X', 'Z', 1),
                           same_type_wall_spec(d),
                           init_bot, init_top, init_bot, init_top)


def build_mirror_mixed_case(d, init_bot='X', init_top='Z'):
    """Mixed measurement X̄⊗Z̄ in the mirror arrangement (both patches bulk
    phase 1)."""
    return build_wall_case(d, ('Z', 'X', 1), ('X', 'Z', 1),
                           mirror_mixed_wall_spec(d),
                           init_bot, init_top, init_bot, init_top)


def _check(builder, d, n_obs):
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


@pytest.mark.parametrize("d", [3, 5])
def test_same_type_wall(d):
    # X-basis init of both patches: X̄b and X̄t individually deterministic,
    # joint measured by the wall -> 2 observables
    _check(build_same_type_case(d, 'X', 'X'), d, 2)


@pytest.mark.parametrize("d", [3, 5])
def test_same_type_wall_teleport(d):
    # Z init bottom / X init top: only the wall outcome links them -> 1 obs
    _check(build_same_type_case(d, 'Z', 'X'), d, 1)


@pytest.mark.parametrize("d", [3, 5])
def test_mirror_mixed_wall(d):
    # this arrangement's carriers: bottom rect('Z','X',1) contributes Z̄,
    # top rect('X','Z',1) contributes X̄ — aligned init is (Z, X)
    _check(build_mirror_mixed_case(d, 'Z', 'X'), d, 2)


@pytest.mark.parametrize("d", [3, 5])
def test_mirror_mixed_wall_teleport(d):
    _check(build_mirror_mixed_case(d, 'X', 'Z'), d, 1)
