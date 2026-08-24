"""Direct seams that need TWO orientations of the same patch.

A joint measurement across a direct (cell-adjacent) seam owes the parallel
law: the measured logical must run PARALLEL to the seam line, which fixes
the target's orientation from (seam axis, measured letter) with no freedom
left.  A data patch measured X̄ across one seam and Z̄ across another
therefore needs BOTH orientations over its life — no single birth
orientation serves both, and the patch must physically rotate in between.

That is the shape of every foreign compact layout the replays import
(DASCOT's CNOT gadget sandwiches an H between two CXs on the same data
patch); before 2026-08-16 the planner skipped direct-seam steps outright and
the engine raised ``SeamRuleError: ... must run PARALLEL to the seam``.

Covered here: the two-patch minimal reproduction with the planner on and
off, the protocol choice (litinski when its bar has room, rotate_90
otherwise), the law table itself, the live-footprint probes after a real
litinski rotation (retired rows are not part of the patch), and the snake
gateway declining a corridor whose attach seam breaks the same law.
"""
import pytest

from circls.core.joint_merge import SeamRuleError, patch_view
from circls.core.multi_patch_coupler import PatchSpec, origin_of
from circls.core.sequential_ppm_ls import PPMStep, SequentialPPMExperiment

pytestmark = pytest.mark.smoke

D = 3


def _exp(specs, steps, **kw):
    names = [s.name for s in specs]
    return SequentialPPMExperiment(
        specs, steps,
        initial_states={nm: 'Z' for nm in names},
        final_measure_states={nm: 'Z' for nm in names},
        rounds=D, rounds_init=1, parallel_steps=False,
        rotation_kind='auto', **kw)


def _run(exp, shots=256):
    """Build and return (silent, deterministic)."""
    circuit = exp.build()
    det, obs = circuit.compile_detector_sampler(seed=0).sample(
        shots, separate_observables=True)
    return (not det.any(),
            (obs.shape[1] == 0) or bool((obs == obs[0]).all()))


def _side_by_side():
    """Two patches on cells (0,0) and (1,0) — a VERTICAL seam — measured
    X⊗X and then X⊗Z: q1 owes X_vertical to the first and X_horizontal to
    the second.  q1's litinski bar is blocked by q0's home cell, so the
    repair must reach for rotate_90."""
    specs = [PatchSpec('q0', origin_of(0, 0, D, seam=True), D, 'X_vertical'),
             PatchSpec('q1', origin_of(1, 0, D, seam=True), D, 'X_vertical')]
    steps = [PPMStep([('q0', 'X'), ('q1', 'X')]),
             PPMStep([('q0', 'X'), ('q1', 'Z')])]
    return specs, steps


def test_planner_carries_the_direct_seam_constraint():
    specs, steps = _side_by_side()
    exp = _exp(specs, steps, auto_rotate=True)
    assert _run(exp) == (True, True)
    assert exp.rotation_log == [(1, 'q1', 'rotate_90')]
    # planned, not repaired: the plan itself carries the move
    assert exp._rotate_plan == {1: [('q1', 'rotate_90')]}
    # both seams classified (row 1 same-letter/same-type, row 4 after the
    # colour swap) — the parallel law held at registration
    assert sorted(exp._rules) == [0, 1]


def test_repair_fires_with_the_planner_off():
    """The feasibility-only fallback is independent of ``auto_rotate``: with
    no plan at all the direct-seam registration repairs itself."""
    specs, steps = _side_by_side()
    exp = _exp(specs, steps, auto_rotate=False)
    assert _run(exp) == (True, True)
    assert exp._rotate_plan == {}          # nothing was planned
    assert exp.rotation_log == [(1, 'q1', 'rotate_90')]


def test_litinski_is_preferred_when_its_bar_has_room():
    """Colour-keeping first: q0 sits on cell (1,0) with cell (0,0) empty, so
    its (2d-1)xd bar fits and the repair takes the litinski protocol (which
    flips the TYPE — the seam drops from row 1 to row 3)."""
    specs = [PatchSpec('q0', origin_of(1, 0, D, seam=True), D, 'X_vertical'),
             PatchSpec('q1', origin_of(2, 0, D, seam=True), D, 'X_vertical')]
    steps = [PPMStep([('q0', 'X'), ('q1', 'X')]),
             PPMStep([('q0', 'Z'), ('q1', 'X')])]
    exp = _exp(specs, steps, auto_rotate=True)
    assert _run(exp) == (True, True)
    assert exp.rotation_log == [(1, 'q0', 'litinski')]
    assert exp._rules[0].row == 1 and exp._rules[1].row == 3


def test_no_legal_protocol_still_raises_the_parallel_law_error():
    """``rotation_kind='litinski'`` with a blocked bar leaves no legal move:
    the repair adds nothing and the honest seam error survives."""
    specs, steps = _side_by_side()
    exp = SequentialPPMExperiment(
        specs, steps, initial_states={'q0': 'Z', 'q1': 'Z'},
        final_measure_states={'q0': 'Z', 'q1': 'Z'}, rounds=D, rounds_init=1,
        parallel_steps=False, auto_rotate=True, rotation_kind='litinski')
    with pytest.raises(SeamRuleError, match="PARALLEL"):
        exp.build()


@pytest.mark.parametrize("axis", ['vertical', 'horizontal'])
@pytest.mark.parametrize("pauli", ['X', 'Z'])
def test_parallel_orientation_is_the_rule_table_law(axis, pauli):
    """``_parallel_orientation`` must return exactly the orientation
    ``joint_merge._check_parallel`` accepts, and only that one."""
    from circls.core.joint_merge import _check_parallel

    want = SequentialPPMExperiment._parallel_orientation(pauli, axis)
    for orientation in ('X_vertical', 'X_horizontal'):
        view = type('V', (), {'name': 'q', 'orientation': orientation})()
        if orientation == want:
            _check_parallel(view, pauli, axis)          # must not raise
        else:
            with pytest.raises(SeamRuleError, match="PARALLEL"):
                _check_parallel(view, pauli, axis)


def _rotated_system(d=3):
    """One patch, d rounds of SE, then the real 5-step litinski rotation."""
    from lightstim.ir.builder import CircuitBuilder
    from lightstim.ir.qec_system import QECSystem
    from lightstim.ir.tracker import SyndromeTracker
    from lightstim.qec_code.surface_code.rotated.diagonal_se import (
        DiagonalSurfaceCodeExtractionBlock)
    from lightstim.qec_code.surface_code.rotated.litinski_layouts import start_w2
    from lightstim.qec_code.surface_code.rotated.litinski_rotation import (
        rotate_patches_litinski)
    from lightstim.qec_code.surface_code.rotated.rule_based_patch import (
        RuleBasedRectPatch)

    system = QECSystem()
    system.add_patch(RuleBasedRectPatch(distance_x=d, distance_z=d,
                                        forced=start_w2(d), phase=0),
                     offset=(0, 2 * (d - 1)), name='q2')
    tracker = SyndromeTracker(num_qubits=system.num_qubits,
                              expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker=tracker, system_config=system,
                             if_detector=True)
    system.register_tracker(tracker)
    system.register_builder(builder)
    builder.write_coordinates()
    home = {tuple(system.qubit_coords[q]) for q in sorted(system.data_indices)}
    builder.initialize({q: 'Z' for q in sorted(system.data_indices)},
                       n=system.num_qubits)
    builder.apply_syndrome_extraction(
        DiagonalSurfaceCodeExtractionBlock(system).circuit, rounds=d)
    rotate_patches_litinski(system, builder, ['q2'], directions={'q2': '-y'})
    return system, tracker, home


def test_live_footprint_probes_ignore_retired_rows():
    """A litinski rotation measures out the rows it grew through and leaves
    them owned-but-RETIRED.  Counting those reads the d x d patch as a
    rectangle ('not square (4x3)' / '|data|=15'), at a shifted origin: both
    live-footprint probes must filter them."""
    from lightstim.qec_code.surface_code.rotated.swap_rotation import (
        _prep_rotation)

    system, tracker, home = _rotated_system(D)
    owned = [q for q in system.data_indices
             if system.index_to_owner_map.get(q) == 'q2']
    assert len(owned) > D * D, "the stale rows are the point of this test"
    assert sum(1 for q in owned if q in tracker.retired_qubits) == len(owned) - D * D

    view = patch_view(system, 'q2')
    assert view.distance == D
    assert view.origin == min(home)
    assert view.orientation == 'X_vertical'      # litinski flipped it

    prep = _prep_rotation(system, 'q2')          # rotate_90 on a rotated patch
    assert prep['d'] == D and prep['origin'] == min(home)


def test_snake_gateway_declines_a_parallel_law_illegal_attach_seam():
    """A stretched wall fixes a COLOUR mismatch, never an ORIENTATION one.
    Here the corridor leaves q0's cell downward while q0 (X_horizontal after
    the first step's rotate_90) needs an E/W seam for Z̄: the snake gateway
    must decline so the rotation search takes the step (it used to hand
    ``_register_snake`` a corridor the physics layer cannot host)."""
    specs = [PatchSpec('q0', origin_of(2, 2, D, seam=True), D, 'X_vertical'),
             PatchSpec('q3', origin_of(3, 4, D, seam=True), D, 'X_vertical')]
    steps = [PPMStep([('q0', 'X'), ('q3', 'Z')], route=[(2, 3), (3, 3)]),
             PPMStep([('q0', 'Z'), ('q3', 'Z')], route=[(2, 3), (3, 3)])]
    exp = _exp(specs, steps, auto_rotate=True, liveness=True,
               keep_patches={'q0', 'q3'})
    assert _run(exp) == (True, True)
    assert exp._snake_plans == {}
    assert [nm for _, nm, _ in exp.rotation_log] == ['q0', 'q0']
