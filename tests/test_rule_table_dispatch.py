"""End-to-end four-row rule-table dispatch through SequentialPPMExperiment:
two cell-adjacent patches, the step's construction chosen by the live types
and measured Paulis (rows 1/4 = merge through the zero-cell corridor, rows
2/3 = the stretched-stabilizer wall coupler on the diagonal schedule)."""
import contextlib
import io

import pytest

from circls.core.sequential_ppm_ls import PPMStep, SequentialPPMExperiment
from circls.core.routed_multi_patch_ls import PatchSpec, origin_of
from circls.core.joint_merge import SeamRuleError
from lightstim.noise.config import NoiseConfig

pytestmark = pytest.mark.smoke

D = 3
NP = NoiseConfig(p_1q=1e-3, p_2q=1e-3, p_meas=1e-3, p_reset=1e-3, p_idle=1e-3)


def _spec(nm, a, b, o):
    return PatchSpec(nm, origin_of(a, b, D, seam=True), D, o)


def _run(px, target, states, **kw):
    exp = SequentialPPMExperiment(
        px, [PPMStep(target, **kw.pop('step_kw', {}))], initial_states=states,
        final_measure_states=states, rounds=D, rounds_init=1, **kw)
    with contextlib.redirect_stdout(io.StringIO()):
        c = exp.build()
    return exp, c


def _verify(exp, c, row, wall):
    assert exp._rules[0].row == row
    assert (0 in exp._walls) == wall
    det, obs = c.compile_detector_sampler(seed=0).sample(
        512, separate_observables=True)
    assert not det.any(), "detector fired at p=0"
    assert not obs.any(), "observable not deterministic at p=0"
    noisy = exp.builder.build_noisy_circuit(noise_params=NP,
                                            noise_model='circuit_level')
    noisy.detector_error_model(decompose_errors=True)
    assert len(noisy.shortest_graphlike_error()) == D


def test_row1_same_pauli_same_type_plain_merge():
    # both place(X_horizontal) = conjugate type; Z⊗Z through the E/W seam:
    # plain merge (seam line is DATA), bent schedule
    exp, c = _run([_spec("A", 0, 0, "X_horizontal"),
                   _spec("B", 1, 0, "X_horizontal")],
                  [("A", "Z"), ("B", "Z")], {"A": "Z", "B": "Z"})
    assert exp._sched[0] == 'bent'
    _verify(exp, c, row=1, wall=False)


def test_row2_same_pauli_diff_type_wall():
    # the verified stacked pair (checkerboard phases 0 and 1, positions
    # differ): A = place(X_horizontal), B = place(X_vertical) with swapped
    # colours (live X̄ horizontal, textbook positions); X⊗X through the N/S
    # seam -> uniform-domino wall on the diagonal schedule
    exp, c = _run([_spec("A", 0, 0, "X_horizontal"),
                   _spec("B", 0, 1, "X_vertical")],
                  [("A", "X"), ("B", "X")], {"A": "X", "B": "X"},
                  colour_swapped={"B"})
    assert exp._sched[0] == 'diagonal'
    _verify(exp, c, row=2, wall=True)


def test_row3_mixed_diff_type_wall():
    # the verified mirror pair (phases 1,1): A minority -> conjugate type,
    # Z̄ horizontal; B colour-swapped -> textbook type, X̄ horizontal;
    # Z⊗X through the N/S seam -> mixed-domino wall, diagonal schedule
    exp, c = _run([_spec("A", 0, 0, "X_horizontal"),
                   _spec("B", 0, 1, "X_vertical")],
                  [("A", "Z"), ("B", "X")], {"A": "Z", "B": "X"},
                  colour_swapped={"A", "B"})
    assert exp._sched[0] == 'diagonal'
    _verify(exp, c, row=3, wall=True)


def test_row4_mixed_same_type_recoloured_merge():
    # the corridor-classic pair on adjacent cells: both textbook positions,
    # one recoloured by the minority mechanism; X⊗Z through the E/W seam ->
    # recoloured single-cell mixed checks; straight seam -> bent by the
    # schedule policy (diagonal also works, see the forced variant below)
    exp, c = _run([_spec("A", 0, 0, "X_vertical"),
                   _spec("B", 1, 0, "X_vertical")],
                  [("A", "X"), ("B", "Z")], {"A": "X", "B": "Z"},
                  colour_swapped={"B"})
    assert exp._sched[0] == 'bent'
    _verify(exp, c, row=4, wall=False)


def test_wall_step_rejects_bent_schedule():
    with pytest.raises(SeamRuleError, match="bent"):
        _run([_spec("A", 0, 0, "X_horizontal"),
              _spec("B", 0, 1, "X_vertical")],
             [("A", "X"), ("B", "X")], {"A": "X", "B": "X"},
             colour_swapped={"B"}, step_kw={'schedule': 'bent'})


def test_forced_wall_on_same_type_pair_raises():
    # construction='wall' on a row-1 pair: wall_spec refuses (same type has
    # no wall — that seam takes the plain merge)
    with pytest.raises(SeamRuleError, match="row-1"):
        _run([_spec("A", 0, 0, "X_horizontal"),
              _spec("B", 1, 0, "X_horizontal")],
             [("A", "Z"), ("B", "Z")], {"A": "Z", "B": "Z"},
             step_kw={'construction': 'wall'})


def test_bent_chunk_raises_on_kf_records():
    # the silent-hazard guard: se_round_chunk must refuse stretched checks
    from lightstim.ir.qec_system import QECSystem
    from lightstim.ir.qec_patch import QECPatch
    from lightstim.qec_code.surface_code.rotated.bent_joint_se import (
        se_round_chunk)

    class P(QECPatch):
        def _process_params(self):
            self.shift = (0, 0)

        def build(self):
            for q, role in [((1, 1), 'data'), ((3, 1), 'data'),
                            ((2, 2), 'syndrome_x'), ((2, 0), 'syndrome_z'),
                            ((1, 3), 'syndrome_z')]:
                self.add_qubit(q[0], q[1], role=role)
            self.stabilizers.append(
                {'pauli': {(1, 1): 'X', (3, 1): 'X'}, 'type': 'MIXED',
                 'syn_coord': (2, 2),
                 'kf': {'flag': (2, 0), 'shared': (1, 3), 'orient': '+'}})
            self.num_logicals = 0

    system = QECSystem()
    system.add_patch(P(), offset=(0, 0), name="p")
    with pytest.raises(ValueError, match="DiagonalSurfaceCodeExtractionBlock"):
        se_round_chunk(system)


def test_row4_forced_diagonal_full_distance():
    # the recoloured column is diagonal-schedulable through the _M_SLOTS
    # variant tables (it used to be a genuine Sec. II conflict under the
    # per-own-Pauli lookup)
    exp, c = _run([_spec("A", 0, 0, "X_vertical"),
                   _spec("B", 1, 0, "X_vertical")],
                  [("A", "X"), ("B", "Z")], {"A": "X", "B": "Z"},
                  colour_swapped={"B"},
                  step_kw={'schedule': 'diagonal'})
    assert exp._sched[0] == 'diagonal'
    _verify(exp, c, row=4, wall=False)


def test_row3_transposed_ew_wall_cap_slot():
    # M8 regression: X(x)Z through the E/W (vertical) seam — ve/ho pair,
    # same colour phase, no colour swap -> row-3 mixed stretched wall.
    # The cap end follows the alternating-spacing rule:
    # the south end row already carries both patches' corner lobes one gap
    # away (no slot), so the cap must sit at the free NORTH end.  The old
    # far-side-colour proxy put it south: cap + both corner lobes fought
    # over one corner and all three fired at 50% under p=0.
    exp, c = _run([_spec("A", 0, 0, "X_vertical"),
                   _spec("B", 1, 0, "X_horizontal")],
                  [("A", "X"), ("B", "Z")], {"A": "Z", "B": "Z"})
    _verify(exp, c, row=3, wall=True)
    wall = [st for st in exp.system.stabilizers
            if st.get('patch_name') == 'ppm_0']
    caps = [st for st in wall if len(st['data_indices']) == 2]
    assert len(caps) == 1
    assert caps[0]['syn_coord'][1] == 0, \
        f"cap must sit at the free north end, got {caps[0]['syn_coord']}"
