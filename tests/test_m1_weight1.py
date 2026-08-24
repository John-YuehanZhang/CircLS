"""M1 (weight-1 data measurements): reorder-commuting pre-pass + terminal-
readout folding.  Solution A': every weight-1 m op is moved to the tail
(legal: the m_pauli frame commutes — checked pairwise, fail-loud), and after
the sweep each one either stayed weight-1 (-> deferred to the patch's
terminal readout, no step at all) or was fattened by gadget masks (-> an
ordinary joint PPMStep).  No new lattice primitive, no measurement ancilla.
"""
import contextlib
import io

import pytest

from circls.interop.ir.gosc_gadgets import (append_program_observables,
                                         expand_gadgets, folded_out_bits,
                                         to_experiment_inputs)
from circls.interop.ir.pauli_algebra import reorder_weight1_last, y_free_sweep
from circls.interop.ir.pauli_ir import PauliCircuit, PauliOp
from tests.test_yfree_sweep import _joint_distribution

D = 3


def test_reorder_moves_weight1_last_with_perm():
    c = PauliCircuit(3, [PauliOp("m", {0: "X"}, 1),
                         PauliOp("m", {1: "Z"}, -1),
                         PauliOp("m", {1: "Z", 2: "Z"}, 1)])
    r, perm = reorder_weight1_last(c)
    assert [op.paulis for op in r.ops] == [{1: "Z", 2: "Z"}, {0: "X"}, {1: "Z"}]
    assert perm == [2, 0, 1]
    # distribution equivalence under the bit relabelling
    d0, d1 = _joint_distribution(c), _joint_distribution(r)
    for bits, prob in d0.items():
        assert abs(d1[tuple(bits[perm[k]] for k in range(3))] - prob) < 1e-10


def test_reorder_rejects_anticommuting_frame():
    c = PauliCircuit(1, [PauliOp("m", {0: "X"}, 1), PauliOp("m", {0: "Z"}, 1)])
    with pytest.raises(ValueError, match="anticommute"):
        reorder_weight1_last(c)


def test_folded_out_bits_and_reuse_rejection():
    # repro-3 shape, reordered: [Z1Z2, X0, Z1] -> bits 1 and 2 fold
    ordered = PauliCircuit(3, [PauliOp("m", {1: "Z", 2: "Z"}, 1),
                               PauliOp("m", {0: "X"}, 1),
                               PauliOp("m", {1: "Z"}, 1)])
    prog = expand_gadgets(y_free_sweep(ordered))
    assert folded_out_bits(prog) == {1: (0, "X"), 2: (1, "Z")}
    # un-reordered: the weight-1 Z1 precedes Z1Z2 -> loud rejection
    bad = PauliCircuit(3, [PauliOp("m", {1: "Z"}, 1),
                           PauliOp("m", {1: "Z", 2: "Z"}, 1)])
    with pytest.raises(ValueError, match="weight-1"):
        folded_out_bits(expand_gadgets(y_free_sweep(bad)))


def _chain(qasm, distance=3, rounds=3, noise=None):
    from circls.interop.nwqec.frontend import load_clifford_mpauli
    from circls.core.sequential_ppm_ls import SequentialPPMExperiment
    ordered, perm = reorder_weight1_last(load_clifford_mpauli(qasm))
    prog = expand_gadgets(y_free_sweep(ordered))
    specs, steps, init, final, _ = to_experiment_inputs(prog, distance=distance)
    exp = SequentialPPMExperiment(specs, steps, initial_states=init,
                                  final_measure_states=final, rounds=rounds,
                                  rounds_init=1, noise_params=noise)
    with contextlib.redirect_stdout(io.StringIO()):
        c = exp.build()
    return c, exp, prog, perm


def test_repro3_end_to_end():
    """`h q0; cx q1,q2; s q1` — the ordinary Clifford that used to crash the
    pipeline at PPM 0 (single-patch step).  Now: one joint step + two folded
    bits; silent, decomposable, full distance."""
    pytest.importorskip("nwqec")
    from lightstim.noise.config import NoiseConfig
    QASM = ('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\n'
            'h q[0];\ncx q[1],q[2];\ns q[1];\n')
    c, exp, prog, perm = _chain(QASM)
    obs_map = append_program_observables(c, exp, prog)
    # bit 1 (= original m0 = X0 on |0>) is a free coin; the other two are
    # deterministic — one joint closure, one folded terminal readout
    assert perm == [2, 0, 1]
    assert obs_map[1] is None
    assert obs_map[0] is not None and obs_map[2] is not None
    det, obs = c.compile_detector_sampler(seed=2).sample(
        2048, separate_observables=True)
    assert not det.any()
    for j in range(c.num_observables):
        assert (obs[:, j] == obs[0, j]).all()
    NP = NoiseConfig(p_1q=1e-3, p_2q=1e-3, p_meas=1e-3, p_reset=1e-3,
                     p_idle=1e-3)
    cn, expn, progn, _ = _chain(QASM, noise=NP)
    append_program_observables(cn, expn, progn)
    cn.detector_error_model(decompose_errors=True)
    assert len(cn.shortest_graphlike_error()) == D


def test_random_cliffords_compile_through():
    """The input class that used to crash 236/400: random Cliffords with
    weight-1 terminal measurements now compile and stay silent."""
    pytest.importorskip("nwqec")
    import random
    from tests.test_gosc_pauli_algebra import _random_clifford
    rng = random.Random(2026)
    built = 0
    for _ in range(6):
        qasm, _ = _random_clifford(rng, 3, 12)
        c, exp, prog, _ = _chain(qasm)
        append_program_observables(c, exp, prog)
        det, obs = c.compile_detector_sampler(seed=1).sample(
            512, separate_observables=True)
        assert not det.any()
        for j in range(c.num_observables):
            assert (obs[:, j] == obs[0, j]).all()
        built += 1
    assert built == 6
