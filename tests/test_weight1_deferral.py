"""Regression for CircLS issue #2 (github.com/John-YuehanZhang/CircLS/issues/2).

With ``t_as_s=True`` and default re-selection, a 4-qubit circuit reduced
from TopoLS's vqe_16 was rejected by ``folded_out_bits`` ("weight-1
measurement ... followed by later operations").  Re-selection makes every
terminal measurement weight-1, two of them with a Y letter; the Y-free
sweep rewrites ``Y2`` by folding the pending pi/4 masks into it, which
widens it to ``-Z1 X2 Z3`` behind the weight-1 ``Z1``.  The new
``defer_weight1_terminal`` pass moves such a weight-1 measurement to the
tail when every op it passes commutes with it.
"""
import contextlib
import io
from pathlib import Path

import pytest

from circls.interop.ir.pauli_algebra import defer_weight1_terminal
from circls.interop.ir.pauli_ir import PauliCircuit, PauliOp
from circls.pipeline import analyze_qasm, compile_qasm
from circls.tools.evaluate import verify
from circls.tools.reporting import (affine_structure, sample_program_bits,
                                      source_truth_structure)

QASM = (Path(__file__).resolve().parent / "fixtures" / "issue2"
        / "vqe16_min22.qasm").read_text()


def _compile(**kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return compile_qasm(QASM, distance=3, t_as_s=True, **kw)


def test_issue2_circuit_compiles_with_default_options():
    cp = _compile()
    r = verify(cp, shots=256)
    assert r.ok and r.logical is True, r
    truth = source_truth_structure(cp.source_qasm, shots=256)
    assert affine_structure(sample_program_bits(cp, 256, 29)) == truth
    # the no-re-selection compile of the same program agrees
    cp0 = _compile(measure_reduction=False)
    assert affine_structure(sample_program_bits(cp0, 256, 29)) == truth
    assert len(cp.experiment.ppm_sequence) == len(cp0.experiment.ppm_sequence)
    # the deferred weight-1 readouts are the last ops on their qubits
    prog = cp.program
    for i, op in enumerate(prog.ops):
        if op.kind == "mpp" and len(op.targets) == 1:
            (q, _), = op.targets.items()
            assert not any(q in later.targets for later in prog.ops[i + 1:])


def test_analyze_qasm_takes_the_same_path():
    an = analyze_qasm(QASM.replace("t q", "s q"))      # Clifford reading of the same gates
    assert an.program.out_bits


def test_pass_defers_past_commuting_ops_and_keeps_bit_order():
    c = PauliCircuit(3, [PauliOp("m", {1: "Z"}),
                         PauliOp("m", {1: "Z", 2: "X"}),
                         PauliOp("s", {0: "Z", 1: "Z"}, -1),
                         PauliOp("m", {0: "X"})])
    out, m_perm = defer_weight1_terminal(c)
    kinds = [(op.kind, dict(op.paulis)) for op in out.ops]
    assert kinds == [("m", {1: "Z", 2: "X"}), ("s", {0: "Z", 1: "Z"}),
                     ("m", {0: "X"}), ("m", {1: "Z"})]
    assert m_perm == [1, 2, 0]          # new m ordinal -> old m ordinal


def test_pass_is_identity_when_nothing_follows():
    c = PauliCircuit(2, [PauliOp("m", {0: "Z", 1: "Z"}), PauliOp("m", {0: "X"})])
    out, m_perm = defer_weight1_terminal(c)
    assert [dict(op.paulis) for op in out.ops] == [{0: "Z", 1: "Z"}, {0: "X"}]
    assert m_perm == [0, 1]


def test_pass_refuses_an_anticommuting_follower():
    c = PauliCircuit(2, [PauliOp("m", {1: "X"}), PauliOp("s", {0: "Z", 1: "Z"}, -1)])
    with pytest.raises(ValueError, match="anticommuting"):
        defer_weight1_terminal(c)
