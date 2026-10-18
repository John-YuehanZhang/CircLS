"""Regression for CircLS issue #1 (github.com/John-YuehanZhang/CircLS/issues/1).

``verify()`` reported ``logical=False`` on correct compilations of pure
Clifford circuits whose PPM program contains both X-type merges and the
pi/4 gadgets inserted by the Y-free rewrite: the record-level
reconstruction of the program bits mis-accounted the surgery byproducts
and read a deterministic bit as random.  Re-selection was only the
trigger on the 6-gate case (the re-selected ``-YYI`` needs a gadget);
the larger circuits fail with ``measure_reduction=False`` as well.

The checks are now record-free (reporting.py): a logical-level
simulation of the expanded program against the source truth, and the
physical circuit's expected logical stabilizers peeked before the
terminal readout.  Negative controls keep both checks non-vacuous.
"""
import copy
import contextlib
import io
from pathlib import Path

import pytest

from circls.pipeline import compile_qasm
from circls.tools.evaluate import verify
from circls.tools.reporting import (affine_structure,
                                      logical_program_structure,
                                      logical_state_check,
                                      sample_program_bits,
                                      source_truth_structure)

FIX = Path(__file__).resolve().parent / "fixtures" / "issue1"


def _compile(name, **kw):
    qasm = (FIX / name).read_text()
    with contextlib.redirect_stdout(io.StringIO()):
        return compile_qasm(qasm, distance=3, **kw), qasm


@pytest.mark.parametrize("reduce", [True, False], ids=["reselect", "no_reselect"])
def test_min6_verify_and_readout(reduce):
    """The issue's delta-debugged 6-gate circuit: default options used to
    give logical=False; measure_reduction=False passed (no gadget)."""
    cp, qasm = _compile("min6.qasm", measure_reduction=reduce)
    assert (len(cp.program.gadgets) > 0) == reduce
    r = verify(cp)
    assert r.ok and r.logical is True, r
    assert r.logical_program is True and r.logical_state is True, r
    assert not any(x in r.skipped for x in ("logical_program",
                                            "logical_state")), r
    truth = source_truth_structure(qasm)
    assert affine_structure(sample_program_bits(cp, 256, 29)) == truth


@pytest.mark.parametrize("reduce", [True, False], ids=["reselect", "no_reselect"])
def test_n8_relabelled_verify(reduce):
    """8 qubits, 101 gates (the issue's follow-up): fails with and without
    re-selection on the record-level reader (6 and 4 gadgets)."""
    cp, qasm = _compile("mr_n8_d16_t0.0_s1__relabel.qasm",
                        measure_reduction=reduce)
    assert len(cp.program.gadgets) > 0
    r = verify(cp)
    assert r.ok and r.logical is True, r
    assert affine_structure(sample_program_bits(cp, 256, 29)) == \
        source_truth_structure(qasm)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["rc_n12_d16_t0.0_cnn_s1.qasm",
                                  "rc_n12_d32_t0.0_cnn_s0.qasm"])
@pytest.mark.parametrize("reduce", [True, False], ids=["reselect", "no_reselect"])
def test_n12_verify(name, reduce):
    cp, _ = _compile(name, measure_reduction=reduce)
    r = verify(cp)
    assert r.ok and r.logical is True, r


def test_program_check_catches_dropped_correction():
    """Negative control for logical_program: remove the gadget corrections
    of out bit 1 and the structure must no longer match the truth."""
    cp, qasm = _compile("min6.qasm")
    truth = source_truth_structure(qasm)
    assert logical_program_structure(cp) == truth
    ob = cp.program.out_bits[1]
    assert ob.gadgets, "bit 1 is the gadget-corrected bit of this program"
    ob.gadgets = []
    assert logical_program_structure(cp) != truth


def test_program_check_catches_wrong_kappa():
    cp, qasm = _compile("min6.qasm")
    truth = source_truth_structure(qasm)
    cp.program.gadgets[0].kappa ^= 1
    assert logical_program_structure(cp) != truth


def test_state_check_catches_missing_gadget():
    """Negative control for logical_state: an expected program with the
    second gadget removed is not what the circuit realises, so some
    expected stabilizer peeks at 0.  The control works because the
    gadgets of this program act after the first X0X1 merge, on a state
    that is not an eigenstate of their masks; a gadget whose mask
    stabilises the state is quantum-trivial and invisible to the check
    (as it is to the output distribution)."""
    cp, _ = _compile("min6.qasm")
    good = logical_state_check(cp)
    assert good.ok is True and good.min_abs_expectation == 1, good
    wrong = copy.copy(cp)
    prog = copy.deepcopy(cp.program)
    g = prog.gadgets[1]
    drop = {i for i, op in enumerate(prog.ops)
            if g.ancilla in op.targets}
    assert len(drop) == 3                   # init_y, joint, X readout
    prog.ops = [op for i, op in enumerate(prog.ops) if i not in drop]
    wrong.program = prog
    bad = logical_state_check(wrong)
    assert bad.ok is False and bad.min_abs_expectation == 0, bad


def test_state_check_sees_gates_before_the_terminal_readout():
    """Review finding: the physical simulation must apply every gate up to
    the first terminal-readout record.  A transversal H on one data patch
    inserted right before the readout leaves the code space; the state
    check must report it (silent catches it too, via the readout
    detectors), and a circuit with only a TICK inserted there must still
    pass."""
    import stim
    cp, _ = _compile("min6.qasm")
    base = logical_state_check(cp)
    assert base.ok is True, base
    cut = base.cut_record
    owner = cp.experiment.system.index_to_owner_map
    q0 = sorted(q for q in cp.experiment.system.data_indices
                if owner.get(q) == "q0")

    def with_insert(instr):
        out, rec = stim.Circuit(), 0
        inserted = False
        for inst in cp.circuit.flattened():
            k = inst.num_measurements
            if k and rec == cut and not inserted:
                out.append(instr)
                inserted = True
            out.append(inst)
            rec += k
        assert inserted
        return out

    bad = copy.copy(cp)
    bad.circuit = with_insert(stim.CircuitInstruction("H", q0))
    assert logical_state_check(bad).ok is False
    fine = copy.copy(cp)
    fine.circuit = with_insert(stim.CircuitInstruction("TICK", []))
    assert logical_state_check(fine).ok is True


def test_state_check_skips_when_no_terminal_readout_is_found():
    """Review finding: an instruction after the final readout used to make
    verify() raise; it must report the state check as skipped."""
    cp, _ = _compile("min6.qasm")
    odd = copy.copy(cp)
    c = cp.circuit.copy()
    c.append("I", [0])
    odd.circuit = c
    rep = logical_state_check(odd)
    assert rep.ok is None and "last run" in rep.detail, rep
    r = verify(odd)
    assert r.logical_state is None and "logical_state" in r.skipped, r
