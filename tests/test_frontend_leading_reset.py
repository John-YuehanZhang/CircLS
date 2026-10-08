"""Leading resets are the |0> initialisation the compiler performs anyway:
they are accepted and change nothing; a mid-circuit reset is rejected."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from benchsuite import ghz

from circls.interop.nwqec.frontend import load_clifford_mpauli, load_clifford_t_as_s
from circls.pipeline import compile_qasm

GHZ = ghz(3)
GHZ_RESET = GHZ.replace("h q[0];", "reset q[0];\nh q[0];").replace(
    "cx q[0],q[1];", "reset q[1];\ncx q[0],q[1];").replace(
    "cx q[1],q[2];", "reset q[2];\ncx q[1],q[2];")
GHZ_RESET_REG = GHZ.replace("h q[0];", "reset q;\nh q[0];")


def _ops(pc):
    return [(op.kind, op.paulis, op.sign) for op in pc.ops]


@pytest.mark.parametrize("qasm", [GHZ_RESET, GHZ_RESET_REG])
def test_leading_resets_load_as_the_reset_free_program(qasm):
    assert "reset" in qasm
    assert _ops(load_clifford_mpauli(qasm)) == _ops(load_clifford_mpauli(GHZ))
    assert _ops(load_clifford_t_as_s(qasm)) == _ops(load_clifford_t_as_s(GHZ))


def test_leading_resets_compile_to_the_same_circuit():
    a = compile_qasm(GHZ, distance=3, measure_reduction=False)
    b = compile_qasm(GHZ_RESET, distance=3, measure_reduction=False)
    assert str(a.circuit) == str(b.circuit)
    assert a.observables == b.observables


def test_mid_circuit_reset_is_rejected():
    bad = GHZ.replace("cx q[1],q[2];", "cx q[1],q[2];\nreset q[1];")
    with pytest.raises(ValueError, match="mid-circuit reset"):
        load_clifford_mpauli(bad)
    bad_reg = GHZ.replace("cx q[1],q[2];", "cx q[1],q[2];\nreset q;")
    with pytest.raises(ValueError, match="mid-circuit reset"):
        load_clifford_mpauli(bad_reg)


def test_leading_resets_accepted_from_a_file_too(tmp_path):
    path = tmp_path / "ghz_reset.qasm"
    path.write_text(GHZ_RESET)
    assert _ops(load_clifford_mpauli(str(path))) == _ops(load_clifford_mpauli(GHZ))
    assert _ops(load_clifford_t_as_s(str(path))) == _ops(load_clifford_t_as_s(GHZ))
