"""Pure-LightStim core of the interop layer: IR + direct-joint 2-colour check.
No external (nwqec / lsqecc) dependency — exercises only circls.interop's
pauli_ir + ppm_import against the real SequentialPPMExperiment planner."""
import pytest

from circls.interop import PauliOp, PauliCircuit, to_ppm_steps

pytestmark = pytest.mark.smoke


def _circ(num_qubits, ops):
    return PauliCircuit(num_qubits, [PauliOp("m", dict(p)) for p in ops])


def test_ir_basics():
    op = PauliOp("t", {0: "X", 1: "Z"})
    assert op.qubits == [0, 1]
    assert op.weight == 2
    assert op.is_mixed() and not op.has_y()
    assert PauliOp("t", {0: "X", 1: "Y"}).has_y()
    assert not PauliOp("t", {0: "X", 1: "X"}).is_mixed()
    with pytest.raises(ValueError):
        PauliOp("q", {0: "X"})              # bad kind
    with pytest.raises(ValueError):
        PauliOp("t", {0: "A"})              # bad pauli


def test_to_ppm_steps_rejects_y():
    circ = PauliCircuit(2, [PauliOp("t", {0: "X", 1: "Y"})])
    with pytest.raises(ValueError, match="X/Z-only"):
        to_ppm_steps(circ)


def test_rotations_and_measurements_mix():
    # a t-rotation followed by measurements: every op becomes one PPMStep
    circ = PauliCircuit(3, [
        PauliOp("t", {0: "X", 1: "X"}),
        PauliOp("t", {0: "Z", 2: "X"}),
        PauliOp("m", {1: "Z", 2: "Z"}),
    ])
    steps = to_ppm_steps(circ)
    assert len(steps) == 3
    assert steps[2].interaction_type == [("q1", "Z"), ("q2", "Z")]
