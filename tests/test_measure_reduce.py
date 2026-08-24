"""Terminal measurement-set re-selection: algebraic oracle + regression guards.

The completeness oracle is exact operator identity (no sampling): every
original m_k must equal (-1)^const[k] times the stim product of its
reconstruction support over the executed set.
"""
import sys
from pathlib import Path

import pytest
import stim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from benchsuite import (bbpssw_chain, bv, ghz, random_clifford, steane_encode,
                        twisted_ghz)

from circls.compiler.measure_reduce import _to_stim, reduce_measurements
from circls.interop.nwqec.frontend import load_clifford_mpauli
from circls.interop.ir.pauli_ir import PauliCircuit, PauliOp
from circls.pipeline import compile_qasm


def _oracle(raw, red, recon):
    n = raw.num_qubits
    execs = [_to_stim(o, n) for o in red.ops]
    for k, orig in enumerate(raw.ops):
        prod = stim.PauliString(n)
        for j in sorted(recon.inverse[k]):
            prod = prod * execs[j]
        want = _to_stim(orig, n)
        assert prod == (want if recon.const[k] == 0 else -want), \
            f"reconstruction identity fails on measurement {k}"


@pytest.mark.parametrize("qasm_fn", [lambda: ghz(16), steane_encode,
                                     lambda: twisted_ghz(8), lambda: bv(16)])
def test_oracle_identity(qasm_fn):
    raw = load_clifford_mpauli(qasm_fn())
    red, recon = reduce_measurements(raw)
    _oracle(raw, red, recon)
    assert max(len(o.paulis) for o in red.ops) <= \
        max(len(o.paulis) for o in raw.ops)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_oracle_random_clifford(seed):
    raw = load_clifford_mpauli(random_clifford(6, 20, seed=seed))
    red, recon = reduce_measurements(raw)
    _oracle(raw, red, recon)


def test_ghz_collapses_to_weight1():
    # {X0, X0Z1, ..., X0Z15} * pairwise -> {X0, Z1, ..., Z15}
    red, _ = reduce_measurements(load_clifford_mpauli(ghz(16)))
    assert max(len(o.paulis) for o in red.ops) == 1


def test_steane_max_weight_drops():
    raw = load_clifford_mpauli(steane_encode())
    red, _ = reduce_measurements(raw)
    assert max(len(o.paulis) for o in raw.ops) == 6
    assert max(len(o.paulis) for o in red.ops) == 3


def test_anticommuting_guard():
    bad = PauliCircuit(1, [PauliOp("m", {0: "X"}), PauliOp("m", {0: "Z"})])
    with pytest.raises(ValueError, match="anticommute"):
        reduce_measurements(bad)


def test_non_measurement_guard():
    bad = PauliCircuit(1, [PauliOp("s", {0: "Z"})])
    with pytest.raises(ValueError, match="m-only"):
        reduce_measurements(bad)


def _silent(cp, shots=128):
    det, _ = cp.circuit.compile_detector_sampler(seed=0).sample(
        shots, separate_observables=True)
    return not det.any()


def test_steane_row_major_compiles():
    # pre-reduction this was BentLayoutError: PPM 4 (weight 6, 3 walls
    # needed) had no feasible candidate under row_major placement
    cp = compile_qasm(steane_encode())
    assert cp.reconstruction is not None
    assert _silent(cp)


def test_twisted_ghz8_compiles():
    # pre-reduction: litinski post-rotation collision at (12, 14)
    cp = compile_qasm(twisted_ghz(8))
    assert _silent(cp)


def test_ghz_needs_no_joint_steps():
    cp = compile_qasm(ghz(8))
    assert len(cp.experiment.ppm_sequence) == 0
    assert _silent(cp)


def test_reduction_off_flag():
    cp = compile_qasm(ghz(4), measure_reduction=False)
    assert cp.reconstruction is None
    assert _silent(cp)


def test_bbpssw_still_compiles():
    cp = compile_qasm(bbpssw_chain(4))
    assert _silent(cp)
