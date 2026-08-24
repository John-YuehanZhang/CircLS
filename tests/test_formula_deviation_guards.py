"""Zero-observable guards in the B-dev sampling helpers.

A circuit with no observable (e.g. qrng_n4 / graphstate_* under the
no-trick compile: every terminal measurement conjugates to a single-qubit
Pauli, no dagger row) has no defined LER.  The pymatching paths used to XOR
predictions against a width-0 observable array, count zero failures and
store a spurious 0.0 instead of raising (fix 2026-08-24); the mwpf fallback
always refused.  Pin the refusal on every path."""
import pathlib
import sys

import pytest
import stim

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]
                       / "experiments"))


def test_sample_ler_refuses_zero_observable_circuit():
    from formula_deviation import _sample_ler
    c = stim.Circuit("R 0\nM 0")
    assert c.num_observables == 0
    with pytest.raises(ValueError, match="no observables"):
        _sample_ler(c, seed=1, target_errors=1, max_shots=100)


def test_faithful_audit_refuses_zero_observable_circuit():
    from formula_deviation import _faithful_audit
    c = stim.Circuit("R 0\nM 0\nDETECTOR rec[-1]")
    assert c.num_observables == 0
    with pytest.raises(ValueError, match="no observables"):
        _faithful_audit(c)
