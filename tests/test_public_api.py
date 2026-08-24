"""The public surface: top-level imports, verify, measure_ler,
decision accessors.  Everything here is what an external user calls
first, so it must work from ``import circls`` alone."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from benchsuite import ghz

from circls import compile_qasm, measure_ler, verify

QASM = ghz(8)


def test_verify_full_pipeline_ghz():
    out = compile_qasm(QASM, distance=3)
    r = verify(out)
    assert r.silent and r.deterministic
    assert r.distance is True
    assert r.logical is True
    assert r.ok


def test_verify_no_trick_ghz():
    out = compile_qasm(QASM, distance=3, measure_reduction=False)
    assert verify(out).ok


def test_measure_ler_returns_stats():
    out = compile_qasm(QASM, distance=3)
    st = measure_ler(out, p=2e-3, max_errors=10, max_shots=20_000,
                     num_workers=2)
    assert st.logical_error_rate >= 0


def test_decision_accessors():
    out = compile_qasm(QASM, distance=3, measure_reduction=False)
    assert set(out.placement) >= {f"q{i}" for i in range(8)}
    assert out.lifetimes, "joint steps must give a lifetime table"
    assert len(out.routes) == len(out.experiment.ppm_sequence)
    assert out.stats()["V1_volume_blocks"] > 0
