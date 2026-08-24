"""Square Sparse mapping + promoted QASM pipeline.

Floor formula transcribed from DASCOT (OOPSLA 2025, Sec. 6): side length
``2*ceil(sqrt(n)) + 1``, every data patch surrounded by routing cells.
Fault-distance preservation on this floor is covered by the notebook suite
and probe runs (ghz3 / s-twist both = d); here we pin the cheap invariants.
"""
import pytest

from circls.pipeline import compile_qasm
from circls.compiler.mapping import grid_side, place_patches, square_sparse_cells
from circls.metrics import experiment_stats

pytestmark = pytest.mark.smoke


def ghz(n):
    s = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[%d];\nh q[0];\n' % n
    return s + ''.join(f'cx q[{i}],q[{i+1}];\n' for i in range(n - 1))


STWIST = ('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\n'
          'h q[0];\ncx q[0],q[1];\ns q[1];\ncx q[1],q[2];\nsx q[2];\n')


# ── floor formula oracles ─────────────────────────────────────────────────────

def test_grid_side_formula():
    # DASCOT: "a side length of 2⌈√n⌉ + 1"
    assert [grid_side(n) for n in (1, 2, 4, 5, 9, 10)] == [3, 5, 5, 7, 7, 9]


def test_square_sparse_cells_properties():
    for n in range(1, 26):
        cells = square_sparse_cells(n)
        assert len(cells) == n == len(set(cells))
        side = grid_side(n)
        for a, b in cells:
            assert a % 2 == 1 and b % 2 == 1     # odd-odd: routing on all sides
            assert 1 <= a <= side - 2 and 1 <= b <= side - 2
    assert square_sparse_cells(3) == [(1, 1), (3, 1), (1, 3)]


def test_natural_order_assignment():
    specs = place_patches([f"q{i}" for i in range(12)], 3)
    assert [s.name for s in specs] == [f"q{i}" for i in range(12)]  # q10 after q9


def test_orientation_rules():
    specs = place_patches(["q0", "q1", "y0"], 3,
                          first_letters={"q0": "Z", "q1": "X", "y0": "Z"})
    by = {s.name: s.orientation for s in specs}
    assert by["y0"] == "X_vertical"        # Gidney birth v1 constraint wins
    assert by["q0"] == "X_horizontal"      # first use Z
    assert by["q1"] == "X_vertical"        # first use X


# ── end-to-end on the sparse floor ────────────────────────────────────────────

@pytest.mark.parametrize("name, qasm", [
    ("bell2", ghz(2)), ("ghz3", ghz(3)), ("s-twist", STWIST),
])
def test_compiles_silent_and_gated(name, qasm):
    cp = compile_qasm(qasm)
    det, _ = cp.circuit.compile_detector_sampler(seed=0).sample(
        512, separate_observables=True)
    assert not det.any(), f"{name}: noiseless detectors fired"
    es = experiment_stats(cp.experiment, cp.circuit)  # S4==S5 gate runs inside
    assert es.tiles_touched <= es.tiles_bbox
    assert es.circuit.num_observables >= 1


def test_t_gate_rejected():
    with pytest.raises(ValueError):
        compile_qasm('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nt q[0];\n')
