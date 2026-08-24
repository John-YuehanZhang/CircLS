"""Assignment optimizer: metric oracles,
exhaustive-optimum check, end-to-end wins, determinism."""
import itertools

import pytest

from circls.compiler.assignment import (ExactScorer, _Program, legal_axis,
                               optimize_assignment, pair_cost, pair_cost_law)
from circls.pipeline import compile_qasm
from circls.metrics import experiment_stats

pytestmark = pytest.mark.smoke


# ── closed-form metric oracles (probe-verified formulas) ──────────────────────

def test_pair_cost_closed_form():
    assert pair_cost((1, 1), (3, 1)) == 1      # adjacent slots: single seam
    assert pair_cost((1, 1), (3, 3)) == 3      # diagonal neighbour
    assert pair_cost((1, 1), (5, 1)) == 5      # axis-aligned, lane detour +2
    assert pair_cost((1, 1), (5, 5)) == 7
    assert pair_cost((1, 1), (9, 1)) == 9      # +2 is flat, not per blocker


def test_parallel_law_only_bites_adjacent():
    # Z on X_horizontal is legal on E/W seams
    assert legal_axis("Z", "X_horizontal") == "EW"
    assert legal_axis("X", "X_horizontal") == "NS"
    ew, ns = "EW", "NS"
    # adjacent horizontally: legal axes -> no penalty; NS-wanting endpoint -> +2
    assert pair_cost_law((1, 1), (3, 1), ew, ew) == 1
    assert pair_cost_law((1, 1), (3, 1), ns, ew) == 3
    # distance >= 2: the law never costs anything (probe-measured)
    assert pair_cost_law((1, 1), (5, 1), ns, ns) == pair_cost((1, 1), (5, 1))


# ── exhaustive optimum on a tiny instance ─────────────────────────────────────

def test_ring4_reaches_exhaustive_optimum():
    names = [f"q{i}" for i in range(4)]
    steps = [[("q0", "Z"), ("q1", "Z")], [("q1", "Z"), ("q2", "Z")],
             [("q2", "Z"), ("q3", "Z")], [("q0", "Z"), ("q3", "Z")]]
    cells, _ = optimize_assignment(names, steps)
    slots = [(1, 1), (3, 1), (1, 3), (3, 3)]
    prog = _Program(names, steps, {})
    scorer = ExactScorer(slots)
    optimum = min(scorer.program_cost(prog, list(p))
                  for p in itertools.permutations(slots))
    assert scorer.program_cost(prog, [cells[nm] for nm in names]) == optimum


def test_chain9_snake_optimum():
    # 8 chain steps can all be slot-adjacent (a snake through the 3x3 grid):
    # exact cost 8 is the floor, and the optimizer reaches it.
    names = [f"q{i}" for i in range(9)]
    steps = [[(f"q{i}", "Z"), (f"q{i+1}", "Z")] for i in range(8)]
    cells, report = optimize_assignment(names, steps)
    assert min(report.values()) == 8


# ── end-to-end through the pipeline ───────────────────────────────────────────

CROSS6 = ('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[6];\n'
          'h q[0];\nh q[2];\nh q[4];\n'
          'cx q[0],q[5];\ncx q[5],q[0];\ncx q[2],q[3];\ncx q[0],q[5];\n'
          'cx q[1],q[4];\ncx q[4],q[1];\ncx q[1],q[4];\n')


def test_optimized_beats_row_major_on_cross6():
    # measure_reduction collapses CROSS6's terminal set to weight-1 (zero
    # corridors under BOTH policies) — pin it off; this test exercises the
    # assignment optimizer on joint steps, which need corridors to exist
    stats = {}
    corridor = {}
    for policy in ("row_major", "optimized"):
        cp = compile_qasm(CROSS6, assignment=policy, measure_reduction=False)
        det, _ = cp.circuit.compile_detector_sampler(seed=0).sample(
            512, separate_observables=True)
        assert not det.any()
        stats[policy] = experiment_stats(cp.experiment, cp.circuit)
        corridor[policy] = sum(len(r) for r in cp.routes)
    # measured 2026-08-04: 5 -> 1 corridor cells; pin the direction, not the
    # exact values (routing internals may shave a cell either way)
    assert corridor["optimized"] < corridor["row_major"]
    assert stats["optimized"].volume_blocks < stats["row_major"].volume_blocks
    assert (stats["optimized"].circuit.measurement_layers
            == stats["row_major"].circuit.measurement_layers)  # time unchanged


def test_optimized_deterministic_byte_identical():
    def ghz4():
        return ('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[4];\nh q[0];\n'
                'cx q[0],q[1];\ncx q[1],q[2];\ncx q[2],q[3];\n')
    a = compile_qasm(ghz4(), assignment="optimized").circuit
    b = compile_qasm(ghz4(), assignment="optimized").circuit
    assert str(a) == str(b)


def test_unknown_policy_rejected():
    with pytest.raises(ValueError, match="assignment"):
        compile_qasm(CROSS6, assignment="mcts")
