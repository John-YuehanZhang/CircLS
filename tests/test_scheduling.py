"""Lifetime-aware step scheduling: DAG safety, star optimality, no-worse
guarantee, end-to-end resource wins with observables preserved."""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from benchsuite import dj, teleport_chain

from circls.interop.ir.pauli_ir import PauliCircuit, PauliOp
from circls.pipeline import compile_qasm
from circls.metrics import experiment_stats
from circls.metrics.report import to_dict
from circls.compiler.scheduling import _lifetime_sum, schedule_ops


def _m(paulis):
    return PauliOp("m", paulis)


def _cost(circ):
    return _lifetime_sum(list(range(len(circ.ops))), circ.ops)


_TOPOLS_GHZ = Path(os.environ.get(
    "TOPOLS_DIR",
    str(Path(__file__).resolve().parents[2] / "TopoLS"))) / "docs/benchmark/ghz_16.qasm"
needs_topols = pytest.mark.skipif(
    not _TOPOLS_GHZ.exists(),
    reason="needs a TopoLS checkout (TOPOLS_DIR); see CONTRIBUTING")


def test_star_moves_hub_to_median():
    # 7 pairs (q_i, q7) + one big op over everyone, big LAST — the dj shape.
    ops = [_m({i: "Z", 7: "Z"}) for i in range(7)]
    ops.append(_m({i: "Z" for i in range(8)}))
    new, perm = schedule_ops(PauliCircuit(8, ops))
    big_pos = perm.index(7)
    assert big_pos in (3, 4), f"hub not at median: {big_pos}"
    # optimum: satellites around the median slot (hub occupies one slot, so
    # distances are 3+2+1+1+2+3+4 = 16) plus the hub patch's own span 7
    assert _cost(new) == 16 + 7
    assert _cost(new) < _cost(PauliCircuit(8, ops))


def test_anticommuting_pairs_keep_order():
    # anticommuting pairs must keep their relative order whatever the
    # objective prefers; commuting ops may still move around them
    ops = [_m({0: "X", 1: "X"}), _m({1: "Z", 2: "Z"}),
           _m({3: "Z", 4: "Z"}), _m({0: "X", 3: "X"})]
    new, perm = schedule_ops(PauliCircuit(5, ops))
    assert perm.index(0) < perm.index(1)


def test_never_worse_and_deterministic():
    ops = [_m({0: "Z", 1: "Z"}), _m({2: "Z", 3: "Z"}),
           _m({0: "Z", 2: "Z"}), _m({1: "Z", 3: "Z"})]
    c = PauliCircuit(4, ops)
    n1, p1 = schedule_ops(c)
    n2, p2 = schedule_ops(c)
    assert p1 == p2
    assert _cost(n1) <= _cost(c)


def test_weight1_ops_stay_put():
    ops = [_m({0: "Z", 1: "Z"}), _m({4: "Z"}), _m({0: "Z", 2: "Z"}),
           _m({1: "Z", 2: "Z"}), _m({3: "Z"})]
    new, perm = schedule_ops(PauliCircuit(5, ops))
    assert perm[1] == 1 and perm[4] == 4     # fixed ops keep positions


def _resources(qasm, **kw):
    cp = compile_qasm(qasm, assignment="optimized", **kw)
    det, _ = cp.circuit.compile_detector_sampler(seed=0).sample(
        256, separate_observables=True)
    d = to_dict(experiment_stats(cp.experiment, cp.circuit))
    return (cp.circuit.num_observables, bool(not det.any()),
            d["V1_volume_blocks"])


def test_dj8_scheduled_liveness_wins():
    obs0, s0, v0 = _resources(dj(8), step_scheduling=False)
    obs1, s1, v1 = _resources(dj(8), liveness=True, keep_patches=set())
    assert obs1 == obs0 and s0 and s1
    assert v1 < v0, f"no V1 win: {v0} -> {v1}"


@needs_topols
def test_teleport_pinned_off_in_benchmark_config():
    # the decoder-faithfulness audit pins the teleport family's scheduling
    # OFF in every benchmark config (geometric offender; ler.py enforces
    # the general rule per point via its gate + fallback)
    from ablation import compile_kwargs
    from benchsuite import suite
    case = next(c for c in suite(include_qasmbench=False)
                if c.name == "teleport_4")
    assert compile_kwargs("full", case)["step_scheduling"] is False
