"""Execution-level equivalence of measurement reduction (user request
2026-08-05): compile each case with the pass ON and OFF, sample the REAL
circuits, assemble the ORIGINAL program bits per shot through every
bookkeeping layer, and compare the deterministic-parity affine structures
— the complete invariant of a stabilizer output distribution.  This
verifies the reconstruction WIRING end to end, not just the front-end
operator algebra (which tests/test_measure_reduce.py already pins).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from benchsuite import (bbpssw_chain, bv, dj, ghz, graph_state_ring,
                        steane_encode, twisted_ghz)

from circls.pipeline import compile_qasm
from circls.tools.reporting import affine_structure, sample_program_bits


def test_affine_structure_hand_example():
    # col0 constant 1, col1 fair coin, col2 = col1 XOR 1
    rng = np.random.default_rng(3)
    c1 = rng.integers(0, 2, 256).astype(bool)
    bits = np.stack([np.ones(256, dtype=bool), c1, ~c1], axis=1)
    masks, vals = affine_structure(bits)
    assert set(zip(masks, vals)) == {(0b001, 1), (0b110, 1)}


CASES = [("ghz_8", lambda: ghz(8)),
         ("graphstate_8", lambda: graph_state_ring(8)),
         ("bv_8", lambda: bv(8)),
         ("dj_8", lambda: dj(8)),
         ("steane", steane_encode),
         ("twistedghz_4", lambda: twisted_ghz(4)),
         ("twistedghz_8", lambda: twisted_ghz(8)),
         ("bbpssw_4", lambda: bbpssw_chain(4))]


# per-case placement keeping BOTH configs compilable (feasibility gaps,
# not equivalence questions): steane+reduction-off is infeasible under
# row_major (3-wall geometry); twistedghz_8+reduction-off is infeasible
# under optimized (q2 blocked beyond single-flip repair)
_PLACEMENT = {"steane": "optimized"}


@pytest.mark.parametrize("name,qasm_fn", CASES)
def test_distribution_identical_with_and_without_reduction(name, qasm_fn):
    qasm = qasm_fn()
    shots = 512
    structs = {}
    for tag, red in (("off", False), ("on", True)):
        cp = compile_qasm(qasm,
                          assignment=_PLACEMENT.get(name, "row_major"),
                          measure_reduction=red)
        bits = sample_program_bits(cp, shots=shots, seed=23)
        structs[tag] = affine_structure(bits)
        n_bits = bits.shape[1]
    m_off, v_off = structs["off"]
    m_on, v_on = structs["on"]
    assert m_off == m_on, (
        f"{name}: deterministic-parity SPACES differ "
        f"({len(m_off)} vs {len(m_on)} constraints over {n_bits} bits)")
    assert v_off == v_on, f"{name}: deterministic VALUES differ"
