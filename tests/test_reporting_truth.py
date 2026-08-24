"""Ground-truth oracle for the program-bit reporting layer.

The logical-level TableauSimulator measures the ORIGINAL terminal
operators directly, giving a compilation-independent joint distribution.
This is a STRICTLY stronger standard than cross-compilation consistency
(two equally-wrong compilations agree with each other — measured
2026-08-05, when the parallel work surfaced that dj_8's serial
extraction was already losing 4 of 7 deterministic constraints).

Green cases pin the full extraction surface: folded readouts, gadget
corrections, and — since the seed-closure/anchor-algebra fix — joint
steps whose ideal value must survive execution reordering (scheduling
hoisted dj_8's X^8 window ahead of four Z windows; the split byproduct
frame made their raw post-closures per-shot random, so those bits are
now reconstructed from virgin-moment seed closures instead).
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import stim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from benchsuite import bv, dj, ghz, twisted_ghz

from circls.interop.nwqec.frontend import load_clifford_mpauli
from circls.pipeline import compile_qasm
from circls.tools.reporting import affine_structure, sample_program_bits

_L = {"X": 1, "Y": 2, "Z": 3}


def truth_structure(qasm, shots=512):
    raw = load_clifford_mpauli(qasm)
    n = raw.num_qubits
    mops = []
    for op in raw.ops:
        ps = stim.PauliString(n)
        for q, letter in op.paulis.items():
            ps[q] = _L[letter]
        if op.sign == -1:
            ps.sign = -1
        mops.append(ps)
    rows = []
    for shot in range(shots):
        ts = stim.TableauSimulator(seed=1000 + shot)
        rows.append([ts.measure_observable(p) for p in mops])
    return affine_structure(np.array(rows, dtype=bool))


def extracted_structure(qasm, **kw):
    cp = compile_qasm(qasm, assignment="optimized", liveness=True,
                      keep_patches=set(), **kw)
    return affine_structure(sample_program_bits(cp, shots=512, seed=29))


CASES = [("ghz_8", lambda: ghz(8)), ("twistedghz_4", lambda: twisted_ghz(4)),
         ("dj_8", lambda: dj(8)), ("bv_8", lambda: bv(8))]


@pytest.mark.parametrize("name,fn", CASES)
def test_extraction_matches_logical_truth(name, fn):
    q = fn()
    assert extracted_structure(q) == truth_structure(q), name
