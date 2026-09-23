"""The public noise pass (circls.tools.evaluate.inject_uniform_noise) and the
paper's pass (experiments/noise_inject.py) must give byte-identical output on
every circuit the paper measured: same idle noise in empty TICK spans, in a
leading or trailing span, and in a first span whose only mention of a qubit
is QUBIT_COORDS."""
import sys
from pathlib import Path

import stim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from benchsuite import ghz
from noise_inject import inject_uniform_noise as paper_pass

from circls import compile_qasm
from circls.tools.evaluate import inject_uniform_noise as public_pass

P = 1e-3
HANDMADE = [
    "R 0\nTICK\nTICK\nH 0\nTICK\nM 0",                       # empty span
    "TICK\nR 0\nTICK\nM 0",                                  # leading TICK
    "R 0\nTICK\nM 0\nTICK",                                  # trailing TICK
    "QUBIT_COORDS(0, 1) 1\nR 0\nTICK\nR 1\nTICK\nCX 0 1\nTICK\nM 0 1",  # coords-only first span
    "R 0 1\nTICK\nH 0\nDETECTOR rec[-1]\nTICK\nTICK\nRX 1\nTICK\nMX 1\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]",
    "REPEAT 3 {\nR 0\nTICK\nTICK\nM 0\nTICK\n}",
]


def test_handmade_spans_identical():
    for text in HANDMADE:
        c = stim.Circuit(text)
        assert str(public_pass(c, P)) == str(paper_pass(c, P)), text


def test_compiled_program_identical():
    out = compile_qasm(ghz(4), distance=3)
    c = out.circuit
    a, b = public_pass(c, P), paper_pass(c, P)
    assert str(a) == str(b)
    assert a.detector_error_model() == b.detector_error_model()
