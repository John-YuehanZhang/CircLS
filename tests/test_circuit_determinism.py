"""PYTHONHASHSEED determinism regression (upstream #82 fix, ported):
the emitted circuit text must be byte-identical across process hash
seeds.  Guards the sorted() materializations in lightstim/ir/tableau.py
and lightstim/ir/tracker.py against rotting back into list(set)."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

_ROOT = str(Path(__file__).resolve().parents[1])
_SCRIPT = (
    "import contextlib, io, sys\n"
    f"sys.path.insert(0, {_ROOT!r})\n"
    f"sys.path.insert(0, {_ROOT!r} + '/experiments')\n"
    "from benchsuite import ghz\n"
    "from circls.pipeline import compile_qasm\n"
    "with contextlib.redirect_stdout(io.StringIO()):\n"
    "    cp = compile_qasm(ghz(4), distance=3, measure_reduction=False)\n"
    "sys.stdout.write(str(cp.circuit))\n")


def _circuit_text(hash_seed: int) -> str:
    env = dict(os.environ, PYTHONHASHSEED=str(hash_seed))
    r = subprocess.run([sys.executable, "-c", _SCRIPT],
                       capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0, r.stderr[-800:]
    assert r.stdout, "build produced no circuit text"
    return r.stdout


def test_circuit_text_deterministic_across_hash_seeds():
    assert _circuit_text(0) == _circuit_text(1)
