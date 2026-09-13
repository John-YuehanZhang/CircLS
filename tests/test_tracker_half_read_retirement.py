"""Liveness retirement readouts that half-read two standing logical rows.

A patch readout that commutes with two standing logical rows can determine
their PRODUCT without determining either row: the tracker then carries two
standing rows for one degree of freedom and the next WriteBack / add_patch
raises "Logical Count Mismatch".  Measured 2026-09-06 on QASMBench simon_n6
(T -> S proxy) with parallel_steps=False; the |+> ancilla variant fails on
the default schedule too.  The opt-in resolution (env
on by default; LIGHTSTIM_RESOLVE_DEPENDENT_LOGICALS=0 opts out) emits the resolved parity as an
observable and retires the dependent row, only on the patch readout that
half-read it.
"""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SIMON_N6 = """OPENQASM 2.0; include "qelib1.inc"; qreg q[6]; h q[0]; h q[1]; h q[2]; cx q[2], q[4]; x q[3];
cx q[2], q[3]; h q[3]; cx q[1], q[3]; t q[3]; cx q[0], q[3]; t q[3]; cx q[1], q[3]; t q[3]; cx q[0], q[3];
t q[1]; t q[3]; cx q[0], q[1]; t q[0]; t q[1]; cx q[0], q[1]; h q[3]; x q[0]; x q[1]; h q[3]; cx q[1], q[3];
t q[3]; cx q[0], q[3]; t q[3]; cx q[1], q[3]; t q[3]; cx q[0], q[3]; t q[1]; t q[3]; cx q[0], q[1]; t q[0];
t q[1]; cx q[0], q[1]; h q[3]; x q[0]; x q[1]; x q[3]; h q[0]; h q[1]; h q[2]; creg c[6]; measure q -> c;
"""

SCRIPT = textwrap.dedent("""
    import sys, json, contextlib, io
    sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[1] + "/experiments")
    from circls import pipeline as P
    from ablation import _BASE, CONFIGS          # the paper's full pipeline settings (liveness on)
    kw = dict(_BASE); kw.update(CONFIGS["full"]); kw.pop("liveness", None)
    kw.update(liveness=True, keep_patches=set(), parallel_steps=False)
    qasm = open(sys.argv[2]).read()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            cp = P.compile_qasm(qasm, distance=3, t_as_s=True, **kw)
    except RuntimeError as e:
        print(json.dumps({"error": str(e)[:80]})); sys.exit(0)
    c = cp.circuit
    det, obs = c.compile_detector_sampler(seed=1).sample(64, separate_observables=True)
    print(json.dumps({"observables": c.num_observables, "p0_silent": bool(not det.any()),
                      "p0_deterministic": bool(all((obs[:, j] == obs[0, j]).all() for j in range(obs.shape[1]))),
                      "logicals_ok": bool(cp.observables is not None)}))
""")


def _run(tmp_path, flag):
    root = Path(__file__).resolve().parent.parent
    q = tmp_path / "simon_n6.qasm"; q.write_text(SIMON_N6)
    s = tmp_path / "run.py"; s.write_text(SCRIPT)
    env = dict(os.environ); env.pop("LIGHTSTIM_RESOLVE_DEPENDENT_LOGICALS", None)
    if flag is not None:
        env["LIGHTSTIM_RESOLVE_DEPENDENT_LOGICALS"] = flag
    r = subprocess.run([sys.executable, str(s), str(root), str(q)], capture_output=True, text=True, env=env, timeout=1800)
    line = [l for l in r.stdout.splitlines() if l.startswith("{")]
    assert line, r.stderr[-2000:]
    import json
    return json.loads(line[-1])


def test_half_read_retirement_resolves_by_default(tmp_path):
    out = _run(tmp_path, None)
    assert "error" not in out, out
    assert out["p0_silent"] and out["p0_deterministic"] and out["observables"] >= 1, out


def test_half_read_retirement_opt_out_reproduces_the_documented_failure(tmp_path):
    # LIGHTSTIM_RESOLVE_DEPENDENT_LOGICALS=0 restores the pre-fix behaviour
    out = _run(tmp_path, "0")
    assert "error" in out and "Logical Count Mismatch" in out["error"], out
