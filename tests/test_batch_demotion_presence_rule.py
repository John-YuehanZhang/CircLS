"""A demoted shared-window batch must not leave the router a stale presence rule.

``_try_run_batch`` allocates + initialises every batch member's patches in its
preamble and, when the batch then demotes to the serial path, leaves those
allocations standing ("allocations stick").  The serial presence rule
``_specs_for_step._absent`` used to decide first-use-init absence from the
lifetime ledger alone (``fu > i``), so a patch born early by a demoted batch was
offered to the router as borrowable ground; the corridor ran through its cell
and ``QECSystem.add_patch`` raised "Coordinate collision with ACTIVE qubit"
(measured 2026-09-07: QASMBench/TopoLS multiply_n13 tt2, d=3, liveness off,
row_major, |+> gadget ancillas: batch [4..9] allocated q4 (first use 9) at step
4, demoted, and ppm_4's serial corridor took q4's cell).  The rule now also
consults the physical allocation (``system.patches``).

Two tests.  The fast one caps the compile after step 8 (the collision was at
step 4) and asserts the cap is reached.  The slow one runs the batch-two
recipe to the end: the failing rung (row_major with scheduling) must now end
in a routing verdict (BentLayoutError at step 37), the ladder's next rung
(row_major, no scheduling) must compile, p = 0 must be silent and
deterministic, and the observable set must span exactly the skeleton rank of
the proxy program (every t deleted: 13 deterministic output bits).
"""
import contextlib
import io
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

MULTIPLY_N13_TT2 = """\
OPENQASM 2.0;
include "qelib1.inc";
qreg q[13];
x q[0];
x q[1];
x q[2];
x q[4];
h q[5];
cx q[0], q[5];
t q[5];
cx q[2], q[5];
t q[5];
cx q[0], q[5];
t q[5];
cx q[2], q[5];
t q[0];
t q[5];
cx q[2], q[0];
t q[2];
t q[0];
cx q[2], q[0];
h q[5];
h q[6];
cx q[1], q[6];
t q[6];
cx q[2], q[6];
t q[6];
cx q[1], q[6];
t q[6];
cx q[2], q[6];
t q[1];
t q[6];
cx q[2], q[1];
t q[2];
t q[1];
cx q[2], q[1];
h q[6];
h q[7];
cx q[0], q[7];
t q[7];
cx q[3], q[7];
t q[7];
cx q[0], q[7];
t q[7];
cx q[3], q[7];
t q[0];
t q[7];
cx q[3], q[0];
t q[3];
t q[0];
cx q[3], q[0];
h q[7];
h q[8];
cx q[1], q[8];
t q[8];
cx q[3], q[8];
t q[8];
cx q[1], q[8];
t q[8];
cx q[3], q[8];
t q[1];
t q[8];
cx q[3], q[1];
t q[3];
t q[1];
cx q[3], q[1];
h q[8];
h q[9];
cx q[0], q[9];
t q[9];
cx q[4], q[9];
t q[9];
cx q[0], q[9];
t q[9];
cx q[4], q[9];
t q[0];
t q[9];
cx q[4], q[0];
t q[4];
t q[0];
cx q[4], q[0];
h q[9];
h q[10];
cx q[1], q[10];
t q[10];
cx q[4], q[10];
t q[10];
cx q[1], q[10];
t q[10];
cx q[4], q[10];
t q[1];
t q[10];
cx q[4], q[1];
t q[4];
t q[1];
cx q[4], q[1];
h q[10];
cx q[6], q[11];
cx q[7], q[11];
cx q[8], q[12];
cx q[9], q[12];
creg c[13];
measure q -> c;
"""

SCRIPT = textwrap.dedent("""
    import sys, json, contextlib, io
    sys.path.insert(0, sys.argv[1])
    import circls.pipeline as P
    from circls.core.sequential_ppm_ls import SequentialPPMExperiment as SE
    CAP = int(sys.argv[3])
    class Reached(Exception):
        pass
    _orig = SE._apply_ppm_step
    def _capped(self, i, step, birth_names=None):
        if i >= CAP:
            raise Reached(i)
        return _orig(self, i, step, birth_names)
    SE._apply_ppm_step = _capped
    # the |+> (X-state) gadget-ancilla proxy: y patches become ordinary data
    # patches, so gadget steps are batch-eligible (the |Y> proxy never batches
    # them and cannot reach the demotion path)
    orig = P.to_experiment_inputs
    def xproxy(*a, **k):
        specs, steps, init, final, prog = orig(*a, **k)
        return specs, steps, {nm: ("X" if v == "Y" else v) for nm, v in init.items()}, final, prog
    P.to_experiment_inputs = xproxy
    qasm = open(sys.argv[2]).read()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            P.compile_qasm(qasm, distance=3, t_as_s=True, assignment="row_major",
                           measure_reduction=True, first_use_init=True,
                           step_scheduling=True, parallel_steps=True)
        print(json.dumps({"result": "completed"}))
    except Reached as e:
        print(json.dumps({"result": "reached", "step": int(str(e))}))
    except Exception as e:
        print(json.dumps({"result": "error", "error": f"{type(e).__name__}: {e}"[:200]}))
""")


def test_demoted_batch_births_are_router_obstacles(tmp_path):
    root = Path(__file__).resolve().parent.parent
    q = tmp_path / "multiply_n13_tt2.qasm"
    q.write_text(MULTIPLY_N13_TT2)
    s = tmp_path / "run.py"
    s.write_text(SCRIPT)
    r = subprocess.run([sys.executable, str(s), str(root), str(q), "8"],
                       capture_output=True, text=True, timeout=1200)
    line = [l for l in r.stdout.splitlines() if l.startswith("{")]
    assert line, r.stderr[-2000:]
    out = json.loads(line[-1])
    assert out["result"] == "reached", out
    assert out["step"] == 8


FULL_SCRIPT = textwrap.dedent("""
    import sys, json, contextlib, io, re
    sys.path.insert(0, sys.argv[1])
    import stim
    import circls.pipeline as P
    orig = P.to_experiment_inputs
    def xproxy(*a, **k):
        specs, steps, init, final, prog = orig(*a, **k)
        return specs, steps, {nm: ("X" if v == "Y" else v) for nm, v in init.items()}, final, prog
    P.to_experiment_inputs = xproxy
    qasm = open(sys.argv[2]).read()

    def gf2_rank(rows):
        basis = {}; rank = 0
        for r in rows:
            v = int("".join(map(str, r)), 2) if r else 0
            while v:
                b = v.bit_length() - 1
                if b in basis: v ^= basis[b]
                else: basis[b] = v; rank += 1; break
        return rank

    def skeleton_rank(text, n):               # deterministic Z-parity output bits of the t/s-deleted program
        G1 = {"h": "H", "x": "X", "z": "Z", "y": "Y"}; G2 = {"cx": "CX", "cz": "CZ", "swap": "SWAP"}
        c = stim.Circuit(); c.append("I", range(n))
        for line in text.splitlines():
            for st in line.split("//")[0].split(";"):
                st = st.strip()
                if not st: continue
                head = st.split()[0].lower()
                if head in ("openqasm", "include", "qreg", "creg", "measure", "barrier", "t", "tdg", "s", "sdg"): continue
                args = [int(a) for a in re.findall(r"\\[(\\d+)\\]", st)]
                if head in G1: c.append(G1[head], args)
                elif head in G2: c.append(G2[head], args)
                else: raise ValueError(head)
        tab = c.to_tableau(); rows = [tab.z_output(i) for i in range(n)]
        return n - gf2_rank([[1 if rows[i][j] in (1, 2) else 0 for j in range(n)] for i in range(n)])

    n = int(re.search(r"qreg\\s+\\w+\\[(\\d+)\\]", qasm).group(1))
    out = {"rank": skeleton_rank(qasm, n)}
    KW = dict(measure_reduction=True, first_use_init=True, parallel_steps=True, assignment="row_major")
    # the failing rung: row_major with scheduling (liveness off).  Pre-fix: ValueError, coordinate collision at ppm_4.
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            P.compile_qasm(qasm, distance=3, t_as_s=True, step_scheduling=True, **KW)
        out["rung2"] = "compiled"
    except Exception as e:
        out["rung2"] = f"{type(e).__name__}: {e}"[:160]
    # the ladder's next rung: row_major, no scheduling
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            cp = P.compile_qasm(qasm, distance=3, t_as_s=True, step_scheduling=False, **KW)
    except Exception as e:
        out["rung3"] = f"{type(e).__name__}: {e}"[:160]; print(json.dumps(out)); sys.exit(0)
    c = cp.circuit
    det, obs = c.compile_detector_sampler(seed=1).sample(32, separate_observables=True)
    try:
        c.detector_error_model(); dem_ok = True
    except Exception:
        dem_ok = False
    tot = 0; recsets = {}
    for inst in c.flattened():
        if inst.name.startswith("M"):
            tot += sum(1 for t in inst.targets_copy() if not t.is_combiner) if inst.name == "MPP" else len(inst.targets_copy())
        if inst.name == "OBSERVABLE_INCLUDE":
            j = int(inst.gate_args_copy()[0]); recsets.setdefault(j, set())
            for t in inst.targets_copy(): recsets[j] ^= {tot + t.value}
    vecs = [recsets[j] for j in sorted(recsets)]; allrec = sorted(set().union(*vecs)) if vecs else []
    out.update({"rung3": "compiled", "ppms": len(cp.experiment.ppm_sequence), "patches": len(cp.placement),
                "observables": c.num_observables,
                "obs_distinct": gf2_rank([[1 if r in v else 0 for r in allrec] for v in vecs]) if vecs else 0,
                "p0_silent": bool(not det.any()),
                "p0_deterministic": bool(all((obs[:, j] == obs[0, j]).all() for j in range(obs.shape[1]))),
                "dem_ok": dem_ok})
    print(json.dumps(out))
""")


@pytest.mark.slow
def test_no_live_multiply_compiles_through_the_ladder(tmp_path):
    """The batch-two cell itself (multiply_n13 d=3, liveness off): the failing rung
    ends in a routing verdict instead of the collision, the next rung compiles,
    p = 0 is deterministic and the observables span the proxy skeleton's rank."""
    root = Path(__file__).resolve().parent.parent
    q = tmp_path / "multiply_n13_tt2.qasm"
    q.write_text(MULTIPLY_N13_TT2)
    s = tmp_path / "run_full.py"
    s.write_text(FULL_SCRIPT)
    r = subprocess.run([sys.executable, str(s), str(root), str(q)],
                       capture_output=True, text=True, timeout=3600)
    line = [l for l in r.stdout.splitlines() if l.startswith("{")]
    assert line, r.stderr[-2000:]
    out = json.loads(line[-1])
    assert "Coordinate collision" not in out["rung2"], out          # the F2 defect
    assert out["rung2"] == "compiled" or "BentLayoutError" in out["rung2"], out   # a routing verdict is the only legal failure
    assert out["rung3"] == "compiled", out
    assert out["ppms"] == 42, out
    assert out["p0_silent"] and out["p0_deterministic"] and out["dem_ok"], out
    assert out["rank"] == 13, out
    assert out["obs_distinct"] == out["rank"], out
