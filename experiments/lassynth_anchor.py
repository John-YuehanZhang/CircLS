"""LaSsynth optimality anchors (phase 1+2): proven-optimal depth vs ours.

For a few small suite programs, ask LaSsynth (Tan/Niu/Gidney, ISCA'24 —
the SAT-exact lattice-surgery synthesizer shipped in quantumlib/Stim's
glue/lattice_surgery) for the OPTIMAL spacetime depth of preparing the
program's pre-measurement stabilizer state, with an UNSAT proof at
depth k*-1; then compile the same program with the CircLS pipeline and
record our executed rounds.  Anchor semantics:

  - the spec is STATE PREPARATION (ports = outputs only, stabilizers of
    U|0..0>): a solver free to reach the state any way it likes lower-
    bounds every compiler on the same program;
  - depth is box-conditional, so each program is proven at TWO
    footprints (the tight one and a slacker one) — a k* that survives
    the slack footprint is not an artifact of space starvation;
  - unit note for the table: one LaSsynth layer = one merge window =
    d QEC rounds; our measured rounds include the initialization
    baseline and the terminal readout (raw values, definitions in the
    caption).

The SAT solving runs in the probe's conda env (python 3.11 + z3 +
kissat, see REPRODUCE.md) via subprocess; this script itself runs in
the repo env for the CircLS side.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "experiments"))

# program -> (spec kind, footprints to prove at)
ANCHORS = {
    "cat_state_n4": ("ghz", 4, [(3, 3), (4, 4)]),
    "ghz_8": ("ghz", 8, [(8, 2), (4, 4)]),
    "grover_n2": ("state_of_circuit", 2, [(2, 2), (3, 3)]),
    "steane_encode": ("state_of_circuit", 7, [(3, 3), (4, 4)]),
}

_SOLVER_DRIVER = r"""
import json, sys, time
sys.path.insert(0, sys.argv[4])          # probe dir: spec_builder + kissat
from lassynth import LatticeSurgerySynthesizer
spec = json.load(open(sys.argv[1]))
kissat = sys.argv[4] + "/kissat/build/"
synth = LatticeSurgerySynthesizer(solver="kissat", kissat_dir=kissat)
t0 = time.time()
res = synth.solve(specification=spec)
wall = time.time() - t0
out = {"status": "SAT" if res is not None else "UNSAT",
       "wall": round(wall, 2)}
if res is not None and sys.argv[3] == "save":
    res = res.after_default_optimizations()
    res.save_lasre(sys.argv[2])
    out["verified"] = bool(res.verify_stabilizers_stimzx(
        specification=spec))
print("DRIVER_RESULT " + json.dumps(out))
"""


def build_spec(kind, n, mi, mj, mk, qasm=None):
    import stim
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from lassynth_spec import ghz_stabilizers, state_spec
    if kind == "ghz":
        return state_spec(ghz_stabilizers(n), mi, mj, mk)
    # state_of_circuit: stabilizers of U|0..0> from the program's
    # unitary prefix (measure/creg lines stripped)
    lines = [l for l in qasm.splitlines()
             if not l.strip().startswith(("measure", "creg", "OPENQASM",
                                          "include", "qreg", "barrier",
                                          "//"))
             and l.strip()]
    import re
    c = stim.Circuit()
    for l in lines:
        l = l.strip().rstrip(";")
        m = re.match(r"(\w+)\s+(.*)", l)
        g, args = m.group(1), m.group(2)
        qs = [int(x) for x in re.findall(r"q\[(\d+)\]", args)]
        gate = {"h": "H", "x": "X", "z": "Z", "s": "S", "sdg": "S_DAG",
                "cx": "CX", "cz": "CZ"}[g]
        c.append(gate, qs)
    t = c.to_tableau()
    stabs = []
    for q in range(n):
        img = t.z_output(q)
        stabs.append("".join(".XYZ"[img[p]] for p in range(n)))
    return state_spec(stabs, mi, mj, mk)


def solve(spec, probe_dir, env_python, save_path=None, timeout=1800):
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json",
                                     delete=False) as fh:
        json.dump(spec, fh)
        spec_path = fh.name
    proc = subprocess.run(
        [env_python, "-c", _SOLVER_DRIVER, spec_path,
         save_path or "/dev/null", "save" if save_path else "nosave",
         probe_dir],
        capture_output=True, text=True, timeout=timeout)
    for line in proc.stdout.splitlines():
        if line.startswith("DRIVER_RESULT "):
            return json.loads(line[len("DRIVER_RESULT "):])
    return {"status": "error",
            "error": (proc.stderr or proc.stdout)[-300:]}


def our_side(case, d=3):
    """Both frames: the full pipeline (whose re-selection may dissolve
    a small program into terminal readouts entirely — steps=0 is a
    disclosed result, not a failure) and measure_reduction=False (the
    backend actually routes joint measurements — the layout-depth
    number the anchor compares)."""
    import contextlib
    import io
    from ablation import compile_kwargs
    from circls.pipeline import compile_qasm
    from circls.metrics.circuit_stats import circuit_stats
    out = {}
    for tag, extra in (("full", {}), ("nored",
                                     {"measure_reduction": False})):
        kw = {**compile_kwargs("full", case), **extra}
        t0 = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            cp = compile_qasm(case.qasm, distance=d, **kw)
        cs = circuit_stats(cp.circuit)
        out.update({f"our_{tag}_rounds": cs.measurement_layers,
                    f"our_{tag}_steps": len(cp.experiment.ppm_sequence),
                    f"our_{tag}_qubits": cs.qubits_active,
                    f"our_{tag}_compile_s": round(time.time() - t0, 1)})
    return out


def main():
    import argparse
    from benchsuite import suite
    from provenance import provenance
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", required=True,
                    help="the lassynth probe dir (env/, kissat/, scripts)")
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--out", default=str(
        _ROOT / "experiments/results/best5/lassynth_anchor.jsonl"))
    args = ap.parse_args()
    prov = provenance(args.allow_dirty)
    probe = str(Path(args.probe_dir).resolve())
    env_python = probe + "/env/bin/python"
    cases = {c.name: c for c in suite()}
    out = Path(args.out)
    artifacts = out.parent / "lassynth"
    artifacts.mkdir(parents=True, exist_ok=True)
    with out.open("a") as fh:
        fh.write(json.dumps({**prov, "record": "provenance",
                             "d": args.d}) + "\n")
        for name, (kind, n, footprints) in ANCHORS.items():
            case = cases[name]
            ours = our_side(case, d=args.d)
            for (mi, mj) in footprints:
                k, hist = 1, []
                while True:
                    spec = build_spec(kind, n, mi, mj, k, qasm=case.qasm)
                    save = (str(artifacts / f"{name}_{mi}x{mj}x{k}"
                                            ".lasre.json")
                            if True else None)
                    r = solve(spec, probe, env_python, save_path=save)
                    hist.append({"k": k, **r})
                    print(f"{name} {mi}x{mj} k={k}: {r['status']} "
                          f"({r.get('wall', '?')}s)", flush=True)
                    if r["status"] == "SAT" or r["status"] == "error" \
                            or k > 12:
                        break
                    k += 1
                row = {"record": "lassynth_anchor", "program": name,
                       "spec": kind, "n": n, "footprint": [mi, mj],
                       "k_star": k if hist[-1]["status"] == "SAT"
                       else None,
                       "history": hist, "d": args.d,
                       "verified": hist[-1].get("verified"),
                       **ours}
                fh.write(json.dumps(row) + "\n")
                fh.flush()


if __name__ == "__main__":
    main()
