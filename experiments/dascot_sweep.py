"""DASCOT (wisq, scmr/dascot mode) layout sweep over the benchsuite.

Phase A of the DASCOT baseline: run THEIR mapper+router on every suite
program under a FIXED SEED SET and record their raw outputs — the map,
the scheduled steps (each gate with its explicit corridor path), step
count and wall time — one jsonl row per (program, seed), self-contained
so the later replay phase reads rows, not files.

Reproducibility notes (probe 2026-08-15):
  - wisq exposes no seed; its SA routing draws from the GLOBAL python
    and numpy RNGs in-process, so seeding both before map_and_route
    makes runs bit-identical (verified 3x md5-equal).  Solution quality
    varies BY seed, so the paper reports the per-program MEDIAN step
    count over the seed set; every seed's row is recorded.
  - their QASM front end regex-parses only `cx q[i], q[j];` (+ t/tdg):
    1q gates and measure lines are silently dropped (their model treats
    them as free), a program with no cx at all dies on ZeroDivisionError,
    and a two-qubit gate outside the regex (cz, swap) dies on KeyError —
    both recorded as front-end rejections, not our failures.
  - on timeout wisq writes a DIFFERENT schema ("steps": "timeout") and
    then crashes with TypeError; treated as status=timeout.

Each (program, seed) runs in a fresh subprocess (the global-RNG seeding
must not leak across runs, and their SIGALRM timeout handling is
per-process state).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "experiments"))

SEEDS = [7, 11, 23, 42, 101, 202, 303]

_CHILD = r"""
import json, random, sys
import numpy as np
random.seed(int(sys.argv[2])); np.random.seed(int(sys.argv[2]))
from wisq import map_and_route
map_and_route(sys.argv[1], "square_sparse_layout", sys.argv[3],
              int(sys.argv[4]), mode="dascot")
"""


def run_one(qasm_path: Path, seed: int, out_json: Path, timeout: int):
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, str(qasm_path), str(seed),
         str(out_json), str(timeout)],
        capture_output=True, text=True, timeout=timeout + 120)
    wall = round(time.time() - t0, 1)
    err = (proc.stderr or "").strip().splitlines()
    tail = err[-1] if err else ""
    if out_json.exists():
        try:
            d = json.loads(out_json.read_text())
        except json.JSONDecodeError:
            return {"status": "bad_json", "wall": wall, "error": tail}
        if d.get("steps") == "timeout" or not isinstance(
                d.get("steps"), list):
            return {"status": "timeout", "wall": wall}
        return {"status": "ok", "wall": wall, "n_steps": len(d["steps"]),
                "map": d["map"], "steps": d["steps"],
                "arch": d.get("arch"), "gates": d.get("gates")}
    if "ZeroDivisionError" in tail:
        return {"status": "frontend_reject", "wall": wall,
                "error": "no cx gates (ZeroDivisionError)"}
    if "KeyError" in tail:
        return {"status": "frontend_reject", "wall": wall,
                "error": f"unparsed 2q gate ({tail[:80]})"}
    return {"status": "error", "wall": wall, "error": tail[:160]}


def main():
    import argparse
    from benchsuite import suite
    from provenance import provenance
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--out", default=str(
        _ROOT / "experiments/results/best5/dascot_layout.jsonl"))
    args = ap.parse_args()
    prov = provenance(args.allow_dirty)
    wd = Path(args.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    cases = [c for c in suite()
             if args.cases is None or c.name in args.cases]
    out = Path(args.out)
    with out.open("a") as fh:
        fh.write(json.dumps({**prov, "record": "provenance",
                             "seeds": args.seeds,
                             "timeout": args.timeout}) + "\n")
        for c in sorted(cases, key=lambda c: c.n_qubits):
            qasm = wd / f"{c.name}.qasm"
            qasm.write_text(c.qasm)
            for seed in args.seeds:
                oj = wd / f"{c.name}.s{seed}.json"
                try:
                    r = run_one(qasm, seed, oj, args.timeout)
                except subprocess.TimeoutExpired:
                    r = {"status": "hard_timeout", "wall": args.timeout}
                row = {"record": "dascot_layout", "program": c.name,
                       "n": c.n_qubits, "seed": seed,
                       "arch_name": "square_sparse_layout", **r}
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                print(f"{c.name} s{seed}: {r['status']} "
                      f"{r.get('n_steps', '')} ({r['wall']}s)",
                      flush=True)


if __name__ == "__main__":
    main()
