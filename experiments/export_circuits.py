"""Export the compiled CircLS stim circuits as a release asset.

One .stim file per (program, distance) under --outdir, plus a
manifest.jsonl (provenance row + one row per circuit with its sha256).
Covers the 45-program suite at d=3 under the paper's full
configuration, plus the d=5 rows of the head-to-head table.  The
circuits are deterministically regenerable with compile_qasm; this
export exists so artifact reviewers can inspect them without
installing anything.

Usage:
    python experiments/export_circuits.py --outdir <dir> [--workers N]
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "experiments"))

D5_CASES = ["ghz_16_mixed", "bv_16", "steane_encode"]


def _export_one(task):
    name, qasm, d, outdir = task
    import contextlib, io
    from ablation import compile_kwargs
    from benchsuite import suite
    from circls.pipeline import compile_qasm
    path = Path(outdir) / f"{name}_d{d}.stim"
    if path.exists():                      # deterministic: reuse
        text = path.read_text()
        return {"name": name, "d": d, "file": path.name,
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "bytes": len(text), "reused": True}
    case = next(c for c in suite() if c.name == name)
    kw = compile_kwargs("full", case)
    kw["distance"] = d
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            cp = compile_qasm(qasm, **kw)
    except Exception as e:  # record, never abort the export
        return {"name": name, "d": d, "error": str(e)[:200]}
    text = str(cp.circuit)
    path.write_text(text)
    return {"name": name, "d": d, "file": path.name,
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "bytes": len(text), "seconds": round(time.time() - t0, 1)}


def main():
    from benchsuite import suite
    from provenance import provenance
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    # bv_n140 / bv_n280 are replay-grid-only instances outside the
    # 45-program suite (bv_n280 has no full-pipeline compile)
    cases = sorted((c for c in suite()
                    if c.name not in ("bv_n140", "bv_n280")),
                   key=lambda c: c.n_qubits)
    tasks = [(c.name, c.qasm, 3, str(out)) for c in cases]
    tasks += [(c.name, c.qasm, 5, str(out)) for c in cases
              if c.name in D5_CASES]
    prov = provenance(args.allow_dirty)
    rows = []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for r in pool.imap_unordered(_export_one, tasks):
            rows.append(r)
            tag = "ERR " if "error" in r else ""
            print(f"{tag}{r['name']} d{r['d']} "
                  f"({r.get('seconds', '-')}s)", flush=True)
    rows.sort(key=lambda r: (r["d"], r["name"]))
    with (out / "manifest.jsonl").open("w") as fh:
        fh.write(json.dumps({**prov, "record": "provenance",
                             "config": "full", "n_circuits": len(rows)})
                 + "\n")
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    n_err = sum(1 for r in rows if "error" in r)
    print(f"DONE: {len(rows) - n_err} circuits, {n_err} errors", flush=True)


if __name__ == "__main__":
    main()
