"""Stage 1: compile every resample target on the FINAL code, verify
(force the distance search where affordable), archive circuit + stats."""
import json, sys, time, hashlib, subprocess
import multiprocessing as mp
from pathlib import Path
SP = Path("<workdir>")
REPO = SP / "circls_dev"
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments"))
OUT = SP / "resample" / "compile_stage.jsonl"
CIRC = SP / "resample" / "circuits"

TCLASS = ["qec_en_n5", "teleportation_n3", "toffoli_n3", "bell_n4", "fredkin_n3",
          "adder_n4", "simon_n6", "multiply_n13", "sat_n7"]

def tasks():
    t = []
    for n in TCLASS:
        t.append((n, 3, "tclass", "full"))
        if n != "sat_n7":                      # sat d5 arrives from its own 10 h compile
            t.append((n, 5, "tclass", "full"))
    for d in (3, 5, 7, 9, 11):                 # panel (b)
        for cfg in ("full", "static"):
            t.append(("qec_en_n5", d, "panel", cfg))
    for n in ("twistedghz_4", "twistedghz_8"):  # the two changed Clifford rows
        t.append((n, 3, "clifford", "full"))
    return t

def job(task):
    name, d, kind, cfg = task
    tag = f"{name}_{kind}_{cfg}_d{d}"
    out = CIRC / f"{tag}.stim"
    if out.exists():
        return {"tag": tag, "status": "cached"}
    import contextlib, io
    import stim
    from benchsuite import suite, tclass_suite
    from ablation import compile_kwargs
    from circls.pipeline import compile_qasm
    from circls.tools.evaluate import verify
    from circls.metrics import experiment_stats
    from circls.metrics.report import to_dict
    if kind == "clifford":
        case = {c.name: c for c in suite()}[name]
        kw = compile_kwargs("full", case)
    else:
        case = {c.name: c for c in tclass_suite()}[name]
        if cfg == "full":
            kw = compile_kwargs("full", case)
        else:
            kw = compile_kwargs("no_live", case); kw["first_use_init"] = False
        kw["t_as_s"] = True
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            cp = compile_qasm(case.qasm, distance=d, **kw)
    except Exception as e:
        return {"tag": tag, "status": "compile_error", "error": f"{type(e).__name__}: {str(e)[:160]}"}
    t_c = round(time.time() - t0, 1)
    with contextlib.redirect_stdout(io.StringIO()):
        r = verify(cp, force_distance=(d <= 5))
    flat = cp.circuit.flattened()
    out.write_text(str(flat))
    st = to_dict(experiment_stats(cp.experiment, cp.circuit))
    row = {"tag": tag, "name": name, "d": d, "kind": kind, "config": cfg,
           "status": "ok", "compile_s": t_c, "verify_ok": bool(r.ok),
           "silent": bool(r.silent), "logical": r.logical,
           "distance": r.distance, "skipped": list(r.skipped),
           "dets": flat.num_detectors, "obs": flat.num_observables,
           "qubits": flat.num_qubits, "ticks": flat.num_ticks,
           "sha256": hashlib.sha256(out.read_bytes()).hexdigest()[:16],
           "V1_volume_blocks": st.get("V1_volume_blocks"),
           "V2_qubit_rounds": st.get("V2_qubit_rounds"),
           "T1_rounds": st.get("T1_rounds")}
    return row

if __name__ == "__main__":
    ts = tasks()
    fh = open(OUT, "a")
    with mp.get_context("spawn").Pool(24) as pool:
        for r in pool.imap_unordered(job, ts):
            fh.write(json.dumps(r) + "\n"); fh.flush()
            print(r.get("tag"), r.get("status"), r.get("compile_s", ""),
                  "dist", r.get("distance"), flush=True)
    fh.close(); print("COMPILE STAGE DONE", flush=True)
