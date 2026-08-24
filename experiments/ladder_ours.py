"""A1(2) our-side ladder: bv/dj at n = 32/64/100, d = 3/5.

The TopoLS+tqec baseline cannot emit circuits past n=16 for these
families (tqec v0.2.0 spatial-Hadamard NotImplementedError, recorded in
the topols_circuits manifest as capability_x) — so this side runs alone:
full-pipeline compile, resource metrics, and system-level LER
(decoder included; single-error audit reported as diagnostics).  bv_16 stays the
largest same-QASM LER comparison point; these rows show the compiler
keeps producing runnable, decodable circuits to n=100.

Usage:  python experiments/ladder_ours.py [--d 3 5] [--p 1e-3 5e-4]
"""
import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "experiments"))

from benchsuite import bv, dj
from provenance import provenance

OBS_CHUNK = 30   # not 60: stim also caps a SINGLE error's decomposition
# at 64 terms, and one hub-window error in dj_n flips ~n observables —
# 60 obs terms + detector terms broke dj_64/dj_100 (measured 2026-08-07)


def _obs_chunks(noisy):
    """Rewrite the circuit into observable chunks of <= OBS_CHUNK: stim's
    error decomposition refuses observable ids > 63 (measured on bv_64)
    AND any single error decomposing into > 64 terms (measured on
    dj_64: one merge-window error flips ~n observables), so decoding
    runs per chunk — same detectors, same samples, chunk matchers'
    predictions concatenated."""
    import re
    import stim
    n_obs = noisy.num_observables
    if n_obs <= OBS_CHUNK:
        return None
    text = str(noisy)
    chunks = []
    for lo in range(0, n_obs, OBS_CHUNK):
        hi = min(lo + OBS_CHUNK, n_obs)
        out = []
        for line in text.splitlines():
            m = re.match(r"(\s*)OBSERVABLE_INCLUDE\((\d+)\)(.*)$", line)
            if m:
                k = int(m.group(2))
                if not (lo <= k < hi):
                    continue
                line = f"{m.group(1)}OBSERVABLE_INCLUDE({k - lo}){m.group(3)}"
            out.append(line)
        chunks.append(stim.Circuit("\n".join(out)))
    return chunks


def _single_audit_counts(noisy):
    """Whole-circuit matcher + miscorrection diagnostics (never raises)."""
    import numpy as np
    import pymatching
    dem = noisy.detector_error_model(decompose_errors=True)
    m = pymatching.Matching.from_detector_error_model(dem)
    flat = dem.flattened()
    bad = total = 0
    batch_d, batch_o = [], []

    def _flush():
        nonlocal bad, total
        if not batch_d:
            return
        D = np.zeros((len(batch_d), flat.num_detectors), dtype=bool)
        O = np.zeros((len(batch_d), flat.num_observables), dtype=bool)
        for i, ds in enumerate(batch_d):
            D[i, ds] = True
        for i, os_ in enumerate(batch_o):
            O[i, os_] = True
        pred = m.decode_batch(D).astype(bool)
        bad += int((pred ^ O).any(axis=1).sum())
        total += len(batch_d)
        batch_d.clear()
        batch_o.clear()

    for inst in flat:
        if inst.type != "error":
            continue
        ds, os_ = [], []
        for tg in inst.targets_copy():
            if tg.is_relative_detector_id():
                ds.append(tg.val)
            elif tg.is_logical_observable_id():
                os_.append(tg.val)
        if len(set(ds)) != len(ds):
            ds = [x for x in set(ds) if ds.count(x) % 2]
        if len(set(os_)) != len(os_):
            os_ = [x for x in set(os_) if os_.count(x) % 2]
        batch_d.append(ds)
        batch_o.append(os_)
        if len(batch_d) >= 20_000:
            _flush()
    _flush()
    return m, bad, total


def _chunked_matchers(noisy):
    """Matchers (one per observable chunk, or a single whole-circuit
    matcher) plus single-error audit DIAGNOSTICS (miscorrected, total).
    Never raises: system-level LER ruling 2026-08-09."""
    import numpy as np
    import pymatching
    from compare_tqec import faithful_or_raise
    chunks = _obs_chunks(noisy)
    if chunks is None:
        m, bad, total = _single_audit_counts(noisy)
        return [m], bad, total
    matchers = []
    bad = total = 0
    for sub in chunks:
        dem = sub.detector_error_model(decompose_errors=True)
        m = pymatching.Matching.from_detector_error_model(dem)
        flat = dem.flattened()
        # VECTORISED slab audit.  One fresh np.zeros per mechanism plus a
        # list->array conversion measured ~100x slower than fancy-index
        # filling a preallocated slab (2026-08-08: a 20k slab stalled
        # >10 min vs 0.2 s measured; per-row 60+ h vs ~12 min).  Index
        # lists carry XOR semantics via the duplicate guard (stim's
        # canonical DEM lists each target once per error).
        mech_d, mech_o = [], []

        def _flush():
            nonlocal bad, total
            if not mech_d:
                return
            D = np.zeros((len(mech_d), flat.num_detectors), dtype=bool)
            O = np.zeros((len(mech_d), flat.num_observables), dtype=bool)
            for i, ds in enumerate(mech_d):
                D[i, ds] = True
            for i, os_ in enumerate(mech_o):
                O[i, os_] = True
            pred = m.decode_batch(D).astype(bool)
            bad += int((pred ^ O).any(axis=1).sum())
            total += len(mech_d)
            mech_d.clear()
            mech_o.clear()

        for inst in flat:
            if inst.type != "error":
                continue
            ds, os_ = [], []
            for t in inst.targets_copy():
                if t.is_relative_detector_id():
                    ds.append(t.val)
                elif t.is_logical_observable_id():
                    os_.append(t.val)
            if len(set(ds)) != len(ds):
                ds = [x for x in set(ds) if ds.count(x) % 2]
            if len(set(os_)) != len(os_):
                # decomposed components can each carry the same observable;
                # flattening lists it twice and the flips cancel (measured
                # 2026-08-08: 8816/800318 mechanisms on bv_64 d3) — fold
                # by parity, preserving the old XOR semantics
                os_ = [x for x in set(os_) if os_.count(x) % 2]
            mech_d.append(ds)
            mech_o.append(os_)
            if len(mech_d) >= 20_000:
                _flush()
        _flush()
        matchers.append(m)
    # SYSTEM-LEVEL LER ruling (user, 2026-08-09): the audit no longer
    # gates or falls back — published LER = circuit + decoder as one
    # system; miscorrection counts ride along as diagnostics.
    return matchers, bad, total


def _chunked_ler(noisy, matchers, seed, target_errors, max_shots):
    import numpy as np
    sampler = noisy.compile_detector_sampler(seed=seed)
    shots = joint = 0
    while joint < target_errors and shots < max_shots:
        n = min(20_000, max_shots - shots)
        det, obs = sampler.sample(n, separate_observables=True)
        preds = [m.decode_batch(det).astype(bool) for m in matchers]
        pred = np.concatenate(preds, axis=1)
        joint += int((pred ^ obs).any(axis=1).sum())
        shots += n
    return shots, joint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", nargs="+", type=int, default=[3, 5])
    ap.add_argument("--p", nargs="+", type=float, default=[1e-3, 5e-4])
    ap.add_argument("--target-errors", type=int, default=100)
    ap.add_argument("--max-shots", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=61)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--skip-ler", action="store_true",
                    help="resource + p=0 silence rows only (n=100 policy "
                         "2026-08-07: joint LER saturates by n=32 and the "
                         "baseline is capability_x)")
    ap.add_argument("--names", nargs="+", default=None,
                    help="subset of ladder names (parallel per-row runs "
                         "write separate --out files; the table merges "
                         "every ladder_ours*.jsonl)")
    ap.add_argument("--out",
                    default="experiments/results/topols_compare/"
                            "ladder_ours.jsonl")
    args = ap.parse_args()
    prov = provenance(allow_dirty=args.allow_dirty)

    from circls.pipeline import compile_qasm
    from circls.metrics import experiment_stats
    from circls.metrics.report import to_dict
    from noise_inject import inject_uniform_noise

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in open(out):
            r = json.loads(line)
            if (r.get("status") == "OK"
                    and all("ler" in pt for pt in r.get("ler", []))):
                done.add((r["name"], r["d"]))
    fh = open(out, "a")
    fh.write(json.dumps({**prov, "record": "provenance"}) + "\n")
    fh.flush()

    # compile cache: reruns (chunk-size fixes, fallback recompiles) must
    # not pay the d=5 compile again (dj_64 d5 took 4820 s)
    import stim as _stim
    cachedir = out.parent / "ladder_cache"
    cachedir.mkdir(exist_ok=True)

    def compiled(name, qasm, d, **over):
        tag = "_".join([name, f"d{d}"] + sorted(over))
        cpath = cachedir / f"{tag}.stim"
        spath = cachedir / f"{tag}.stats.json"
        if cpath.exists() and spath.exists():
            meta = json.loads(spath.read_text())
            return (_stim.Circuit(cpath.read_text()), meta["stats"],
                    meta["compile_seconds"])
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            cp = compile_qasm(qasm, distance=d, assignment="optimized",
                              liveness=True, keep_patches=set(), **over)
        secs = round(time.perf_counter() - t0, 1)
        st = to_dict(experiment_stats(cp.experiment, cp.circuit))
        cpath.write_text(str(cp.circuit))
        spath.write_text(json.dumps({"stats": st, "compile_seconds": secs,
                                     "overrides": over}))
        return cp.circuit, st, secs

    idx = 0
    ladder = [("bv_32", bv(32)), ("bv_64", bv(64)), ("bv_100", bv(100)),
              ("dj_32", dj(32)), ("dj_64", dj(64)), ("dj_100", dj(100))]
    if args.names:
        ladder = [(n, q) for n, q in ladder if n in args.names]
    for name, qasm in ladder:
        for d in args.d:
            if (name, d) in done:
                continue
            t0 = time.perf_counter()
            row = {"name": name, "d": d}
            try:
                circuit, row["stats"], row["compile_seconds"] = compiled(
                    name, qasm, d)
                row["ler"] = []
                if args.skip_ler:
                    det, _ = circuit.compile_detector_sampler(
                        seed=0).sample(128, separate_observables=True)
                    row["silent"] = bool(not det.any())
                    row["silence_shots"] = 128
                    row["ler_policy"] = ("skipped by ruling 2026-08-07: "
                                         "joint LER saturated, baseline "
                                         "capability_x")
                    row["status"] = "OK"
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    print(f"{name} d={d}: resource row, silent="
                          f"{row['silent']}", flush=True)
                    continue
                for p in args.p:
                    pt = {"p": p}
                    t1 = time.perf_counter()
                    try:
                        noisy = inject_uniform_noise(circuit, p)
                        matchers, mis, n_mech = _chunked_matchers(noisy)
                        pt["audit"] = {"miscorrected": mis,
                                       "mechanisms": n_mech}
                        seed = args.seed + idx
                        idx += 1
                        shots, joint = _chunked_ler(
                            noisy, matchers, seed, args.target_errors,
                            args.max_shots)
                        pt.update({"shots": shots, "joint_errors": joint,
                                   "ler": joint / shots, "seed": seed,
                                   "obs_chunks": len(matchers),
                                   "seconds": round(
                                       time.perf_counter() - t1, 1)})
                    except Exception as e:
                        pt.update({"status": type(e).__name__,
                                   "error": str(e)[:200]})
                    row["ler"].append(pt)
                    print(f"{name} d={d} p={p:g}: "
                          f"{pt.get('ler', pt.get('status'))}", flush=True)
                row["status"] = "OK"
            except Exception as e:
                row.update({"status": type(e).__name__,
                            "error": str(e)[:300],
                            "compile_seconds": round(
                                time.perf_counter() - t0, 1)})
                print(f"{name} d={d}: {type(e).__name__} {str(e)[:100]}",
                      flush=True)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
    fh.close()
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
