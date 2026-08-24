"""B-dev: measured program LER vs the literature's composition formula.

The field budgets a program's logical error rate WITHOUT simulating it:

    LER_pred = V_blocks * eps_block(d, p)
    eps_block = a * (p / p_th)^((d+1)/2)

(V_blocks = spacetime volume in d^3 blocks — our V1 metric; the rule is
the Litinski block budget / Azure resource-estimator model, Beverland
et al. 2022.)  This script produces the first circuit-level deviation
measurement: measured/predicted per (case, d, p), two prediction rows —

    published    a, p_th taken from the literature (constants must be
                 re-verified against the papers at writing time)
    calibrated   eps_block measured HERE: single-patch memory circuits
                 through the same pipeline, noise model and decoder; the
                 per-block rate is the round-slope of LER(R) times d.
                 This row isolates the COMPOSITION assumption itself.

Config (design decision 2026-08-06): every compiler trick OFF — liveness,
first-use init, measurement reduction, step scheduling, parallel steps,
optimized placement — so the circuit is the plain protocol the formula
speaks about, with every patch alive start to end.  Correction-fitting
is deliberately NOT done here (postponed by the same ruling).

Usage:
    python experiments/formula_deviation.py [--quick] [--d 3 5]
        [--p 1e-3 5e-4] [--target-errors 100] [--outdir ...]
"""
import argparse
import contextlib
import io
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "experiments"))

import numpy as np

from benchsuite import suite
from provenance import provenance

# Beverland et al. 2022 (arXiv:2211.07629), gate-based ns-regime qubit
# model: P_L = a (p/p_th)^((d+1)/2) with a = 0.03, p_th = 0.01.
# TODO(writing): re-verify both constants verbatim against the paper.
PUB_A = 0.03
PUB_PTH = 0.01

NOTRICK = dict(assignment="row_major", measure_reduction=False,
               step_scheduling=False, parallel_steps=False,
               first_use_init=False, liveness=False)

QUICK_CASES = ["ghz_8", "bv_8", "dj_8", "teleport_4", "twistedghz_4",
               "ghz_16_mixed"]


def _noise(p):
    from lightstim.noise.config import NoiseConfig
    return NoiseConfig(p_1q=p, p_2q=p, p_meas=p, p_reset=p, p_idle=p)


def _faithful_audit(circuit):
    """Single-error audit — a per-point DIAGNOSTIC, not a gate.

    System-level LER ruling (design decision 2026-08-09, extended to B-dev
    2026-08-10): published LER treats circuit + decoder as one system.
    A decoder that miscorrects some rare single-error mechanisms (the
    closure-parity/MWPM interaction) contributes that to the system's
    real error rate; the point is measured and the audit recorded
    alongside — the old gate silently discarded ~40% of the roster.
    Returns (matching, mis, n_mechanisms)."""
    import pymatching
    if circuit.num_observables == 0:
        # same guard as _sample_ler/_mwpf_sample_ler: with no observable the
        # XOR below runs over width-0 arrays and reports mis=0, a meaningless
        # clean-audit verdict instead of an error
        raise ValueError("_faithful_audit: circuit has no observables "
                         "(dagger row) -- audit undefined")
    dem = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)
    flat = dem.flattened()
    dets, obss = [], []
    for inst in flat:
        if inst.type != "error":
            continue
        dv = np.zeros(flat.num_detectors, dtype=bool)
        ov = np.zeros(flat.num_observables, dtype=bool)
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                dv[t.val] ^= True
            elif t.is_logical_observable_id():
                ov[t.val] ^= True
        dets.append(dv)
        obss.append(ov)
    mis = int((matching.decode_batch(np.array(dets)).astype(bool)
               ^ np.array(obss)).any(axis=1).sum())
    return matching, mis, len(dets)


MWPF_TIMEOUT_S = 3.0
MWPF_WORKERS = int(os.environ.get("MWPF_WORKERS", "32"))


def _mwpf_decoder(circuit, gate=True):
    """Hypergraph decoder for circuits whose DEM has non-graphlike
    mechanisms (design decision 2026-08-10: twistedghz d=5 — the closure
    parities make one two-qubit error flip 7 detectors, MWPM cannot be
    built; compute the point with mwpf and annotate the decoder).

    Same audit standard as ``_faithful_audit``: count single-error
    mechanisms that do not decode to their own observable flip
    (``gate=False`` skips it — pool workers reuse the parent's audit).
    Returns (decode_fn, n_mechanisms) where decode_fn maps a defect
    index list to the predicted observable bitmask (python int)."""
    import math
    import mwpf
    dem = circuit.detector_error_model(flatten_loops=True)
    # fold mechanisms with the SAME (parity-folded) detector set:
    # combined flip probability pa(1-pb)+pb(1-pa), observable mask of
    # the most probable representative
    folded = {}
    raw = []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p_e = inst.args_copy()[0]
        dets, obs = set(), 0
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                dets ^= {t.val}
            elif t.is_logical_observable_id():
                obs ^= (1 << t.val)
        key = frozenset(dets)
        raw.append((sorted(dets), obs))
        if key in folded:
            p0, obs0 = folded[key]
            folded[key] = (p0 * (1 - p_e) + p_e * (1 - p0),
                           obs0 if p0 >= p_e else obs)
        else:
            folded[key] = (p_e, obs)
    keys = sorted(folded, key=sorted)
    obs_masks = []
    hyperedges = []
    W = 1000.0  # integer weight scale (mwpf edge weights)
    for key in keys:
        p_e, obs = folded[key]
        w = max(1, int(round(W * math.log((1 - p_e) / p_e))))
        hyperedges.append(mwpf.HyperEdge(sorted(key), w))
        obs_masks.append(obs)
    init = mwpf.SolverInitializer(dem.num_detectors, hyperedges)
    # per-shot solve timeout: rare pathological syndromes make the exact
    # solver grind for minutes (measured 2026-08-10: one 98-defect shot
    # >5 min; with 3 s timeout the same 200-shot batch averages 157
    # ms/shot, 0 decode errors).  On timeout mwpf returns its best
    # solution so far — worst case slightly SUBOPTIMAL decoding, i.e. a
    # conservative (upward) LER bias on those rare shots.
    solver = mwpf.SolverSerialJointSingleHair(init,
                                              {"timeout": MWPF_TIMEOUT_S})

    def decode(defects):
        solver.solve(mwpf.SyndromePattern(defect_vertices=defects))
        sub = solver.subgraph()
        solver.clear()
        m = 0
        for ei in sub:
            m ^= obs_masks[ei]
        return m

    mis = None
    if gate:
        mis = 0
        for dets, obs in raw:
            if decode(list(dets)) != obs:
                mis += 1
    return decode, len(raw), mis


_MWPF_POOL_G = {}


def _mwpf_pool_init(circuit_text):
    import stim
    _MWPF_POOL_G["decode"] = _mwpf_decoder(stim.Circuit(circuit_text),
                                           gate=False)[0]


def _audit_dict(mis, n_mech):
    return {"miscorrected": mis, "mechanisms": n_mech}


def _mwpf_pool_chunk(chunk):
    decode = _MWPF_POOL_G["decode"]
    return sum(1 for defects, mask in chunk if decode(defects) != mask)


def _mwpf_sample_ler(circuit, seed, target_errors, max_shots,
                     workers=MWPF_WORKERS):
    """~1.7 s/shot serial on twistedghz_4 d5 (measured 2026-08-10) makes
    a 100-error budget serially infeasible; shots are independent, so
    decode them in a process pool (each worker builds its own solver —
    construction is <1 s — and skips the audit the parent already ran)."""
    if circuit.num_observables == 0:
        raise ValueError("_mwpf_sample_ler: circuit has no observables "
                         "(dagger row) -- LER undefined, refusing to sample")
    # audit dropped: system-level LER ruling (design decision 2026-08-13) — on
    # bv_n70 the parent-side mechanism sweep alone burned 2.5 h.
    sampler = circuit.compile_detector_sampler(seed=seed)
    shots = joint = 0
    BATCH = 20_000
    # spawn, NOT fork: the case process is forked from main()'s
    # ThreadPoolExecutor threads — forking AGAIN from that state hands
    # the pool workers inherited locks and a mangled interpreter (wave-2
    # 2026-08-10: workers died with impossible stack traces, pool.map
    # hung on the dead workers, all four cases burned the full timeout).
    # Spawned workers re-import cleanly and rebuild the decoder from the
    # circuit text (~1-2 s each, once per case).
    with mp.get_context("spawn").Pool(
            workers, initializer=_mwpf_pool_init,
            initargs=(str(circuit),)) as pool:
        while joint < target_errors and shots < max_shots:
            n = min(BATCH, max_shots - shots)
            det, obs = sampler.sample(n, separate_observables=True)
            items = []
            for k in range(n):
                mask = 0
                for b in np.flatnonzero(obs[k]):
                    mask ^= (1 << int(b))
                items.append(([int(i) for i in np.flatnonzero(det[k])],
                              mask))
            step = max(1, len(items) // (workers * 4))
            chunks = [items[i:i + step]
                      for i in range(0, len(items), step)]
            joint += sum(pool.map(_mwpf_pool_chunk, chunks))
            shots += n
    return shots, joint, {"skipped": "ruling 2026-08-13"}


def _sample_ler(circuit, seed, target_errors, max_shots):
    import pymatching
    if circuit.num_observables == 0:
        # match _mwpf_sample_ler's guard: with no observable the pymatching
        # path would XOR against a width-0 obs array, count zero failures and
        # report a spurious LER of 0.0 instead of raising (fix 2026-08-24).
        raise ValueError("_sample_ler: circuit has no observables (dagger "
                         "row) -- LER undefined, refusing to sample")
    # audit dropped: system-level LER ruling (2026-08-13) — the audit
    # is quadratic in circuit size and burned the rescue's ghz_32/dj_32
    # d3 cells to their 2 h timeout; build the matching directly.
    try:
        matching = pymatching.Matching.from_detector_error_model(
            circuit.detector_error_model(decompose_errors=True))
    except ValueError as e:
        if ("Failed to decompose" not in str(e) and
                "observable ids larger than 63" not in str(e)):
            raise
        shots, joint, audit = _mwpf_sample_ler(circuit, seed,
                                               target_errors, max_shots)
        return shots, joint, "mwpf", audit
    sampler = circuit.compile_detector_sampler(seed=seed)
    shots = joint = 0
    BATCH = 20_000
    while joint < target_errors and shots < max_shots:
        n = min(BATCH, max_shots - shots)
        det, obs = sampler.sample(n, separate_observables=True)
        pred = matching.decode_batch(det).astype(bool)
        joint += int((pred ^ obs).any(axis=1).sum())
        shots += n
    return shots, joint, "mwpm", {"skipped": "ruling 2026-08-13"}


def calibrate_block(d, p, seed, target_errors, max_shots):
    """eps_block from single-patch memory: LER(R) at two R, slope*d.

    The same compile path (no tricks) on a 1-qubit measure-only program;
    ``rounds`` scales the memory length.  Two lengths difference out the
    boundary (init/readout) contribution: eps_round = (LER2-LER1)/(R2-R1),
    eps_block = eps_round * d."""
    from circls.pipeline import compile_qasm
    from noise_inject import inject_uniform_noise
    qasm = ('OPENQASM 2.0;\ninclude "qelib1.inc";\n'
            'qreg q[1];\ncreg c[1];\nmeasure q -> c;\n')
    out = {}
    lers = {}
    for tag, R in (("R1", 10 * d), ("R2", 20 * d)):
        with contextlib.redirect_stdout(io.StringIO()):
            cp = compile_qasm(qasm, distance=d, rounds_init=R,
                              keep_patches=None, **NOTRICK)
        noisy = inject_uniform_noise(cp.circuit, p)   # uniform pass (2026-08-23)
        shots, joint, _dec, _aud = _sample_ler(noisy, seed,
                                               target_errors, max_shots)
        lers[tag] = joint / shots
        out[tag] = {"rounds": R, "shots": shots, "joint_errors": joint,
                    "ler": lers[tag]}
        seed += 1
    eps_round = (lers["R2"] - lers["R1"]) / (10 * d)
    out["eps_round"] = eps_round
    out["eps_block"] = eps_round * d
    return out


def _case_worker(case, d, p, eps_block, seed, target_errors, max_shots,
                 conn):
    """One case in an ISOLATED process: the no-trick compile of a 32-qubit
    hub program can grind for hours (the A2 no_reduce timeout family), and
    pymatching state accumulates — the parent enforces a timeout and gets
    a fresh address space per case (measured 2026-08-06: the in-process
    loop hit 44 GB RSS and wedged on the case after ghz_32)."""
    from circls.pipeline import compile_qasm
    from circls.metrics import experiment_stats
    from circls.metrics.report import to_dict
    from noise_inject import inject_uniform_noise
    t0 = time.perf_counter()
    row = {"record": "case", "name": case.name, "d": d, "p": p,
           "n": case.n_qubits}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            cp = compile_qasm(case.qasm, distance=d,
                              keep_patches=None, **NOTRICK)
        es = experiment_stats(cp.experiment, cp.circuit)
        st = to_dict(es)
        v1 = st["V1_volume_blocks"]
        noisy = inject_uniform_noise(cp.circuit, p)   # uniform pass (2026-08-23)
        shots, joint, decoder, audit = _sample_ler(noisy, seed,
                                                   target_errors, max_shots)
        meas = joint / shots
        pred_pub = min(1.0, v1 * PUB_A * (p / PUB_PTH) ** ((d + 1) / 2))
        pred_cal = min(1.0, v1 * eps_block)
        row.update({"status": "OK", "V1": v1, "stats": st, "shots": shots,
                    "joint_errors": joint, "measured": meas,
                    "decoder": decoder, "audit": audit,
                    **({"decoder_config": {"timeout_s": MWPF_TIMEOUT_S,
                                           "workers": MWPF_WORKERS}}
                       if decoder == "mwpf" else {}),
                    "pred_published": pred_pub, "pred_calibrated": pred_cal,
                    "ratio_published": meas / pred_pub if pred_pub else None,
                    "ratio_calibrated": meas / pred_cal if pred_cal else None,
                    "seed": seed,
                    "seconds": round(time.perf_counter() - t0, 1)})
    except Exception as e:
        row.update({"status": type(e).__name__, "error": str(e)[:200],
                    "seconds": round(time.perf_counter() - t0, 1)})
    conn.send(row)
    conn.close()


def run_case(case, d, p, eps_block, seed, target_errors, max_shots,
             timeout):
    # spawn, not fork: run_case is called from ThreadPoolExecutor
    # threads, and a case that takes the mwpf fallback creates a pool
    # INSIDE its process — a fork child of a threaded parent inherits
    # held locks and wedges on nested process creation (waves 2-3
    # 2026-08-10 burned full timeouts; every stage runs clean from a
    # single-threaded parent).  Spawn re-imports the module fresh.
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    proc = ctx.Process(target=_case_worker,
                       args=(case, d, p, eps_block, seed, target_errors,
                             max_shots, child))
    proc.start()
    if parent.poll(timeout):
        row = parent.recv()
    else:
        row = {"record": "case", "name": case.name, "d": d, "p": p,
               "n": case.n_qubits, "status": "TIMEOUT",
               "seconds": timeout}
        proc.terminate()
    proc.join(5)
    if proc.is_alive():
        proc.kill()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--d", nargs="+", type=int, default=[3, 5])
    ap.add_argument("--p", nargs="+", type=float, default=[1e-3, 5e-4])
    ap.add_argument("--target-errors", type=int, default=100)
    ap.add_argument("--max-shots", type=int, default=2_000_000)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=47)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these case names (surgical re-runs, "
                         "e.g. retrying error rows with the mwpf fallback)")
    ap.add_argument("--outdir", default="experiments/results/formula_dev")
    args = ap.parse_args()
    prov = provenance(allow_dirty=args.allow_dirty)

    from circls.pipeline import compile_qasm
    from circls.metrics import experiment_stats
    from circls.metrics.report import to_dict

    outdir = Path(args.outdir + ("_quick" if args.quick else ""))
    outdir.mkdir(parents=True, exist_ok=True)
    # resume: a killed run's rows stay valid (append-only file, analysis
    # takes the last row per key) — skip finished cases and reuse
    # calibrations from the same file
    done, cal_cache = set(), {}
    ded = outdir / "deviation.jsonl"
    if ded.exists():
        for line in open(ded):
            r = json.loads(line)
            if r.get("record") == "case" and r.get("status") == "OK":
                done.add((r["name"], r["d"], r["p"]))
            elif r.get("record") == "calibration":
                cal_cache[(r["d"], r["p"])] = r
    fh = open(ded, "a")
    fh.write(json.dumps({**prov, "record": "provenance",
                         "pub_a": PUB_A, "pub_pth": PUB_PTH}) + "\n")
    fh.flush()

    cases = [c for c in suite()
             if c.n_qubits <= 32 or c.name == "ghz_16_mixed"]
    if args.quick:
        cases = [c for c in cases if c.name in QUICK_CASES]
    if args.only:
        cases = [c for c in cases if c.name in args.only]

    idx = 0
    for d in args.d:
        for p in args.p:
            if (d, p) in cal_cache:
                cal = cal_cache[(d, p)]
                idx += 1
            else:
                cal = calibrate_block(d, p, args.seed + 1000 * idx,
                                      args.target_errors, args.max_shots)
                idx += 1
                fh.write(json.dumps({"record": "calibration", "d": d,
                                     "p": p, **cal}) + "\n")
                fh.flush()
            print(f"[cal] d={d} p={p:g}: eps_block={cal['eps_block']:.3e} "
                  f"(published {PUB_A * (p / PUB_PTH) ** ((d + 1) / 2):.3e})",
                  flush=True)
            todo = [c for c in cases if (c.name, d, p) not in done]

            def _job(case, _d=d, _p=p, _cal=cal, _idx=idx):
                seed = (args.seed + 10_000 + _idx * 500
                        + hash(case.name) % 499)
                return run_case(case, _d, _p, _cal["eps_block"], seed,
                                args.target_errors, args.max_shots,
                                args.timeout)

            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(args.workers) as pool:
                for row in pool.map(_job, todo):
                    if row["status"] == "OK":
                        print(f"[ OK ] d={d} p={p:g} {row['name']:16s} "
                              f"V1={row['V1']:.0f} "
                              f"meas={row['measured']:.3e} "
                              f"pred_cal={row['pred_calibrated']:.3e} "
                              f"ratio={row['ratio_calibrated']:.2f}",
                              flush=True)
                    else:
                        print(f"[{row['status']:.10s}] d={d} p={p:g} "
                              f"{row['name']}: {row.get('error', '')[:80]}",
                              flush=True)
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
    fh.close()
    print(f"-> {outdir}/deviation.jsonl", flush=True)


if __name__ == "__main__":
    main()
