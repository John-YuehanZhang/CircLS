"""A1(1): head-to-head against tqec on its executable envelope.

Protocol (user-approved 2026-08-05): same logical computation, same
distance (k=1,2 <-> d=3,5), ONE shared noise pass on both sides'
noiseless circuits (experiments/noise_inject.py), same decoder
(pymatching) behind the single-error faithfulness gate.  End-to-end
track: each compiler free to optimize its own input form (tqec: block
graph; ours: QASM).

Pairs v1 — entries whose port fills map to unambiguous QASM (three_cnots
and steane_encoding join after their port->qubit mapping is verified
against tqec source):

  memory          Z-basis idle qubit
  cnot            |++> CX, X-basis readout (fill: all ZXX)
  cz              |00> CZ, Z-basis readout (fill: XZZ/XXZ)
  move_rotation   identity channel (their X->X move+rotate vs our patch
                  in place — the fixed-qubit story, table note)

Usage:
    python experiments/compare_tqec.py [--k 1 2] [--p 2e-3 1e-3 5e-4]
        [--target-errors 100] [--max-shots 1000000]
        [--out experiments/results/tqec_compare/main.jsonl]
"""
import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "experiments"))

from noise_inject import inject_uniform_noise      # noqa: E402
from provenance import provenance                  # noqa: E402

_HDR = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
PAIRS = {
    "memory": _HDR + "qreg q[1];\ncreg c[1];\nmeasure q -> c;\n",
    "cnot": _HDR + ("qreg q[2];\ncreg c[2];\nh q[0];\nh q[1];\n"
                    "cx q[0],q[1];\nh q[0];\nh q[1];\nmeasure q -> c;\n"),
    "cz": _HDR + "qreg q[2];\ncreg c[2];\ncz q[0],q[1];\nmeasure q -> c;\n",
    "move_rotation": _HDR + "qreg q[1];\ncreg c[1];\nmeasure q -> c;\n",
}

TQEC_DIR = Path(__file__).resolve().parent / "results" / "tqec_circuits"


def faithful_or_raise(circuit, chunk=20_000):
    """Single-error faithfulness gate, CHUNKED: baseline circuits reach
    367k mechanisms x 19k detectors (ghz_16 k=2) — a dense matrix is 7 GB
    and killed the first v2 run; blocks of ``chunk`` bound memory at
    ~150 MB with identical semantics."""
    import pymatching
    dem = circuit.detector_error_model(decompose_errors=True).flattened()
    m = pymatching.Matching.from_detector_error_model(dem)
    nd, no = dem.num_detectors, dem.num_observables
    mis = total = 0
    dets, obss = [], []

    def flush():
        nonlocal mis, dets, obss
        if dets:
            pred = m.decode_batch(np.array(dets)).astype(bool)
            mis += int((pred ^ np.array(obss)).any(axis=1).sum())
            dets, obss = [], []

    for inst in dem:
        if inst.type != "error":
            continue
        dv = np.zeros(nd, dtype=bool)
        ov = np.zeros(no, dtype=bool)
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                dv[t.val] ^= True
            elif t.is_logical_observable_id():
                ov[t.val] ^= True
        dets.append(dv)
        obss.append(ov)
        total += 1
        if len(dets) >= chunk:
            flush()
    flush()
    if mis:
        raise RuntimeError(f"decoder unfaithful: {mis}/{total}")
    return m


def adaptive_ler(circuit, matching, seed, target, max_shots):
    if circuit.num_observables == 0:
        # a dagger-class program (every out bit a random coin): LER is
        # undefined, and sampling would silently record 0.0 -- the
        # teleportation d5 MWPF run burned two hours doing exactly that
        raise ValueError("adaptive_ler: circuit has no observables "
                         "(dagger row) -- LER undefined, refusing to sample")
    sampler = circuit.compile_detector_sampler(seed=seed)
    shots = joint = 0
    per = np.zeros(circuit.num_observables, dtype=np.int64)
    while joint < target and shots < max_shots:
        n = min(10_000, max_shots - shots)
        det, obs = sampler.sample(n, separate_observables=True)
        wrong = matching.decode_batch(det).astype(bool) ^ obs.astype(bool)
        per += wrong.sum(axis=0)
        joint += int(wrong.any(axis=1).sum())
        shots += n
    return shots, joint, per.tolist()


def side_metrics(noiseless):
    from circls.metrics.circuit_stats import circuit_stats
    cs = circuit_stats(noiseless)
    return {"qubits_active": cs.qubits_active,
            "qubits_allocated": cs.qubits_allocated,
            "rounds": cs.measurement_layers,
            "qubit_rounds": cs.qubit_rounds,
            "num_detectors": noiseless.num_detectors,
            "num_observables": noiseless.num_observables}


def main():
    import stim
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--p", nargs="+", type=float, default=[2e-3, 1e-3, 5e-4])
    ap.add_argument("--target-errors", type=int, default=100)
    ap.add_argument("--max-shots", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--out",
                    default="experiments/results/tqec_compare/main.jsonl")
    args = ap.parse_args()
    prov = provenance(allow_dirty=args.allow_dirty)
    manifest = json.loads((TQEC_DIR / "manifest.json").read_text())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fh = open(args.out, "a")
    fh.write(json.dumps({**prov, "tqec": {k: manifest[k] for k in
                                          ("tqec_version", "tqec_sha")},
                         "args": vars(args)}) + "\n")
    fh.flush()

    from circls.pipeline import compile_qasm
    idx = 0
    for name, qasm in PAIRS.items():
        for k in args.k:
            d = 2 * k + 1
            tq_noiseless = stim.Circuit(
                (TQEC_DIR / f"{name}_k{k}.stim").read_text())
            with contextlib.redirect_stdout(io.StringIO()):
                cp = compile_qasm(qasm, distance=d, assignment="optimized",
                                  liveness=True, keep_patches=set())
            ours_noiseless = cp.circuit
            row = {"pair": name, "k": k, "d": d,
                   "tqec": side_metrics(tq_noiseless),
                   "ours": side_metrics(ours_noiseless), "ler": []}
            for p in args.p:
                pt = {"p": p}
                for side, circ in (("tqec", tq_noiseless),
                                   ("ours", ours_noiseless)):
                    t0 = time.perf_counter()
                    noisy = inject_uniform_noise(circ, p)
                    try:
                        m = faithful_or_raise(noisy)
                        seed = args.seed + idx
                        idx += 1
                        shots, joint, per = adaptive_ler(
                            noisy, m, seed, args.target_errors,
                            args.max_shots)
                        pt[side] = {"shots": shots, "joint_errors": joint,
                                    "per_obs_errors": per, "seed": seed,
                                    "ler": joint / shots,
                                    "seconds": round(
                                        time.perf_counter() - t0, 1)}
                    except Exception as e:
                        pt[side] = {"status": type(e).__name__,
                                    "error": str(e)[:200]}
                row["ler"].append(pt)
                tq = pt.get("tqec", {})
                us = pt.get("ours", {})
                print(f"{name} k={k} p={p:g}: "
                      f"tqec={tq.get('ler', tq.get('status')):} "
                      f"ours={us.get('ler', us.get('status')):}", flush=True)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(f"[{name} k={k}] tqec qubits={row['tqec']['qubits_active']}"
                  f"/{row['tqec']['qubits_allocated']} "
                  f"rounds={row['tqec']['rounds']} | "
                  f"ours qubits={row['ours']['qubits_active']}"
                  f"/{row['ours']['qubits_allocated']} "
                  f"rounds={row['ours']['rounds']}", flush=True)
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
