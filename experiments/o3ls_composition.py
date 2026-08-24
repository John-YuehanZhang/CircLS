"""Score the O3LS composition formula (intro Challenge 3) directly.

    p_pred = sum_t [ 1 - (1 - P_PPM^(t)) (1 - P_idle^(t)) ]

with P_PR = 0 in the Clifford-only regime.  Per design decision 2026-08-12
this is the paper's main target: the per-layer product-sum over
components RATED IN ISOLATION, exactly as the formula prescribes.

- P_PPM^(t): the layer's joint PPM as its own mini-program (only the
  participating logical qubits, the same joint measurement, terminal
  readouts in the interaction bases so every bit is deterministic),
  compiled with the same no-trick settings and merge window and
  sampled to `--component-errors` failures.  Cached per (ordered
  bases signature, d, p).
- P_idle^(t): 1 - (1 - eps_round)^(rounds * n_idle) with eps_round
  from the B-dev calibration records (same pipeline, decoder, noise).
- Measured program LER: reused from deviation.jsonl (B-dev, no-trick).

Cases whose expanded program carries gadgets or non-mpp ops are
recorded as `undecomposed` (out of the formula's Clifford scope here).
"""
import argparse
import contextlib
import io
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "experiments"))

DEV = _ROOT / "experiments/results/best5/formula_dev/deviation.jsonl"
OUT = _ROOT / "experiments/results/best5/formula_dev/o3ls_composition.jsonl"

NOTRICK = dict(assignment="row_major", measure_reduction=False,
               step_scheduling=False, parallel_steps=False,
               first_use_init=False, liveness=False)


def _load_dev():
    measured, calib = {}, {}
    for line in open(DEV):
        r = json.loads(line)
        if r.get("record") == "calibration":
            calib[(r["d"], r["p"])] = r["eps_round"]
        elif r.get("record") == "case" and r.get("status") == "OK":
            measured[(r["name"], r["d"], r["p"])] = r
    return measured, calib


def _component_ler(sig, d, p, rounds, seed, target, max_shots):
    """P_PPM for one isolated joint PPM with ordered bases `sig`."""
    from circls.interop.ir.gosc_gadgets import (OutBit, PPMProgram, ProgramOp,
                                             append_program_observables,
                                             to_experiment_inputs)
    from circls.interop.ir.ppm_import import patch_name
    from circls.core.sequential_ppm_ls import SequentialPPMExperiment
    from formula_deviation import _sample_ler
    from noise_inject import inject_uniform_noise

    w = len(sig)
    ops = [ProgramOp("mpp", {i: sig[i] for i in range(w)})]
    ops += [ProgramOp("mpp", {i: sig[i]}) for i in range(w)]
    prog = PPMProgram(num_data=w, ops=ops, gadgets=[],
                      out_bits=[OutBit(rec=k, flip=0)
                                for k in range(w + 1)])
    specs, steps, init, final, _ = to_experiment_inputs(
        prog, distance=d, assignment="row_major")
    init = {patch_name(i): sig[i] for i in range(w)}
    final = dict(init)
    with contextlib.redirect_stdout(io.StringIO()):
        exp = SequentialPPMExperiment(
            specs, steps, initial_states=init, final_measure_states=final,
            rounds=rounds, rounds_init=1, first_use_init=False,
            liveness=False, parallel_steps=False)
        circuit = exp.build()
        append_program_observables(circuit, exp, prog)
    det, obs = circuit.compile_detector_sampler(seed=0).sample(
        64, separate_observables=True)
    if det.any() or (obs != obs[0]).any():
        raise RuntimeError(f"mini for sig {sig} not deterministic at p=0")
    noisy = inject_uniform_noise(circuit, p)
    shots, joint, dec, aud = _sample_ler(noisy, seed, target, max_shots)
    return {"sig": "".join(sig), "shots": shots, "joint_errors": joint,
            "ler": joint / shots, "decoder": dec}


def _cell_worker(task):
    """All cases of one (d, p) grid cell, sharing a component cache."""
    (d, p, names, eps_round, comp_target, comp_cap, seed0) = task
    from benchsuite import suite
    from circls.interop.ir.gosc_gadgets import program_step_interactions
    from circls.pipeline import compile_qasm

    cases = {c.name: c for c in suite()}
    cache, rows, seed = {}, [], seed0
    for name in names:
        t0 = time.perf_counter()
        row = {"record": "o3ls", "name": name, "d": d, "p": p}
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                cp = compile_qasm(cases[name].qasm, distance=d,
                                  keep_patches=None, **NOTRICK)
            prog = cp.program
            if prog.gadgets or any(op.kind != "mpp" for op in prog.ops):
                row["status"] = "undecomposed"
                rows.append(row)
                continue
            steps = program_step_interactions(prog)
            n_patches = len(cp.experiment.patches)
            rounds = cp.experiment.rounds
            p_pred, comps = 0.0, []
            for it in steps:
                sig = tuple(P for _, P in it)
                if sig not in cache:
                    seed += 1
                    cache[sig] = _component_ler(
                        sig, d, p, rounds, seed, comp_target, comp_cap)
                c = cache[sig]
                n_idle = n_patches - len(sig)
                p_idle = 1 - (1 - eps_round) ** (rounds * n_idle)
                p_pred += 1 - (1 - c["ler"]) * (1 - p_idle)
                comps.append({"sig": c["sig"], "p_ppm": c["ler"],
                              "p_idle": p_idle})
            row.update({"status": "OK", "n_steps": len(steps),
                        "n_patches": n_patches, "rounds": rounds,
                        "pred_o3ls": p_pred, "components": comps,
                        "seconds": round(time.perf_counter() - t0, 1)})
        except Exception as e:
            row.update({"status": type(e).__name__,
                        "error": str(e)[:200]})
        rows.append(row)
        print(f"[d={d} p={p:g}] {name}: {row['status']} "
              f"pred={row.get('pred_o3ls')}", flush=True)
    comp_rows = [{"record": "component", "d": d, "p": p, **c}
                 for c in cache.values()]
    return rows + comp_rows


def main():
    from provenance import provenance

    ap = argparse.ArgumentParser()
    ap.add_argument("--component-errors", type=int, default=200)
    ap.add_argument("--component-cap", type=int, default=2_000_000)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--seed", type=int, default=211)
    args = ap.parse_args()
    prov = provenance()

    measured, calib = _load_dev()
    cells = {}
    for (name, d, p) in measured:
        cells.setdefault((d, p), []).append(name)
    tasks = []
    for i, ((d, p), names) in enumerate(sorted(cells.items())):
        tasks.append((d, p, sorted(names), calib[(d, p)],
                      args.component_errors, args.component_cap,
                      args.seed + 1000 * i))
    fh = open(OUT, "a")
    fh.write(json.dumps({**prov, "record": "provenance",
                         "args": vars(args)}) + "\n")
    fh.flush()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for rows in pool.imap_unordered(_cell_worker, tasks):
            for r in rows:
                if r["record"] == "o3ls" and r.get("status") == "OK":
                    m = measured[(r["name"], r["d"], r["p"])]["measured"]
                    r["measured"] = m
                    r["ratio_o3ls"] = (m / r["pred_o3ls"]
                                       if r["pred_o3ls"] else None)
                fh.write(json.dumps(r) + "\n")
            fh.flush()
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
