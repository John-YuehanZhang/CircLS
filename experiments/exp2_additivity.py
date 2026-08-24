"""Experiment 2: validation of the cross-sequence additivity rule (a
different PPM per layer, general version).

Program = 8 patches in two families (q1,q2,q5,q6 born in Z; q3,q4,q7,q8
born in X) and six mutually distinct PPM layers (support, bus letter and
corridor shape all vary, and all are aligned with the birth basis => every
outcome is verifiable, with zero rotations throughout at
rotate_saving_threshold=10):

  1. Z̄1Z̄2   horizontal pair, N/S three-tile hook corridor
  2. X̄3X̄4   horizontal pair, E/W one-tile straight seam
  3. Z̄5Z̄6   horizontal pair (second row)
  4. X̄7X̄8   horizontal pair (second row)
  5. Z̄1Z̄5   vertical pair, one-tile straight seam
  6. X̄3X̄7   vertical pair, E/W three-tile hook corridor

first_use_init=False: all 8 patches are present throughout -- the isolated
single-layer circuits keep the remaining patches idling as well, which is
the paper's notion of a "layer" (P_idle is charged to every layer). The
older family that repeats one and the same operator (the special case with
the strongest inter-layer correlation) is kept in git history as a control.

Following the user's convention "run the LER of each layer separately, then
add them up":
  components = the six single-layer circuits (all patches present, only
               layer k performed) plus null;
               dP_k = LER_k - LER_null (common SPAM subtracted);
  measured   = LER(L) of the first L layers run back to back;
  pred_add   = LER_null + sum_{k<=L} dP_k;
  pred_prod  = 1 - (1-LER_null) * prod_{k<=L} (1 - dP_k/(1-LER_null)).

Usage: python exp2_additivity.py [--smoke]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import harnesses  # noqa: E402

from circls.core.sequential_ppm_ls import PPMStep, SequentialPPMExperiment  # noqa: E402
from circls.core.routed_multi_patch_ls import PatchSpec, origin_of          # noqa: E402

COORDS = {"q1": (0, 0), "q2": (2, 0), "q3": (4, 0), "q4": (6, 0),
          "q5": (0, 2), "q6": (2, 2), "q7": (4, 2), "q8": (6, 2)}
BASIS = {"q1": "Z", "q2": "Z", "q5": "Z", "q6": "Z",
         "q3": "X", "q4": "X", "q7": "X", "q8": "X"}
LAYERS = [
    [("q1", "Z"), ("q2", "Z")],
    [("q3", "X"), ("q4", "X")],
    [("q5", "Z"), ("q6", "Z")],
    [("q7", "X"), ("q8", "X")],
    [("q1", "Z"), ("q5", "Z")],
    [("q3", "X"), ("q7", "X")],
]
L_LIST = [1, 2, 3, 4, 5, 6]
L_LIST_SMOKE = [1, 2, 3]
SEED0 = 20260803


def _experiment(d, steps):
    px = [PatchSpec(nm, origin_of(a, b, d, seam=True), d, "X_vertical")
          for nm, (a, b) in COORDS.items()]
    exp = SequentialPPMExperiment(
        px, [PPMStep(s) for s in steps],
        initial_states=dict(BASIS), final_measure_states=dict(BASIS),
        rounds=d, rounds_init=1, auto_rotate=True, rotation_kind='auto', rotate_saving_threshold=10,
        first_use_init=False)
    c = exp.build()
    assert exp.rotation_log == [], f"no rotation expected: {exp.rotation_log}"
    common.assert_valid_scenario(c, f"exp2 d={d} steps={steps}")
    return exp, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=common.DEFAULT_WORKERS)
    ap.add_argument("--d", type=int, nargs="*", default=None,
                    help="run only these d (default 3 5; d=7 automatically "
                         "uses only the three largest p)")
    ap.add_argument("--tag", type=str, default=None)
    args = ap.parse_args()
    p_shots = common.P_SHOTS_SMOKE if args.smoke else common.P_SHOTS
    d_list = (args.d if args.d
              else (common.D_LIST_SMOKE if args.smoke else common.D_LIST))
    l_list = L_LIST_SMOKE if args.smoke else L_LIST

    out = {"experiment": "exp2_additivity",
           "layers": [[list(t) for t in s] for s in LAYERS],
           "notes": ("the six layers are mutually distinct (support/letter/"
                     "corridor all vary); single-layer circuits keep every "
                     "patch present (idle is part of the layer semantics); "
                     "metric = any flip among all observables, with the "
                     "observable count recorded per point."),
           "points": []}

    for d in d_list:
        p_shots_d = [pp for pp in p_shots
                     if not (d >= 7 and pp[0] < 0.999 * 5e-4)]
        exps = {L: _experiment(d, LAYERS[:L]) for L in l_list}
        singles = {k: _experiment(d, [LAYERS[k]]) for k in range(max(l_list))}
        null = harnesses.memory_builder(COORDS, d, 0, rounds_init=1,
                                        basis=BASIS)
        common.assert_valid_scenario(null.circuit, f"exp2 null d={d}")
        for pi, (p, shots) in enumerate(p_shots_d):
            noise = common.uniform_noise(p)
            res_L = {}
            for L in l_list:
                exp, c = exps[L]
                noisy = exp.builder.build_noisy_circuit(
                    noise_params=noise, noise_model="circuit_level")
                res_L[L] = dict(
                    meas=common.run_point(
                        noisy, shots, SEED0 + 10_000 * d + 1000 * pi + 10 * L,
                        workers=args.workers,
                        min_errors=common.min_errors_for(p)),
                    ticks=noisy.num_ticks, observables=c.num_observables)
            comp = {}
            for k, e in singles.items():
                cc = e[0].builder.build_noisy_circuit(
                    noise_params=noise, noise_model="circuit_level")
                comp[k] = common.run_point(
                    cc, shots, SEED0 + 10_000 * d + 1000 * pi + 300 + k,
                    workers=args.workers,
                    min_errors=common.min_errors_for(p))
            null_noisy = null.build_noisy_circuit(
                noise_params=noise, noise_model="circuit_level")
            res0 = common.run_point(null_noisy, shots,
                                    SEED0 + 10_000 * d + 1000 * pi + 5,
                                    workers=args.workers,
                                    min_errors=common.min_errors_for(p))
            p0v = res0["ler"]
            dP = {k: max(comp[k]["ler"] - p0v, 0.0) for k in comp}
            dP_lo = {k: max(comp[k]["ler_lo"] - res0["ler_hi"], 0.0)
                     for k in comp}
            dP_hi = {k: max(comp[k]["ler_hi"] - res0["ler_lo"], 0.0)
                     for k in comp}
            for L in l_list:
                m = res_L[L]["meas"]
                pred_add = p0v + sum(dP[k] for k in range(L))
                pred_add_lo = res0["ler_lo"] + sum(dP_lo[k] for k in range(L))
                pred_add_hi = res0["ler_hi"] + sum(dP_hi[k] for k in range(L))
                prod = 1 - p0v
                for k in range(L):
                    prod *= (1 - dP[k] / max(1 - p0v, 1e-12))
                pred_prod = 1 - prod
                singles_txt = " ".join(f"{comp[k]['ler']:.1e}"
                                       for k in range(L))
                out["points"].append(dict(
                    d=d, p=p, L=L, shots=shots,
                    ticks=res_L[L]["ticks"], observables=res_L[L]["observables"],
                    **{f"meas_{k}": v for k, v in m.items()},
                    **{f"single{k}_ler": comp[k]["ler"] for k in comp},
                    **{f"single{k}_errors": comp[k]["errors"] for k in comp},
                    null_ler=p0v, null_errors=res0["errors"],
                    **{f"dP{k}": dP[k] for k in dP},
                    pred_add=pred_add, pred_add_lo=pred_add_lo,
                    pred_add_hi=pred_add_hi, pred_prod=pred_prod,
                    ratio_add=(m["ler"] / pred_add if pred_add > 0
                               else float("nan")),
                    ratio_add_lo=(m["ler_lo"] / pred_add_hi
                                  if pred_add_hi > 0 else float("nan")),
                    ratio_add_hi=(m["ler_hi"] / max(pred_add_lo, 1e-300)
                                  if pred_add_lo > 0 else float("nan")),
                    ratio_prod=(m["ler"] / pred_prod if pred_prod > 0
                                else float("nan"))))
                print(f"[d={d} p={p:.0e} L={L}] meas={m['ler']:.3e} "
                      f"pred_add={pred_add:.3e} (singles={singles_txt} "
                      f"null={p0v:.2e})", flush=True)

    tag = args.tag or ("smoke" if args.smoke else "full")
    common.save_json(f"exp2_additivity_{tag}", out)

    plt = common.new_figure()
    for d in d_list:
        for p, _ in p_shots:
            pts = sorted((q for q in out["points"]
                          if q["d"] == d and q["p"] == p), key=lambda q: q["L"])
            if not pts:
                continue
            fig, ax = plt.subplots(figsize=(5.2, 4))
            Ls = [q["L"] for q in pts]
            ms = [q["meas_ler"] for q in pts]
            lo = [q["meas_ler"] - q["meas_ler_lo"] for q in pts]
            hi = [q["meas_ler_hi"] - q["meas_ler"] for q in pts]
            ax.errorbar(Ls, ms, yerr=[lo, hi], marker="o", capsize=3,
                        label="measured")
            ax.plot(Ls, [q["pred_add"] for q in pts], "--", label="additive")
            ax.plot(Ls, [q["pred_prod"] for q in pts], ":", label="product")
            ax.set_xlabel("layers L (all distinct PPMs)")
            ax.set_ylabel("LER (any observable)")
            ax.set_title(f"Exp2: cross-sequence additivity  d={d}, p={p:.0e}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            common.save_figure(fig, f"exp2_L_d{d}_p{p:.0e}_{tag}")
            plt.close(fig)
    print("figures in", common.RESULTS)


if __name__ == "__main__":
    main()
