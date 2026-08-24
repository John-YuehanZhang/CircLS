"""Experiment 1: validation of the intra-layer product rule.

One layer = one PPM step (the planner's rotation window + merge window)
plus spectator idle.
Comparison:
  measured  = true sampled LER of the full circuit (PPM + rotation + idle
              in the same layer);
  predicted = 1 - (1-P_PPM)(1-P_PR)(1-P_idle), with each of the three
              components measured in isolation.

Component definitions (v0; open methodology points are listed in the README
and in the notes field of the result JSON):
  P_PPM  : target pair born directly in the pre-rotated orientation
           (X_horizontal), merge window only, no spectators;
  P_PR   : SWAP rotation window on the target pair (init+baseline ->
           rotation -> d SE rounds -> readout);
  P_idle : spectator pair, memory circuit, rounds matched by tick count to
           the duration of the full circuit;
  null   : init+baseline+readout of the four patches (pure overhead,
           recorded for reference only).

Usage: python exp1_intra_layer.py [--smoke]
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

TARGETS = {"q1": (0, 0), "q2": (2, 0)}
SPECTATORS = {"q3": (0, 2), "q4": (2, 2)}
SEED0 = 20260801


def _spec(nm, a, b, d, orient="X_vertical"):
    return PatchSpec(nm, origin_of(a, b, d, seam=True), d, orient)


def build_full(d):
    px = ([_spec(nm, *TARGETS[nm], d) for nm in TARGETS]
          + [_spec(nm, *SPECTATORS[nm], d) for nm in SPECTATORS])
    init = {nm: "Z" for nm in list(TARGETS) + list(SPECTATORS)}
    exp = SequentialPPMExperiment(
        px, [PPMStep([("q1", "Z"), ("q2", "Z")])],
        initial_states=init, final_measure_states=init,
        rounds=d, rounds_init=1, auto_rotate=True, rotation_kind='auto', rotate_saving_threshold=1,
        first_use_init=False)
    c = exp.build()
    return exp, c


def build_ppm_only(d):
    px = [_spec(nm, *TARGETS[nm], d, "X_horizontal") for nm in TARGETS]
    init = {nm: "Z" for nm in TARGETS}
    exp = SequentialPPMExperiment(
        px, [PPMStep([("q1", "Z"), ("q2", "Z")])],
        initial_states=init, final_measure_states=init,
        rounds=d, rounds_init=1, auto_rotate=True, rotation_kind='auto', rotate_saving_threshold=10)
    c = exp.build()
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

    out = {"experiment": "exp1_intra_layer",
           "formula": "1-(1-P_PPM)(1-P_PR)(1-P_idle)",
           "notes": ("v0 open methodology points: each component circuit "
                     "carries one copy of the init/baseline/readout overhead, "
                     "so the product triple-counts it (the null row gives its "
                     "magnitude); P_PR uses d SE rounds; P_idle matches the "
                     "full-circuit duration by tick count."),
           "points": []}

    for d in d_list:
        p_shots_d = [pp for pp in p_shots
                     if not (d >= 7 and pp[0] < 0.999 * 5e-4)]
        exp_full, c_full = build_full(d)
        exp_ppm, c_ppm = build_ppm_only(d)
        rot = harnesses.rotation_builder(TARGETS, d, list(TARGETS))
        idle_r = harnesses.match_idle_rounds(SPECTATORS, d,
                                             c_full.num_ticks, rounds_init=1)
        idle = harnesses.memory_builder(SPECTATORS, d, idle_r, rounds_init=1)
        null = harnesses.memory_builder({**TARGETS, **SPECTATORS}, d, 0,
                                        rounds_init=1)
        common.assert_valid_scenario(c_full, f"exp1 full d={d}")
        common.assert_valid_scenario(c_ppm, f"exp1 ppm d={d}")
        common.assert_valid_scenario(rot.circuit, f"exp1 pr d={d}")
        common.assert_valid_scenario(idle.circuit, f"exp1 idle d={d}")
        common.assert_valid_scenario(null.circuit, f"exp1 null d={d}")
        scen = dict(d=d, rotation_log=[list(t) for t in exp_full.rotation_log],
                    full_ticks=c_full.num_ticks,
                    full_observables=c_full.num_observables,
                    ppm_ticks=c_ppm.num_ticks,
                    rot_ticks=rot.circuit.num_ticks,
                    idle_rounds=idle_r, idle_ticks=idle.circuit.num_ticks)
        print(f"[d={d}] scenario: {scen}")

        for pi, (p, shots) in enumerate(p_shots_d):
            noise = common.uniform_noise(p)
            circs = dict(
                full=exp_full.builder.build_noisy_circuit(
                    noise_params=noise, noise_model="circuit_level"),
                ppm=exp_ppm.builder.build_noisy_circuit(
                    noise_params=noise, noise_model="circuit_level"),
                pr=rot.build_noisy_circuit(
                    noise_params=noise, noise_model="circuit_level"),
                idle=idle.build_noisy_circuit(
                    noise_params=noise, noise_model="circuit_level"),
                null=null.build_noisy_circuit(
                    noise_params=noise, noise_model="circuit_level"),
            )
            res = {}
            for j, (kk, cc) in enumerate(circs.items()):
                res[kk] = common.run_point(
                    cc, shots, seed=SEED0 + 1000 * d + 100 * pi + 10 * j,
                    workers=args.workers,
                    min_errors=common.min_errors_for(p))
            pred = common.compose_product([res["ppm"], res["pr"], res["idle"]])
            ratio = common.ratio_ci(res["full"], pred)
            point = dict(d=d, p=p, shots=shots, **{f"{k}_{kk}": vv
                         for k, r in res.items() for kk, vv in r.items()},
                         **{f"pred_{k}": v for k, v in pred.items()},
                         **{f"ratio_{k}": v for k, v in ratio.items()})
            out["points"].append(point)
            print(f"[d={d} p={p:.0e}] full={res['full']['ler']:.3e} "
                  f"pred={pred['pred']:.3e} ratio={ratio['ratio']:.3f} "
                  f"[{ratio['lo']:.3f},{ratio['hi']:.3f}]  "
                  f"(ppm={res['ppm']['ler']:.2e} pr={res['pr']['ler']:.2e} "
                  f"idle={res['idle']['ler']:.2e} null={res['null']['ler']:.2e})",
                  flush=True)
        out.setdefault("scenarios", []).append(scen)

    tag = args.tag or ("smoke" if args.smoke else "full")
    common.save_json(f"exp1_intra_layer_{tag}", out)

    plt = common.new_figure()
    fig, ax = plt.subplots(figsize=(5.2, 4))
    for d in d_list:
        pts = [q for q in out["points"] if q["d"] == d]
        ps = [q["p"] for q in pts]
        rs = [q["ratio_ratio"] for q in pts]
        lo = [q["ratio_ratio"] - q["ratio_lo"] for q in pts]
        hi = [q["ratio_hi"] - q["ratio_ratio"] for q in pts]
        ax.errorbar(ps, rs, yerr=[lo, hi], marker="o", capsize=3,
                    label=f"d={d}")
    ax.axhline(1.0, ls="--", c="gray", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("physical error rate p")
    ax.set_ylabel("measured / predicted")
    ax.set_title("Exp1: intra-layer product rule (full vs composed)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    print("figure:", common.save_figure(fig, f"exp1_ratio_{tag}"))


if __name__ == "__main__":
    main()
