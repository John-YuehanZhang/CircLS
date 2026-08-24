"""Distance probes for the two Table-1 seam cases no benchmark exercises.

The mapper's orientation choices systematically avoid two of the four
seam cases (same bases + different parities = the stretched seam;
different bases + same parities = the domain-wall seam), so the
benchmark sweep gives them no distance evidence.  This probe builds
each case directly as a purpose-built two-patch program: the patch
positions, orientations and colour conventions are pinned (the choices
the mapper normally makes), and everything downstream -- seam
classification, construction, circuit emission, noise, the graphlike
search -- is the production path.  Mirrors the row-2/row-4 fixtures of
tests/test_rule_table_dispatch.py.

Self-check: each probe asserts the dispatched rule row, the wall flag,
and p=0 determinism before searching, so a probe cannot silently land
in a different case.

Usage:
    python experiments/seam_case_probes.py [--allow-dirty] \
        [--out experiments/results/best5/seam_case_probes.jsonl]
"""
import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "experiments"))

#: (label, code row, wall?, specs-lambda, target, states, colour_swapped)
CASES = [
    ("stretched seam (same bases, different parities)", 2, True,
     lambda d: [("A", 0, 0, "X_horizontal"), ("B", 0, 1, "X_vertical")],
     [("A", "X"), ("B", "X")], {"A": "X", "B": "X"}, {"B"}),
    ("domain-wall seam (different bases, same parities)", 4, False,
     lambda d: [("A", 0, 0, "X_vertical"), ("B", 1, 0, "X_vertical")],
     [("A", "X"), ("B", "Z")], {"A": "X", "B": "Z"}, {"B"}),
]


def _probe(label, row, wall, cells, target, states, swapped, d):
    from circls.core.routed_multi_patch_ls import PatchSpec, origin_of
    from circls.core.sequential_ppm_ls import PPMStep, SequentialPPMExperiment
    from lightstim.noise.config import NoiseConfig

    out = {"record": "seam_case", "case": label, "expect_row": row, "d": d}
    t0 = time.perf_counter()
    specs = [PatchSpec(nm, origin_of(a, b, d, seam=True), d, o)
             for nm, a, b, o in cells]
    exp = SequentialPPMExperiment(
        specs, [PPMStep(target)], initial_states=states,
        final_measure_states=states, rounds=d, rounds_init=1,
        colour_swapped=frozenset(swapped))
    with contextlib.redirect_stdout(io.StringIO()):
        circuit = exp.build()
    out["rule_row"] = exp._rules[0].row
    out["wall"] = 0 in exp._walls
    assert out["rule_row"] == row, f"{label}: landed in row {out['rule_row']}"
    assert out["wall"] == wall
    det, obs = circuit.compile_detector_sampler(seed=0).sample(
        512, separate_observables=True)
    assert not det.any(), "detector fired at p=0"
    assert not obs.any(), "observable not deterministic at p=0"
    noisy = exp.builder.build_noisy_circuit(
        noise_params=NoiseConfig(p_1q=1e-3, p_2q=1e-3, p_meas=1e-3,
                                 p_reset=1e-3, p_idle=1e-3),
        noise_model="circuit_level")
    out["graphlike_distance"] = len(noisy.shortest_graphlike_error())
    out["status"] = "OK"
    out["seconds"] = round(time.perf_counter() - t0, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--out", default=str(
        _ROOT / "experiments/results/best5/seam_case_probes.jsonl"))
    args = ap.parse_args()

    from provenance import provenance
    prov = provenance(args.allow_dirty)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as fh:
        fh.write(json.dumps({**prov, "record": "provenance",
                             "argv": sys.argv}) + "\n")
        for label, row, wall, cells, target, states, swapped in CASES:
            for d in (3, 5):
                r = _probe(label, row, wall, cells(d), target, states,
                           swapped, d)
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                print(f"{label:48s} d={d} row={r['rule_row']} "
                      f"gl={r['graphlike_distance']} ({r['seconds']}s)",
                      flush=True)


if __name__ == "__main__":
    main()
