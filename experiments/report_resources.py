"""One-key resource report.

Usage:
    python experiments/report_resources.py                 # built-in demos
    python experiments/report_resources.py path/to/x.stim  # layer-1 stats of any circuit
"""
import contextlib
import io
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT.parent / "LightStim"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import stim  # noqa: E402

from circls.metrics import circuit_stats, experiment_stats, timed_build  # noqa: E402
from circls.metrics.report import format_circuit_summary, format_summary  # noqa: E402
from circls.core.routed_multi_patch_ls import PatchSpec, origin_of  # noqa: E402
from circls.core.sequential_ppm_ls import PPMStep, SequentialPPMExperiment  # noqa: E402

D = 3


def _demo_two_patch():
    px = [PatchSpec("Q1", origin_of(0, 0, D, seam=True), D, "X_horizontal"),
          PatchSpec("Q2", origin_of(2, 0, D, seam=True), D, "X_horizontal")]
    return SequentialPPMExperiment(
        px, [PPMStep([("Q1", "Z"), ("Q2", "Z")])],
        initial_states={"Q1": "Z", "Q2": "Z"},
        final_measure_states={"Q1": "Z", "Q2": "Z"},
        rounds=D, rounds_init=1)


def _demo_three_patch(liveness):
    px = [PatchSpec("Q1", origin_of(0, 0, D, seam=True), D, "X_horizontal"),
          PatchSpec("Q2", origin_of(2, 0, D, seam=True), D, "X_horizontal"),
          PatchSpec("Q3", origin_of(4, 0, D, seam=True), D, "X_horizontal")]
    seq = [PPMStep([("Q1", "Z"), ("Q2", "Z")]),
           PPMStep([("Q2", "Z"), ("Q3", "Z")])]
    states = {nm: "Z" for nm in ("Q1", "Q2", "Q3")}
    kw = dict(liveness=liveness)
    if liveness:
        kw["keep_patches"] = {"Q2", "Q3"}
    return SequentialPPMExperiment(
        px, seq, initial_states=states, final_measure_states=states,
        rounds=D, rounds_init=1, **kw)


def report_experiment(exp, title):
    with contextlib.redirect_stdout(io.StringIO()):
        circuit, secs = timed_build(exp)
    es = experiment_stats(exp, circuit, compile_seconds=secs)
    print(format_summary(es, title))
    print()
    return es


def main(argv):
    if argv:
        for path in argv:
            cs = circuit_stats(stim.Circuit(Path(path).read_text()))
            print(format_circuit_summary(cs, f"[layer 1] {path}"))
            print()
        return
    report_experiment(_demo_two_patch(), "demo: 2-patch adjacent ZZ (d=3)")
    report_experiment(_demo_three_patch(False), "demo: 3-patch, liveness OFF")
    report_experiment(_demo_three_patch(True), "demo: 3-patch, liveness ON")


if __name__ == "__main__":
    main(sys.argv[1:])
