"""Compile one Table 2 program with CircLS and print the row's cost metrics.

    python gallery/tools/table2_row.py qec_en_n5                    # d = 3, configuration "reselect"
    python gallery/tools/table2_row.py teleportation_n3 --d 5
    python gallery/tools/table2_row.py qec_en_n5 --config off       # all three optional passes off
    python gallery/tools/table2_row.py qec_en_n5 --qasm my.qasm     # any Clifford+T QASM instead

Configurations: optimized mapping, first-use initialization and last-use freeing are always on;
the three optional passes are
    reselect   terminal-measurement re-selection ON, step scheduler OFF, parallel windows OFF
    off        all three OFF
    full       all three ON
and each can be set by hand with --reselect / --reorder / --parallel on|off.

The program is read from experiments/table2_inputs/<program>.qasm, the circuit every toolchain
in Table 2 compiles (see the README there), with a Z readout of every qubit appended.  Every T
is read as a pi/4 rotation gadget and the gadget ancilla is prepared in |+> (the paper's X-state
proxy), so the circuit is Clifford and stim can sample it; the PPM sequence and every mapping and
routing decision are the ones the real program gets.
"""
import argparse
import contextlib
import io
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

import circls.pipeline as P                                   # noqa: E402
from circls.metrics.circuit_stats import circuit_stats        # noqa: E402
from circls.metrics.experiment_stats import experiment_stats  # noqa: E402

OFF = dict(assignment="optimized", first_use_init=True, liveness=True, keep_patches=set(),
           measure_reduction=False, step_scheduling=False, parallel_steps=False)
RESELECT = dict(OFF, measure_reduction=True)
FULL = dict(OFF, measure_reduction=True, step_scheduling=True, parallel_steps=True)
CONFIGS = {"reselect": RESELECT, "off": OFF, "full": FULL}
INPUTS = ROOT / "experiments" / "table2_inputs"


def x_state_proxy():
    """Prepare every gadget ancilla in |+> instead of |Y> (the paper's X-state proxy)."""
    orig = P.to_experiment_inputs

    def patched(*a, **k):
        specs, steps, init, final, prog = orig(*a, **k)
        return specs, steps, {nm: ("X" if v == "Y" else v) for nm, v in init.items()}, final, prog
    P.to_experiment_inputs = patched


def load_qasm(name, path):
    """The Table 2 input of ``name`` (or the file at ``path``) plus a Z readout of every qubit."""
    import re
    src = pathlib.Path(path) if path else INPUTS / f"{name}.qasm"
    if not src.exists():
        known = sorted(q.stem for q in INPUTS.glob("*.qasm"))
        sys.exit(f"no input for {name!r}; Table 2 programs: {known}, or pass --qasm")
    text = src.read_text()
    if re.search(r"^\s*measure\b", text, re.M):
        return text
    m = re.search(r"qreg\s+(\w+)\[(\d+)\]", text)
    return text.rstrip("\n") + f"\ncreg c[{m.group(2)}];\nmeasure {m.group(1)} -> c;\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("program")
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--config", choices=tuple(CONFIGS), default="reselect")
    ap.add_argument("--qasm", help="QASM file to compile instead of the Table 2 input")
    for sw in ("reselect", "reorder", "parallel"):
        ap.add_argument(f"--{sw}", choices=("on", "off"))
    a = ap.parse_args()
    kw = dict(CONFIGS[a.config])
    for sw, key in (("reselect", "measure_reduction"), ("reorder", "step_scheduling"), ("parallel", "parallel_steps")):
        v = getattr(a, sw)
        if v:
            kw[key] = (v == "on")
    x_state_proxy()
    with contextlib.redirect_stdout(io.StringIO()):
        cp = P.compile_qasm(load_qasm(a.program, a.qasm), distance=a.d, t_as_s=True, **kw)
    st = experiment_stats(cp.experiment, cp.circuit)
    cs = circuit_stats(cp.circuit)
    flags = ", ".join(f"{k} {'on' if kw[k] else 'off'}" for k in ("measure_reduction", "step_scheduling", "parallel_steps"))
    print(f"{a.program}, d = {a.d}: {flags}")
    print(f"  PPMs {len(cp.experiment.ppm_sequence)}, patches {len(cp.placement)} "
          f"({sum(1 for n in cp.placement if n.startswith('y'))} gadget ancillas), rounds {cs.measurement_layers}")
    print(f"  allocated volume {st.volume_blocks:.1f} blocks, qubit-cycles {cs.qubit_rounds}, "
          f"{cs.qubits_active} qubits, {cp.circuit.num_observables} observables")


if __name__ == "__main__":
    main()
