"""Compile one distance in a fresh interpreter.

tqec's emitted circuit depends on state left behind by earlier compiles in the same
process (a second compile of the same graph inserts extra TICKs), so each distance is
compiled in its own process; that also keeps the circuits byte-identical to the ones
behind the paper's Table 2, which were produced one distance per process.

usage: compile_worker.py BGRAPH N_DATA ORIGINAL_QASM K OUT_STIM OUT_JSON
"""
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from topologiq_bridge import ZXCube, build, compile_closed, equivalent, fill_rule  # noqa: E402

bgraph, n_data, qasm, k, out_stim, out_json = sys.argv[1:7]
g, info = build(bgraph, int(n_data))
ok, why, (lead, trail, idle) = equivalent(g, pathlib.Path(qasm).read_text(), drop_s=True)
assert ok, why
g.fill_ports({lb: ZXCube.from_str(v) for lb, v in fill_rule(g, lead, trail).items()})
g.validate()
surfaces = g.find_correlation_surfaces()
t0 = time.perf_counter()
circ, conv, chk, errs = compile_closed(g, surfaces, int(k))
seconds = time.perf_counter() - t0
rep = {"k": int(k), "d": 2 * int(k) + 1, "convention": conv, "seconds": round(seconds, 1), "errors": errs,
       "p0": chk, "qubits": circ.num_qubits if circ else None, "detectors": circ.num_detectors if circ else None,
       "observables": circ.num_observables if circ else None}
if circ is not None:
    pathlib.Path(out_stim).write_text(str(circ))
pathlib.Path(out_json).write_text(json.dumps(rep, indent=1))
