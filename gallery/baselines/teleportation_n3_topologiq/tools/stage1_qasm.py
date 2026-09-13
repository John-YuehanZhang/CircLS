"""Section 1: the circuit as QASMBench ships it, and the form topologiq receives
(each T and S rewritten as an injection gadget on a fresh magic wire; see gadget_rewrite.py)."""
import json

from gadget_rewrite import gadget_flow_check, rewrite
from saved import record_saved


def prepare_qasm(here, name):
    """-> (original_qasm_text, gadget_qasm_text, meta).  Writes data/1_*."""
    for sub in ("data", "blockgraph", "stim_circuit"):
        (here / sub).mkdir(exist_ok=True)
    original = (here / "data" / f"1_{name}.qasm").read_text()
    gadget, meta = rewrite(original, s_proxy=True)
    (here / "data" / f"1_{name}_gadgets.qasm").write_text(gadget)
    meta["proxy_gadget_flows"] = gadget_flow_check()
    (here / "data" / f"1_{name}_gadget_map.json").write_text(json.dumps(meta, indent=1))
    record_saved([(here / "data" / f"1_{name}.qasm", "QASMBench original"),
                        (here / "data" / f"1_{name}_gadgets.qasm", "what topologiq receives (T and S as gadgets)"),
                        (here / "data" / f"1_{name}_gadget_map.json", "which magic wire belongs to which gate")])
    return original, gadget, meta
