"""Coverage experiment: export the paper's 45-program roster as
measurement-free QASM in the baseline's input format.

The baseline's front-end rejects QASM containing measure statements
(KeyError in idling_nodes_insertion_block), so each case is exported as
its unitary body with measure/creg/barrier lines stripped, under the
name c45_<case>.qasm in their benchmark directory.  The roster is the
ablation freeze (45 programs); bv_n140/bv_n280 joined the suite after
the freeze and stay out.
"""
import json
import sys
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "experiments"))

from benchsuite import suite  # noqa: E402

TOPOLS_BENCH = Path(os.environ.get(
    "TOPOLS_DIR",
    str(Path(__file__).resolve().parents[2] / "TopoLS"))) / "docs/benchmark"
EXCLUDE = {"bv_n140", "bv_n280"}          # post-freeze additions


def main():
    roster = []
    for c in suite():
        if c.name in EXCLUDE:
            continue
        kept = [l for l in c.qasm.splitlines()
                if not l.strip().startswith(("measure", "creg", "barrier"))]
        (TOPOLS_BENCH / f"c45_{c.name}.qasm").write_text(
            "\n".join(kept).rstrip() + "\n")
        roster.append({"name": c.name, "n": c.n_qubits,
                       "tags": sorted(c.tags)})
    out = _ROOT / "experiments/results/best5/topols_compare/coverage45_roster.json"
    out.write_text(json.dumps(roster, indent=1))
    print(f"{len(roster)} cases exported -> {TOPOLS_BENCH}/c45_*.qasm")
    print(f"roster -> {out}")


if __name__ == "__main__":
    main()
