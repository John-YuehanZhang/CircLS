# Table 2 inputs

The circuit each toolchain in Table 2 compiles, one file per program: the QASMBench program
transpiled to Clifford+T by TopoLS (its `docs/benchmark/<name>_tt*.qasm` output; TopoLS is
Apache-2.0, QASMBench BSD-2). For `teleportation_n3` and `qec_en_n5` the transpiled circuit is
the QASMBench file gate for gate; the others differ (tdg written as t, ccx / rx / ry / u3
decomposed), so reproducing Table 2 requires these files, not the QASMBench originals.
The files carry no `creg`/`measure` lines; `gallery/tools/table2_row.py` appends a full Z readout.
