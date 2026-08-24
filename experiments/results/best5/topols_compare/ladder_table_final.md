# A1(2) scale ladder — FINAL (system-level LER, sha 51fc89ca1)

Baseline TopoLS+tqec: capability_x for every row below (spatial-Hadamard NotImplementedError, see topols_circuits manifest).
LER = circuit + decoder as one system (audit counts stay in the jsonl, not tabulated).

| case | d | qubits | rounds | qubit-rounds | LER p=1e-3 | LER p=5e-4 |
|---|---|---|---|---|---|---|
| bv_32 | 3 | 1576 | 93 | 35952 | 0.9510 | 0.5509 |
| bv_32 | 5 | 3936 | 127 | 145322 | 0.6171 | 0.1193 |
| bv_64 | 3 | 3408 | 181 | 126587 | 1.0000 | 0.9459 |
| bv_64 | 5 | 8474 | 247 | 513807 | 0.9689 | 0.3821 |
| bv_100 | 3 | 5382 | 280 | 291931 | skipped (saturated) | skipped (saturated) |
| bv_100 | 5 | 13446 | 382 | 1187569 | skipped (saturated) | skipped (saturated) |
| dj_32 | 3 | 2819 | 175 | 44913 | 0.9675 | 0.5989 |
| dj_32 | 5 | 6843 | 239 | 177641 | 0.6211 | 0.1172 |
| dj_64 | 3 | 6032 | 351 | 146695 | 1.0000 | 0.9532 |
| dj_64 | 5 | 14704 | 479 | 580993 | 0.9681 | 0.3778 |
| dj_100 | 3 | 9293 | 549 | 323096 | skipped (saturated) | skipped (saturated) |
| dj_100 | 5 | 22624 | 749 | 1280886 | skipped (saturated) | skipped (saturated) |
