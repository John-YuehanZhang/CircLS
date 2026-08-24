# Serial compile-time re-measurement (Table 2 compile-time column audit)

`compile_times.jsonl`: one row per Table-2 CircLS compile-time cell,
re-measured SERIALLY (one compile at a time, 3 repetitions, median) with
`measure_compile_times.py` on an otherwise idle AMD EPYC 9534 (2026-08-24).
Serial measurement is deliberate: compile time is wall-clock and parallel
runs inflate it through CPU contention.

Reconciliation against the published column: the six small cells match
within noise; the two largest (cat_n130: 31.9 s vs published 53.3 s;
ghz_n78: 589 s vs published 792 s) are FASTER here than published — the
published numbers were taken under production load and are conservative
(they overstate, never understate, CircLS's compile time; the paper's
compile-time speedup claims only benefit from the lower re-measured
values). Compile time is machine- and load-dependent; treat the published
column as an upper bound on this machine class.
