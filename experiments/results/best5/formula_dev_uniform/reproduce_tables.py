#!/usr/bin/env python3
"""Reproduce the two LER-formula-validation tables of the CircLS paper from
the archived measurement data --- self-contained, standard library only.

    python reproduce_tables.py            # reads ./data/*.jsonl

Inputs (in ./data/):
  deviation.jsonl        measured program LER (per benchmark) + the block-budget
                         prediction inputs (volume V1, per-block calibration).
                         Produced by experiments/formula_deviation.py under a
                         uniform circuit-level depolarizing noise pass.
  o3ls_composition.jsonl the O3LS composition-formula prediction per benchmark
                         (per-layer product-sum), produced by
                         experiments/o3ls_composition.py, reusing the same
                         measured LER.

Both files are append-only logs; the last row per (name, d, p) wins, and each
begins with a `provenance` record (git sha, package versions, argv, CPU).

Deviation metric (identical for both tables, paper Eq. text
|p_pred - p_meas| / p_meas):

    dev_i = |pred_i - meas_i| / meas_i            (one number per benchmark)
    table cell = median_i(dev_i) over the qualifying benchmarks, in percent.

Qualifying set:
  tab:formula (O3LS)      status OK, meas > 0, and pred_o3ls > 0.  The last
                          drops the benchmarks whose Clifford gates conjugate
                          every terminal measurement to a single-qubit Pauli,
                          so the circuit has no joint measurement and the
                          per-layer formula predicts exactly zero (listed as
                          excluded cases in the appendix).
  tab:blockbudget         status OK and meas > 0 (joint_errors >= 1).  The block
                          budget V*eps is always positive, so no pred filter.

Expected output on the archived uniform data:
  tab:formula      d3: 66/53/46 (N 22/22/22)   d5: 49/58 (N 18/9)
  tab:blockbudget  d3: 31/22/13 (N 27/27/27)   d5: 21/33 (N 23/14)
"""
import json
import statistics
from collections import OrderedDict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
# the archive stores the jsonl files flat next to this script; a data/
# subdirectory (the layout the docstring's ./data mentions) is honoured
# when present
DATA = _HERE / "data" if (_HERE / "data").is_dir() else _HERE
CELLS = [(3, 2e-4), (3, 5e-4), (3, 1e-3), (5, 5e-4), (5, 1e-3)]


def _load(path, record):
    """Last row per (name, d, p) for the given record type."""
    out = OrderedDict()
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("record") == record:
                out[(r["name"], r["d"], r["p"])] = r
    return out


def _calibrations(path):
    cals = {}
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("record") == "calibration":
                cals[(r["d"], r["p"])] = r          # last wins
    return cals


def _median_dev(devs):
    return statistics.median(sorted(devs)) * 100.0


def table_formula(cases, o3ls):
    rows = []
    for (d, p) in CELLS:
        devs = []
        for k, o in o3ls.items():
            if k[1] != d or k[2] != p or o.get("status") != "OK":
                continue
            c = cases.get(k)
            if c is None or c.get("status") != "OK":
                continue
            meas, pred = c["measured"], o["pred_o3ls"]
            if meas > 0 and pred > 0:                # exclude pred==0 cases
                devs.append(abs(pred - meas) / meas)
        rows.append((d, p, len(devs), _median_dev(devs) if devs else None))
    return rows


def table_blockbudget(cases, cals):
    rows = []
    for (d, p) in CELLS:
        devs = []
        for k, c in cases.items():
            if k[1] != d or k[2] != p or c.get("status") != "OK":
                continue
            if (d, p) not in cals:
                continue
            eps = cals[(d, p)]["eps_block"]
            pred = min(1.0, c["V1"] * eps)           # block budget V*eps, clamped
            meas = c["measured"]
            if meas > 0:                             # joint_errors >= 1
                devs.append(abs(pred - meas) / meas)
        rows.append((d, p, len(devs), _median_dev(devs) if devs else None))
    return rows


def _print(title, rows):
    print(f"\n{title}")
    print(f"  {'d':>2}  {'p':>9}  {'N':>3}  deviation")
    for d, p, n, dev in rows:
        s = f"{dev:.0f}%" if dev is not None else "--"
        print(f"  {d:>2}  {p:>9.0e}  {n:>3}  {s}")


def main():
    dev_path = DATA / "deviation.jsonl"
    o3_path = DATA / "o3ls_composition.jsonl"
    cases = _load(dev_path, "case")
    o3ls = _load(o3_path, "o3ls")
    cals = _calibrations(dev_path)
    prov = next((json.loads(l) for l in open(dev_path)
                 if json.loads(l).get("record") == "provenance"), {})
    print(f"provenance: circls_sha={prov.get('circls_sha', '?')[:9]} "
          f"stim={prov.get('versions', {}).get('stim', '?')} "
          f"cpu={prov.get('cpu', '?')}")
    _print("Table: composition formula (O3LS)  [paper Table tab:formula]",
           table_formula(cases, o3ls))
    _print("Table: block budget  [paper Table tab:blockbudget]",
           table_blockbudget(cases, cals))


if __name__ == "__main__":
    main()
