"""Rebuild tab:ablation from the full 54-program campaign.
Cost columns: sums at d=3 over programs compiling in EVERY table config,
  Delta% vs full.  LER: geometric-mean of (cfg_ler / full_ler) over all
  (program, p) points sampled_ok in BOTH full and the config.  Fails:
  programs (of the 54 attempted) that cannot compile in the config.
Prints old-vs-new so narrative claims can be re-checked."""
import json, glob, math
from collections import defaultdict
from pathlib import Path

HERE = Path(".")
TABLE_CFGS = ["full", "no_reduce", "no_fui", "no_schedpar",
              "no_place", "no_live", "reselect_only"]
ROWNAME = {"full": "CircLS (full)", "no_reduce": "w/o re-selection",
           "no_fui": "w/o first-use initialization",
           "no_schedpar": "w/o reordering & parallel",
           "no_place": "w/o mapping", "no_live": "w/o last-use freeing",
           "reselect_only": "re-selection only"}

# ---- compile records: last one per (name,cfg) wins (re-runs) ----
best = {}
for l in open("compile.jsonl"):
    r = json.loads(l)
    best[(r["name"], r["config"])] = r
ok = defaultdict(dict)          # cfg -> {name: rec}
fail = defaultdict(set)
attempted = set()
for (name, cfg), r in best.items():
    attempted.add(name)
    if cfg not in TABLE_CFGS:
        continue
    if r.get("status") == "OK":
        ok[cfg][name] = r
    else:
        fail[cfg].add(name)

common = set.intersection(*[set(ok[c]) for c in TABLE_CFGS])
print(f"programs compiling in every table config: {len(common)}")
print(f"attempted programs total: {len(attempted)}")
# config-induced fails: compile in full, fail (or missing) in this config
full_ok = set(ok["full"])
induced_fail = {c: full_ok - set(ok[c]) for c in TABLE_CFGS}

# ---- cost sums over `common` ----
def sums(cfg):
    V = sum(ok[cfg][n]["V1_volume_blocks"] for n in common)
    Q = sum(ok[cfg][n]["V2_qubit_rounds"] for n in common) / 1000.0
    E = sum(ok[cfg][n]["T1_rounds"] for n in common)
    C = sum(ok[cfg][n]["T4_compile_seconds"] for n in common)
    return V, Q, E, C

base = sums("full")

# ---- LER points ----
def load_ler():
    pts = {}   # (name,cfg,p) -> ler (only real sampled_ok)
    for f in glob.glob("points/*.json"):
        r = json.load(open(f))
        tag = r.get("tag")
        if tag is None:
            continue
        # tag = name__cfg   (cfg may contain no underscores? configs are fixed)
        # split on last "__"
        if "__" not in tag:
            continue
        name, cfg = tag.rsplit("__", 1)
        if r.get("dagger") or r.get("ler_excluded") or r.get("needs_mwpf"):
            continue
        if r.get("ler") is None:
            continue
        pts[(name, cfg, r["p"])] = r["ler"]
    return pts

pts = load_ler()

def ler_geomean(cfg):
    logs = []
    n = 0
    for (name, c, p), ler in pts.items():
        if c != cfg:
            continue
        full_ler = pts.get((name, "full", p))
        if full_ler is None or full_ler <= 0 or ler <= 0:
            continue
        logs.append(math.log(ler / full_ler))
        n += 1
    if not logs:
        return None, 0
    gm = math.exp(sum(logs) / len(logs))
    return gm, n

# ---- emit ----
print("\ncfg               | volume  d%    | qcyc(k) d%    | exec  d%    | "
      "comp(s) d%    | LER d%   (npts) | fails")
rows = {}
for cfg in TABLE_CFGS:
    V, Q, E, C = sums(cfg)
    dV = (V/base[0]-1)*100; dQ = (Q/base[1]-1)*100
    dE = (E/base[2]-1)*100; dC = (C/base[3]-1)*100
    gm, npts = ler_geomean(cfg)
    dL = (gm-1)*100 if gm is not None else None
    rows[cfg] = dict(V=V, Q=Q, E=E, C=C, dV=dV, dQ=dQ, dE=dE, dC=dC,
                     dL=dL, npts=npts, fails=len(induced_fail[cfg]))
    dLs = f"{dL:+.1f}" if dL is not None else "  -- "
    print(f"{cfg:17s} | {V:6.0f} {dV:+5.1f} | {Q:6.0f} {dQ:+5.1f} | "
          f"{E:5.0f} {dE:+5.1f} | {C:6.0f} {dC:+6.1f} | {dLs:>6s} ({npts:2d}) | "
          f"{len(induced_fail[cfg])}")

# ---- LaTeX rows ----
print("\n---- LaTeX table body ----")
def fnum(x): return f"{x:.0f}"
for cfg in TABLE_CFGS:
    r = rows[cfg]
    nm = ROWNAME[cfg]
    if cfg == "full":
        print(f"    \\sys (full) & {fnum(r['V'])} & & {fnum(r['Q'])} & & "
              f"{fnum(r['E'])} & & {fnum(r['C'])} & & & {r['fails']} \\\\")
    else:
        dL = f"${r['dL']:+.1f}$" if r['dL'] is not None else ""
        print(f"    {nm} & {fnum(r['V'])} & ${r['dV']:+.1f}$ & {fnum(r['Q'])} "
              f"& ${r['dQ']:+.1f}$ & {fnum(r['E'])} & ${r['dE']:+.1f}$ & "
              f"{fnum(r['C'])} & ${r['dC']:+.1f}$ & {dL} & {r['fails']} \\\\")
