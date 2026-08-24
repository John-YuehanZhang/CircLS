"""Deepen the eight under-target d = 11, p = 5e-4 points of the
ext911 batch to at least 100 failures.

The original _depth raised the shot cap only for the dynamic arm, so
five static points (and three dynamic ones at the 4e6 cap) stopped
with 22 to 82 failures (adversarial review 2026-08-20).  This run
re-samples those eight points from the ARCHIVED ext911 circuits
(compilation is deterministic; the two surviving temp dirs hold
bit-identical files, sha256 recorded below) with a 8e6-shot cap,
sharded 16 ways by seed: shard j of point i uses seed
SEED0 + 100*i + j and a fixed 500k-shot budget; counts are summed.
Rows supersede the capped ext911 rows (FILES order in the analysis
scripts).  Output: results/best5/dscaling/points_d11_deep.jsonl.
"""
import hashlib
import json
import multiprocessing as mp
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))
OUT_FILE = (ROOT / "experiments" / "results" / "best5" / "dscaling"
            / "points_d11_deep.jsonl")
CIRC_DIR = Path("/nvme2n1/yuehan_zhang/claude_tmp/ext911_circuits_8ysf2jtf")

POINTS = [  # (name, config) — all at d = 11, p = 5e-4
    ("teleport_4", "static"), ("bv_8", "static"), ("dj_8", "static"),
    ("teleport_8", "static"), ("bv_16", "static"),
    ("teleport_4", "full"), ("bv_8", "full"), ("dj_8", "full"),
]
P = 5e-4
SHARDS = 16
PER_SHARD = 500_000
SEED0 = 911100
WORKERS = 32


def _shard(args):
    path, seed = args
    import numpy as np
    import stim
    import pymatching
    from noise_inject import inject_uniform_noise
    circuit = stim.Circuit(Path(path).read_text())
    noisy = inject_uniform_noise(circuit, P)
    m = pymatching.Matching.from_detector_error_model(
        noisy.detector_error_model(decompose_errors=True))
    sampler = noisy.compile_detector_sampler(seed=seed)
    CHUNK = 50_000       # an un-chunked 500k batch is a ~10 GB bool
    err = 0              # matrix at d = 11; 96 of those thrashed the
    done = 0             # whole machine (measured 2026-08-20)
    while done < PER_SHARD:
        n = min(CHUNK, PER_SHARD - done)
        det, obs = sampler.sample(n, separate_observables=True)
        pred = m.decode_batch(det)
        err += int(np.any(pred != obs, axis=1).sum())
        done += n
    return PER_SHARD, err


def main():
    from provenance import provenance
    if OUT_FILE.exists():
        print(f"skip: {OUT_FILE} exists", flush=True)
        return
    circuits = {}
    for name, config in POINTS:
        f = CIRC_DIR / f"{name}_{config}_d11.stim"
        circuits[(name, config)] = f
        assert f.exists(), f
    prov = {**provenance(False), "experiment": "dscaling-d11-deepen",
            "what": "re-sample the eight under-target d=11 5e-4 points to "
                    ">=100 failures (static arm was cap-starved; review "
                    "2026-08-20); archived ext911 circuits, sharded",
            "protocol": {"p": P, "shards": SHARDS, "per_shard": PER_SHARD,
                         "cap": SHARDS * PER_SHARD, "seed0": SEED0,
                         "decoder": "mwpm",
                         "seed_rule": "SEED0 + 100*point_index + shard"},
            "circuit_sha256": {f"{n}_{c}": hashlib.sha256(
                circuits[(n, c)].read_bytes()).hexdigest()[:16]
                for n, c in POINTS}}
    tasks = []
    for i, (name, config) in enumerate(POINTS):
        for j in range(SHARDS):
            tasks.append((i, name, config,
                          str(circuits[(name, config)]),
                          SEED0 + 100 * i + j))
    with open(OUT_FILE, "w") as f:
        f.write(json.dumps(prov) + "\n")
        f.flush()
        agg = {}
        metas = [(t[0], t[1], t[2]) for t in tasks]
        args = [(t[3], t[4]) for t in tasks]
        with mp.get_context("spawn").Pool(WORKERS) as pool:
            for (i, name, config), (shots, err) in zip(
                    metas, pool.imap(_shard, args)):
                k = (i, name, config)
                s, e = agg.get(k, (0, 0))
                agg[k] = (s + shots, e + err)
                print(f"{name}/{config} shard done: "
                      f"{agg[k][1]}/{agg[k][0]}", flush=True)
        for (i, name, config), (shots, err) in sorted(agg.items()):
            row = {"record": "point", "name": name, "config": config,
                   "d": 11, "p": P, "decoder": "mwpm",
                   "seed0": SEED0 + 100 * i, "shards": SHARDS,
                   "target_errors": 100, "max_shots": SHARDS * PER_SHARD,
                   "shots": shots, "errors": err,
                   "ler": err / shots if shots else None}
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(json.dumps(row), flush=True)
    print("D11 DEEPEN DONE", flush=True)


if __name__ == "__main__":
    main()
