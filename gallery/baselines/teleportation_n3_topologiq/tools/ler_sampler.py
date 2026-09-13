"""Shard-parallel logical-error-rate estimation for stim circuits.

The shot budget is cut into fixed-size shards, each sampled with its own
seed (seed + shard index) and decoded in a worker process.  Shards are
tallied strictly in index order until the error target or the shot cap
is reached, so the result depends only on (seed, batch) -- NOT on how
many workers ran it.  Pass ``workers`` to match your machine.

Decoding: MWPM (pymatching) when the detector error model decomposes
into graphlike components; otherwise the hypergraph decoder mwpf, if it
is installed (it is not needed here -- every circuit in this entry
decomposes, and the notebook asks for "mwpm" explicitly).  The decoder is
built once in the parent, surfacing any failure immediately; fork()ed
workers inherit it.
"""
import math
import multiprocessing as mp
import os

import numpy as np

MWPF_TIMEOUT_S = 3.0

_G = {}


def _build_decoder(circ, decoder="auto"):
    """Return count_errors(det, obs) -> int for one sampled batch."""
    if decoder == "mwpf":
        pass                                    # forced hypergraph decoding
    else:
      try:
        import pymatching
        m = pymatching.Matching.from_detector_error_model(
            circ.detector_error_model(decompose_errors=True))

        def count(det, obs):
            pred = m.decode_batch(det).astype(bool)
            return int((pred ^ obs).any(axis=1).sum())

        return count, "mwpm"
      except (ValueError, ImportError):
        if decoder == "mwpm":
            raise
        # non-graphlike DEM -> hypergraph decoder below

    import mwpf
    dem = circ.detector_error_model(flatten_loops=True)
    folded = {}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p_e = inst.args_copy()[0]
        dets, obs_mask = set(), 0
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                dets ^= {t.val}
            elif t.is_logical_observable_id():
                obs_mask ^= (1 << t.val)
        key = frozenset(dets)
        if key in folded:
            p0, obs0 = folded[key]
            folded[key] = (p0 * (1 - p_e) + p_e * (1 - p0),
                           obs0 if p0 >= p_e else obs_mask)
        else:
            folded[key] = (p_e, obs_mask)
    keys = sorted(folded, key=sorted)
    W = 1000.0
    hyperedges, obs_masks = [], []
    for key in keys:
        p_e, obs_mask = folded[key]
        w = max(1, int(round(W * math.log((1 - p_e) / p_e))))
        hyperedges.append(mwpf.HyperEdge(sorted(key), w))
        obs_masks.append(obs_mask)
    init = mwpf.SolverInitializer(dem.num_detectors, hyperedges)
    solver = mwpf.SolverSerialJointSingleHair(init, {"timeout": MWPF_TIMEOUT_S})

    def count(det, obs):
        errs = 0
        for i in range(det.shape[0]):
            defects = np.flatnonzero(det[i])
            if defects.size:
                solver.solve(mwpf.SyndromePattern(
                    defect_vertices=[int(v) for v in defects]))
                sub = solver.subgraph()
                solver.clear()
                pred = 0
                for ei in sub:
                    pred ^= obs_masks[ei]
            else:
                pred = 0
            want = 0
            for b in range(obs.shape[1]):
                if obs[i, b]:
                    want |= (1 << b)
            errs += pred != want
        return errs

    return count, "mwpf"


def _shard(args):
    shard, batch, seed = args
    det, obs = _G["circ"].compile_detector_sampler(seed=seed + shard).sample(
        batch, separate_observables=True)
    return shard, _G["count"](det, obs)


def measure_ler(noisy_circuit, *, target_errors=100, max_shots=2_000_000,
                batch=50_000, workers=None, seed=7, decoder="auto"):
    """Estimate the logical error rate of a noisy stim circuit.

    Returns (ler, errors, shots).  ``workers``: parallel processes
    (default: all CPU cores).
    """
    import stim
    workers = workers or os.cpu_count()
    circ = stim.Circuit(str(noisy_circuit))
    count, decoder = _build_decoder(circ, decoder)  # in the parent: fail loudly
    _G["circ"], _G["count"] = circ, count          # fork()ed workers inherit
    n_shards = max(1, (max_shots + batch - 1) // batch)
    jobs = [(s, batch, seed) for s in range(n_shards)]
    results = {}
    errs = shots = 0
    done_until = 0
    ctx = mp.get_context("fork")
    with ctx.Pool(min(workers, n_shards)) as pool:
        for shard, e in pool.imap_unordered(_shard, jobs):
            results[shard] = e
            while done_until in results:
                errs += results.pop(done_until)
                shots += batch
                done_until += 1
                if errs >= target_errors:
                    pool.terminate()
                    return errs / shots, errs, shots
    return (errs / shots if shots else 0.0), errs, shots
