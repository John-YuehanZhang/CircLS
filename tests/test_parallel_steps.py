"""Parallel step execution (shared merge windows): batching mechanics,
distribution equivalence against the serial path, decoder faithfulness,
and full-distance preservation — the complete gate battery, because the
scheduling-vs-decoder lesson says an optimization is only done when the
decoder agrees.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from benchsuite import dj, graph_state_ring

from circls.pipeline import compile_qasm
from circls.tools.reporting import affine_structure, sample_program_bits
from circls.compiler.scheduling import contiguous_batches

_TOPOLS_GHZ = Path(os.environ.get(
    "TOPOLS_DIR",
    str(Path(__file__).resolve().parents[2] / "TopoLS"))) / "docs/benchmark/ghz_16.qasm"
needs_topols = pytest.mark.skipif(
    not _TOPOLS_GHZ.exists(),
    reason="needs a TopoLS checkout (TOPOLS_DIR); see CONTRIBUTING")

MIXED_GHZ = None


def mixed_ghz_qasm():
    """The v2 comparison's ghz_16 mixed-basis fill — the case parallel
    execution exists for (chain of overlapping pairs, fully commuting)."""
    global MIXED_GHZ
    if MIXED_GHZ is None:
        from compare_topols import matching_qasm
        # the ghz_16_k1_f1 port fill, inlined (was read from the result
        # set's manifest before the data moved out of the repo)
        fill = {
            "input_0": "ZXX",
            "input_1": "ZXX",
            "input_2": "ZXX",
            "input_3": "ZXX",
            "input_4": "ZXX",
            "input_5": "ZXX",
            "input_6": "ZXX",
            "input_7": "ZXX",
            "input_8": "ZXX",
            "input_9": "ZXX",
            "input_10": "ZXX",
            "input_11": "ZXX",
            "input_12": "ZXX",
            "input_13": "ZXX",
            "input_14": "ZXX",
            "input_15": "ZXZ",
            "output_15": "XXZ",
            "output_14": "ZXX",
            "output_13": "XXZ",
            "output_12": "XXZ",
            "output_11": "ZXX",
            "output_10": "XXZ",
            "output_9": "ZXX",
            "output_8": "XXZ",
            "output_7": "ZXX",
            "output_6": "ZXX",
            "output_5": "ZXX",
            "output_4": "ZXX",
            "output_3": "XXZ",
            "output_2": "ZXX",
            "output_1": "ZXX",
            "output_0": "ZXX"
        }
        MIXED_GHZ = matching_qasm("ghz_16", fill)
    return MIXED_GHZ


def test_contiguous_batches_rule():
    qs = [frozenset("ab"), frozenset("cd"), frozenset("ae"), frozenset("f")]
    assert contiguous_batches(qs) == [[0, 1], [2, 3]]


def _compile(qasm, parallel, **kw):
    return compile_qasm(qasm, assignment=kw.pop("assignment", "optimized"),
                        liveness=True, keep_patches=set(),
                        parallel_steps=parallel, **kw)


@needs_topols
def test_mixed_ghz_batches_and_wins():
    ser = _compile(mixed_ghz_qasm(), False)
    par = _compile(mixed_ghz_qasm(), True)
    from circls.metrics import experiment_stats
    rs = experiment_stats(ser.experiment, ser.circuit).circuit.measurement_layers
    rp = experiment_stats(par.experiment, par.circuit).circuit.measurement_layers
    assert any(len(b) > 1 for b in par.experiment._step_batches)
    assert rp < rs, f"no round win: {rs} -> {rp}"
    assert par.circuit.num_observables == ser.circuit.num_observables
    det, _ = par.circuit.compile_detector_sampler(seed=0).sample(
        512, separate_observables=True)
    assert not det.any()


@needs_topols
@pytest.mark.parametrize("placement", ["optimized", "row_major"])
def test_distribution_identical_parallel_vs_serial(placement):
    """Was a strict xfail until 2026-08-05: post-closures read the ACTUAL
    (byproduct-frame-contaminated) value, so any execution reordering —
    scheduling or shared windows — silently shifted extracted bits.  The
    seed-closure/anchor-algebra reconstruction in reporting.py restored
    ideal values; row_major additionally needed the batch planner to stop
    batching lazy-reuse members and pre-registered routes (serial-contract
    cell borrowing is unsound under co-activity)."""
    qasm = mixed_ghz_qasm()
    structs = {}
    for tag, par in (("ser", False), ("par", True)):
        cp = _compile(qasm, par, assignment=placement)
        bits = sample_program_bits(cp, shots=512, seed=29)
        structs[tag] = affine_structure(bits)
    assert structs["ser"] == structs["par"]


def test_no_batchable_case_matches_serial():
    # dj's hub chain cannot batch: parallel mode must reproduce the serial
    # circuit exactly (same objective on singleton batches, same order)
    ser = _compile(dj(8), False)
    par = _compile(dj(8), True)
    assert all(len(b) == 1 for b in par.experiment._step_batches)
    assert str(par.circuit) == str(ser.circuit)


@needs_topols
def test_parallel_faithful_and_full_distance():
    from lightstim.noise.config import NoiseConfig
    import pymatching
    noise = NoiseConfig(p_1q=1e-3, p_2q=1e-3, p_meas=1e-3, p_reset=1e-3,
                        p_idle=1e-3)
    cp = _compile(mixed_ghz_qasm(), True, noise=noise)
    assert len(cp.circuit.shortest_graphlike_error()) == 3
    dem = cp.circuit.detector_error_model(decompose_errors=True).flattened()
    m = pymatching.Matching.from_detector_error_model(dem)
    dets, obss = [], []
    for inst in dem:
        if inst.type != "error":
            continue
        dv = np.zeros(dem.num_detectors, dtype=bool)
        ov = np.zeros(dem.num_observables, dtype=bool)
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                dv[t.val] ^= True
            elif t.is_logical_observable_id():
                ov[t.val] ^= True
        dets.append(dv)
        obss.append(ov)
    mis = int((m.decode_batch(np.array(dets)).astype(bool)
               ^ np.array(obss)).any(axis=1).sum())
    assert mis == 0, f"decoder unfaithful under parallel: {mis}/{len(dets)}"
