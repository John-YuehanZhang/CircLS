"""Two-pass matching: the conflicted-edge frame post-pass.

The synthetic DEM reproduces the measured defect topology in
miniature: a fictional weight-1 boundary edge D0 fed by two hyperedge
families of equal probability whose leftover closures differ in
observable content.  Stock PyMatching merges the two variants and
keeps one frame, mis-correcting the other family at first order; the
post-pass re-frames via the companion edges in the solution.
"""
import sys
from pathlib import Path

import numpy as np
import pymatching
import stim

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from twopass_matching import TwoPassMatching, conflict_audit  # noqa: E402

# family A: {D1,D2} companion + leftover D0, no observable
# family B: {D3,D4} companion + leftover D0, carries L0
# plus the real graphlike edges that back the companions
_DEM = stim.DetectorErrorModel("""
    error(0.001) D1 D2 ^ D0
    error(0.001) D3 D4 ^ D0 L0
    error(0.002) D1 D2
    error(0.002) D3 D4
    error(0.002) D1
    error(0.002) D3
""")


def test_stock_matching_miscorrects_one_family():
    mm0 = pymatching.Matching.from_detector_error_model(_DEM)
    a = mm0.decode(np.array([1, 1, 1, 0, 0], dtype=np.uint8))[:1]
    b = mm0.decode(np.array([1, 0, 0, 1, 1], dtype=np.uint8))[:1]
    # merged D0 edge carries ONE frame: exactly one family decodes wrong
    assert int(a[0] != 0) + int(b[0] != 1) == 1


def test_twopass_corrects_both_families():
    tp = TwoPassMatching(_DEM)
    assert tp.num_conflicted_edges == 1
    dets = np.array([[1, 1, 1, 0, 0],     # family A syndrome -> no flip
                     [1, 0, 0, 1, 1]],    # family B syndrome -> L0 flips
                    dtype=np.uint8)
    obs = tp.decode_batch(dets)
    assert obs[0, 0] == 0 and obs[1, 0] == 1


def test_audit_counts_and_clears_the_floor():
    r = conflict_audit(_DEM)
    assert r["conflicted_edges"] == 1
    assert r["stock_mis"] >= 1
    assert r["twopass_mis"] == 0


def test_conflict_free_dem_decodes_like_stock():
    dem = stim.DetectorErrorModel("""
        error(0.001) D0 D1
        error(0.001) D1 D2 L0
        error(0.002) D0
        error(0.002) D2
    """)
    tp = TwoPassMatching(dem)
    assert tp.num_conflicted_edges == 0
    mm0 = pymatching.Matching.from_detector_error_model(dem)
    rng = np.random.default_rng(7)
    dets = (rng.random((64, 3)) < 0.3).astype(np.uint8)
    got = tp.decode_batch(dets)
    ref = np.array([mm0.decode(d)[:1] for d in dets], dtype=np.uint8)
    assert (got == ref).all()


def test_detectorless_component_does_not_crash():
    dem = stim.DetectorErrorModel("""
        error(0.001) L0
        error(0.001) D0 D1
        error(0.002) D0
        error(0.002) D1
    """)
    tp = TwoPassMatching(dem)
    r = conflict_audit(dem)
    # the bare-L0 mechanism is undecodable for every decoder: it must
    # show up as a mis-correction, not as a crash
    assert r["stock_mis"] >= 1 and r["twopass_mis"] >= 1
