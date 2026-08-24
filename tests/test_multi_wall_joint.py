"""Multi-#6-wall joints (decline lifted 2026-08-04 after the user's
independence argument was confirmed by re-measurement).

Historical context: `_table_wall_dispatch` used to decline any joint needing
more than one native-#6 wall ("the verified splice algebra covers ONE wall"),
with rotation as the planner's escape hatch.  With swap rotation banned
(not fault tolerant) that escape closed; re-measurement showed clean two-
and mixed-wall joints build, verify, run silent and reach FULL graphlike
AND hypergraph distance.  These tests pin that measurement.

NOTE: multi-wall blocks carry intrinsic hyperedges — LER decoding needs
mwpf, not bare matching (paper-material §2.6).
"""
import contextlib
import io

import pytest

from circls.core.routed_multi_patch_ls import PatchSpec, origin_of
from circls.core.sequential_ppm_ls import PPMStep, SequentialPPMExperiment
from lightstim.noise.config import NoiseConfig

D = 3
NP = NoiseConfig(p_1q=1e-3, p_2q=1e-3, p_meas=1e-3, p_reset=1e-3, p_idle=1e-3)

CASES = {
    # two #6 walls (qA, qC minorities under bus X) + one plain stitch
    "zxz3": (dict(qA=(1, 1), qB=(3, 1), qC=(5, 1)),
             ["X_horizontal"] * 3, ["Z", "X", "Z"]),
    # the original roadtest wall: 2 minorities under EITHER bus
    "xzxz4": (dict(q0=(1, 1), q1=(3, 1), q2=(5, 1), q3=(1, 3)),
              ["X_vertical", "X_horizontal", "X_horizontal", "X_horizontal"],
              ["X", "Z", "X", "Z"]),
}


def _build(name, noisy):
    cells, orients, letters = CASES[name]
    px = [PatchSpec(nm, origin_of(*cells[nm], D, seam=True), D, o)
          for nm, o in zip(cells, orients)]
    step = PPMStep([(nm, L) for nm, L in zip(cells, letters)])
    states = {nm: L for nm, L in zip(cells, letters)}
    exp = SequentialPPMExperiment(px, [step], initial_states=states,
                                  final_measure_states=states,
                                  rounds=D, rounds_init=1,
                                  noise_params=NP if noisy else None)
    with contextlib.redirect_stdout(io.StringIO()):
        return exp.build()


@pytest.mark.smoke
@pytest.mark.parametrize("name", sorted(CASES))
def test_multi_wall_builds_silent_full_graphlike(name):
    clean = _build(name, noisy=False)
    assert clean.num_observables >= 2
    det, _ = clean.compile_detector_sampler(seed=0).sample(
        512, separate_observables=True)
    assert not det.any()
    noisy = _build(name, noisy=True)
    noisy.detector_error_model(decompose_errors=True)
    assert len(noisy.shortest_graphlike_error()) == D


@pytest.mark.parametrize("name", sorted(CASES))
def test_multi_wall_full_hypergraph_distance(name):
    noisy = _build(name, noisy=True)
    errs = noisy.search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=6,
        dont_explore_edges_with_degree_above=6,
        dont_explore_edges_increasing_symptom_degree=False)
    assert len(errs) == D
