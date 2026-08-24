# tests/test_tracker_retire.py
import contextlib, io, pytest
import numpy as np
from circls.core.sequential_ppm_ls import SequentialPPMExperiment, PPMStep
from circls.core.routed_multi_patch_ls import PatchSpec, origin_of
from lightstim.noise.config import NoiseConfig
from lightstim.ir.tracker import SyndromeTracker

pytestmark = pytest.mark.smoke
D = 3
NP = NoiseConfig(p_1q=1e-3, p_2q=1e-3, p_meas=1e-3, p_reset=1e-3, p_idle=1e-3)


def _spec(nm, a, b, o):
    return PatchSpec(nm, origin_of(a, b, D, seam=True), D, o)


def _build(exp):
    with contextlib.redirect_stdout(io.StringIO()):
        return exp.build()


def _dist(c):
    c.detector_error_model(decompose_errors=True)
    return len(c.shortest_graphlike_error())


# NOTE: the old same-pair "reproduction" was wrong — same-region corridor reuse
# does NOT crash under up-front registration (coordinate dedup handles it). The
# real failure retire_qubits fixes is measuring a patch out MID-sequence (freeing
# a consumed magic-state / ancilla patch to bound qubit count in large circuits);
# that only becomes reachable once the liveness path is added, so it is driven by
# the liveness feature test (below / later tasks), not a standalone reproduction.
# Pure-tableau unit tests for retire_qubits are added in the next task.


def _mk_tracker(num_qubits, stab_rows, stab_recs, log_rows, log_recs):
    t = SyndromeTracker(num_qubits=num_qubits, expected_num_logicals=len(log_rows))
    if stab_rows:
        t.stabilizers.add_stabilizers(np.array(stab_rows, dtype=np.uint8), stab_recs)
    if log_rows:
        t.logicals.add_stabilizers(np.array(log_rows, dtype=np.uint8), log_recs)
    return t


def test_retire_folds_record_into_surviving_logical():
    # 2 qubits, cols [X0 X1 | Z0 Z1]. S0 = X0 (readout of q0, record m0=7).
    # Surviving logical L_X = X0 X1 (support reaches down to q0). Retire q0.
    t = _mk_tracker(2,
                    stab_rows=[[1, 0, 0, 0]], stab_recs=[[7]],
                    log_rows=[[1, 1, 0, 0]], log_recs=[[]])
    t.retire_qubits([0])
    # L_X becomes X1 with the folded record {7}; the readout stabilizer is dropped.
    assert t.stabilizers.count == 0
    assert np.array_equal(t.logicals.matrix, np.array([[0, 1, 0, 0]], dtype=np.uint8))
    assert set(t.logicals.records[0]) == {7}
    assert t.retired_qubits == {0}
    assert t.num_active_qubits == 1


def test_retire_residual_support_raises():
    # q0 measured in Z (S0 = Z0), but a logical has X0 support -> unclearable -> raise.
    t = _mk_tracker(2,
                    stab_rows=[[0, 0, 1, 0]], stab_recs=[[7]],   # Z0
                    log_rows=[[1, 0, 0, 0]], log_recs=[[]])       # X0 (wrong basis)
    with pytest.raises(ValueError, match="residual"):
        t.retire_qubits([0])


def test_retire_multicolumn_interior_check_pivot():
    # A multi-column pure-S interior check (Z0Z1) listed BEFORE the single-column
    # readouts must not derail retirement (retire uses single-column pivots only).
    t = _mk_tracker(3,
        stab_rows=[[0, 0, 0, 1, 1, 0],   # Z0 Z1 interior check (multi-column, first)
                   [0, 0, 0, 1, 0, 0],   # Z0 readout
                   [0, 0, 0, 0, 1, 0],   # Z1 readout
                   [0, 0, 0, 1, 0, 1]],  # Z0 Z2 surviving stab (Z2 outside S)
        stab_recs=[[3], [1], [2], [4]], log_rows=[], log_recs=[])
    t.retire_qubits([0, 1])
    assert t.stabilizers.count == 1
    assert np.array_equal(t.stabilizers.matrix, np.array([[0, 0, 0, 0, 0, 1]], dtype=np.uint8))
    assert set(t.stabilizers.records[0]) == {1, 4}
    assert t.retired_qubits == {0, 1}


def test_retire_whole_patch_with_interior_checks():
    # retire all 3 qubits: weight-3 interior check listed first + single-column
    # readouts + a logical Z0 that folds to a definite value.
    t = _mk_tracker(3,
        stab_rows=[[0, 0, 0, 1, 1, 1],   # Z0 Z1 Z2 interior (multi-column, first)
                   [0, 0, 0, 1, 0, 0],   # Z0 readout
                   [0, 0, 0, 0, 1, 0],   # Z1 readout
                   [0, 0, 0, 0, 0, 1]],  # Z2 readout
        stab_recs=[[9], [1], [2], [3]],
        log_rows=[[0, 0, 0, 1, 0, 0]], log_recs=[[]])   # logical Z0
    t.retire_qubits([0, 1, 2])
    assert t.stabilizers.count == 0
    assert not t.logicals.matrix.any()
    assert set(t.logicals.records[0]) == {1}
    assert t.num_active_qubits == 0


def test_liveness_chain_two_mid_deaths_full_distance():
    # ZZ chain; Q1 (last-use 0) and Q2 (last-use 1) are retired mid-sequence.
    px = [_spec("Q1", 0, 0, "X_horizontal"), _spec("Q2", 2, 0, "X_horizontal"),
          _spec("Q3", 4, 0, "X_horizontal"), _spec("Q4", 6, 0, "X_horizontal")]
    seq = [PPMStep([("Q1", "Z"), ("Q2", "Z")]),
           PPMStep([("Q2", "Z"), ("Q3", "Z")]),
           PPMStep([("Q3", "Z"), ("Q4", "Z")])]
    st = {n: "X" for n in ("Q1", "Q2", "Q3", "Q4")}
    exp = SequentialPPMExperiment(px, seq, initial_states=st, final_measure_states=st,
                                  rounds=D, rounds_init=1, noise_params=NP, liveness=True, keep_patches=set())
    c = _build(exp)
    assert _dist(c) == D


def test_liveness_ancilla_mid_death_full_distance():
    # A is data (kept); Anc1 is a single-use ancilla retired mid-sequence.
    px = [_spec("Anc1", 0, 0, "X_horizontal"), _spec("A", 2, 0, "X_horizontal"),
          _spec("Anc2", 4, 0, "X_horizontal")]
    seq = [PPMStep([("Anc1", "Z"), ("A", "Z")]),
           PPMStep([("A", "Z"), ("Anc2", "Z")])]
    st = {"Anc1": "X", "A": "X", "Anc2": "X"}
    exp = SequentialPPMExperiment(px, seq, initial_states=st, final_measure_states=st,
                                  rounds=D, rounds_init=1, noise_params=NP, liveness=True, keep_patches=set())
    c = _build(exp)
    assert _dist(c) == D


def test_liveness_with_idle_rounds_full_distance():
    # liveness + idle_rounds together: retired-patch coords in stale SE domains
    # must not desync (se_round_chunk only measures active stabilizers).
    px = [_spec("Q1", 0, 0, "X_horizontal"), _spec("Q2", 2, 0, "X_horizontal"),
          _spec("Q3", 4, 0, "X_horizontal"), _spec("Q4", 6, 0, "X_horizontal")]
    seq = [PPMStep([("Q1", "Z"), ("Q2", "Z")]),
           PPMStep([("Q2", "Z"), ("Q3", "Z")]),
           PPMStep([("Q3", "Z"), ("Q4", "Z")])]
    st = {n: "X" for n in ("Q1", "Q2", "Q3", "Q4")}
    exp = SequentialPPMExperiment(px, seq, initial_states=st, final_measure_states=st,
                                  rounds=D, rounds_init=1, noise_params=NP,
                                  liveness=True, idle_rounds=1, keep_patches=set())
    c = _build(exp)
    assert _dist(c) == D


def test_add_patch_grows_budget_incrementally_after_consume():
    # Pitfall A regression: the define-by-run auto-sync in add_patch must GROW the logical
    # budget by exactly the new patch's logicals, NOT reset it to the stale static total
    # (num_logicals) — else a mid-sequence registration clobbers a prior PPM's consume
    # back up and the next mid-measurement guardrail fails. (Enables deferred allocation.)
    from lightstim.ir.qec_system import QECSystem
    from lightstim.qec_code.surface_code.rotated import RotatedSurfaceCode
    sysm = QECSystem()
    sysm.add_patch(RotatedSurfaceCode(distance=3), name="A", offset=(0, 0))
    sysm.add_patch(RotatedSurfaceCode(distance=3), name="B", offset=(10, 0))
    assert sysm.num_logicals == 2
    t = SyndromeTracker(num_qubits=sysm.num_qubits, expected_num_logicals=sysm.num_logicals)
    sysm.register_tracker(t)
    t.expected_num_logicals = 1                       # simulate a consuming PPM (budget 2->1)
    sysm.add_patch(RotatedSurfaceCode(distance=3), name="C", offset=(20, 0))  # +1 logical
    assert sysm.num_logicals == 3                     # static total still climbs (2->3)
    assert t.expected_num_logicals == 2, (            # budget grew by 1, NOT reset to 3
        f"budget clobbered to stale total: {t.expected_num_logicals}")


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_reuse_retired_patch_space_as_corridor(basis):
    # Q1--Q3--Q2 in a row. PPM0 jointly measures Q1&Q3 (Q3 a target patch); liveness
    # retires Q3; PPM1 jointly measures Q1&Q2 with an EXPLICIT route [(1,0),(2,0),(3,0)]
    # that runs THROUGH Q3's freed cell (2,0). The reuse coupler is auto-detected and
    # registered LAZILY (after Q3 retires), reusing Q3's dormant qubit indices. The
    # circuit must keep full code distance in either init/measure basis.
    px = [_spec("Q1", 0, 0, "X_horizontal"), _spec("Q3", 2, 0, "X_horizontal"),
          _spec("Q2", 4, 0, "X_horizontal")]
    seq = [PPMStep([("Q1", "Z"), ("Q3", "Z")]),
           PPMStep([("Q1", "Z"), ("Q2", "Z")], route=[(1, 0), (2, 0), (3, 0)])]
    st = {n: basis for n in ("Q1", "Q3", "Q2")}
    exp = SequentialPPMExperiment(px, seq, initial_states=st, final_measure_states=st,
                                  rounds=D, rounds_init=1, noise_params=NP, liveness=True, keep_patches=set())
    c = _build(exp)
    # PPM1 auto-detected as a reuse step that must exclude the retired Q3 from routing.
    # A successful full-distance build proves the corridor went THROUGH Q3's cell (2,0):
    # route_and_build would reject that cell while Q3 is still an obstacle.
    assert exp._lazy == {1: {"Q3"}}
    assert _dist(c) == D
    obs = [inst.gate_args_copy()[0] for inst in c.flattened()
           if inst.name == "OBSERVABLE_INCLUDE"]
    assert c.num_observables == len(obs) == len(set(obs)), f"observable collision: {obs}"
    # reuse, not re-allocation: no qubit index is left stale-flagged as retired.
    assert exp.tracker.num_active_qubits == exp.tracker.num_qubits


def test_reuse_lazy_even_without_explicit_route():
    # Same geometry, no explicit route: since the retired-cells-are-free rule
    # (2026-07-28, free cells include the cells a retirement releases), ANY
    # non-adjacent step after a retirement goes lazy — the auto-router may
    # route through Q3's freed cells, so PPM1 is registered lazily with Q3
    # excluded.
    px = [_spec("Q1", 0, 0, "X_horizontal"), _spec("Q3", 2, 0, "X_horizontal"),
          _spec("Q2", 4, 0, "X_horizontal")]
    seq = [PPMStep([("Q1", "Z"), ("Q3", "Z")]),
           PPMStep([("Q1", "Z"), ("Q2", "Z")])]
    st = {n: "X" for n in ("Q1", "Q3", "Q2")}
    exp = SequentialPPMExperiment(px, seq, initial_states=st, final_measure_states=st,
                                  rounds=D, rounds_init=1, liveness=True, keep_patches=set())
    _build(exp)
    assert exp._lazy == {1: {"Q3"}}


def test_liveness_observable_indices_no_collision():
    # A mid-death readout that carries a logical observable (Z init/meas) must NOT
    # reuse observable index 0 already used by a later readout — else stim merges
    # two independent logical readouts into one observable.
    import stim  # noqa
    px = [_spec("Q1", 0, 0, "X_horizontal"), _spec("Q2", 2, 0, "X_horizontal"),
          _spec("Q3", 4, 0, "X_horizontal"), _spec("Q4", 6, 0, "X_horizontal")]
    seq = [PPMStep([("Q1", "Z"), ("Q2", "Z")]),
           PPMStep([("Q2", "Z"), ("Q3", "Z")]),
           PPMStep([("Q3", "Z"), ("Q4", "Z")])]
    st = {n: "Z" for n in ("Q1", "Q2", "Q3", "Q4")}
    exp = SequentialPPMExperiment(px, seq, initial_states=st, final_measure_states=st,
                                  rounds=D, rounds_init=1, noise_params=None, liveness=True, keep_patches=set())
    c = _build(exp)
    obs_idx = [inst.gate_args_copy()[0] for inst in c.flattened()
               if inst.name == "OBSERVABLE_INCLUDE"]
    assert len(obs_idx) == len(set(obs_idx)), f"observable index collision: {obs_idx}"
    assert c.num_observables == len(obs_idx), \
        f"num_observables {c.num_observables} != #OBSERVABLE_INCLUDE {len(obs_idx)}"


def _noiseless_obs_values(exp):
    """Observable values of a noiseless build (all deterministic) — the logical result."""
    c = _build(exp)
    det, obs = c.compile_detector_sampler().sample(2000, separate_observables=True)
    assert bool(np.all(det == det[0])), "noiseless detectors not deterministic"
    assert bool(np.all(obs == obs[0])), "noiseless observables not deterministic"
    return [int(obs[0, i]) for i in range(obs.shape[1])]


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_first_use_init_matches_upfront(basis):
    # first_use_init=True allocates + initialises each patch at its FIRST PPM use instead of
    # up front. On a chain where Q3 is first used only in PPM1, deferral must not change
    # the logical circuit: same code distance and identical (deterministic) observables.
    px = [_spec("Q1", 0, 0, "X_horizontal"), _spec("Q2", 2, 0, "X_horizontal"),
          _spec("Q3", 4, 0, "X_horizontal")]
    seq = [PPMStep([("Q1", "Z"), ("Q2", "Z")]),
           PPMStep([("Q2", "Z"), ("Q3", "Z")])]  # Q3 first used at PPM1 -> deferred
    st = {n: basis for n in ("Q1", "Q2", "Q3")}

    def mk(defer, noise):
        return SequentialPPMExperiment(px, seq, initial_states=st, final_measure_states=st,
                                       rounds=D, rounds_init=1, noise_params=noise,
                                       first_use_init=defer)
    assert _noiseless_obs_values(mk(True, None)) == _noiseless_obs_values(mk(False, None))
    assert _dist(_build(mk(True, NP))) == D


def test_first_use_init_deferred_patch_absent_until_first_use():
    # With first_use_init, a patch first used at PPM1 must NOT be in the system during PPM0.
    px = [_spec("Q1", 0, 0, "X_horizontal"), _spec("Q2", 2, 0, "X_horizontal"),
          _spec("Q3", 4, 0, "X_horizontal")]
    seq = [PPMStep([("Q1", "Z"), ("Q2", "Z")]),
           PPMStep([("Q2", "Z"), ("Q3", "Z")])]
    st = {n: "X" for n in ("Q1", "Q2", "Q3")}
    exp = SequentialPPMExperiment(px, seq, initial_states=st, final_measure_states=st,
                                  rounds=D, rounds_init=1, noise_params=NP, first_use_init=True)
    seen = {}
    orig = SequentialPPMExperiment._apply_ppm_step

    def spy(self, i, step, birth_names=None):
        seen[i] = set(self.system.index_to_owner_map.values())
        return orig(self, i, step, birth_names=birth_names)
    SequentialPPMExperiment._apply_ppm_step = spy
    try:
        _build(exp)
    finally:
        SequentialPPMExperiment._apply_ppm_step = orig
    assert "Q3" not in seen[0], "deferred Q3 was allocated before its first PPM use"
    assert "Q3" in seen[1], "deferred Q3 missing at its first PPM use"


def test_first_use_init_with_retire_and_reuse_bent_routes():
    # The full picture: PPM0 = Z̄_Q1⊗Z̄_Q3 (bent 'red' route, Q1 east seam); liveness
    # retires Q3; Q2 is DEFERRED (allocated only now); PPM1 = X̄_Q1⊗X̄_Q2 (bent 'green'
    # route through Q3's freed cell (2,0), Q1 south seam). Bent perpendicular seams force
    # ZZ-then-XX. Must keep full code distance with deterministic observables.
    px = [_spec("Q1", 0, 2, "X_horizontal"),    # top-left
          _spec("Q3", 2, 0, "X_vertical"),       # bottom-middle
          _spec("Q2", 4, 0, "X_vertical")]       # bottom-right (deferred)
    red = [(1, 2), (2, 2), (2, 1)]
    green = [(0, 1), (0, 0), (1, 0), (2, 0), (3, 0)]  # through Q3's cell (2,0)
    seq = [PPMStep([("Q1", "Z"), ("Q3", "Z")], route=red),
           PPMStep([("Q1", "X"), ("Q2", "X")], route=green)]
    init = {"Q1": "X", "Q3": "Z", "Q2": "X"}
    final = {"Q1": "Z", "Q3": "Z", "Q2": "X"}
    exp = SequentialPPMExperiment(px, seq, initial_states=init, final_measure_states=final,
                                  rounds=D, rounds_init=1, noise_params=NP,
                                  liveness=True, first_use_init=True, keep_patches=set())
    c = _build(exp)
    assert exp._lazy == {1: {"Q3"}}          # PPM1 reuses retired Q3's cell
    assert _dist(c) == D
    exp0 = SequentialPPMExperiment(px, seq, initial_states=init, final_measure_states=final,
                                   rounds=D, rounds_init=1, noise_params=None,
                                   liveness=True, first_use_init=True, keep_patches=set())
    assert len(_noiseless_obs_values(exp0)) == c.num_observables


def test_keep_patches_protects_output_from_retire():
    # CNOT via sequence: C (control), A (ancilla), T (target). C's last PPM use is PPM0, so
    # liveness WOULD retire C after PPM0 — but C is a logical OUTPUT. Declaring it in
    # keep_patches makes liveness skip it, so both the control and target outputs survive:
    # num_observables == 2, all deterministic. (Retiring an output patch would drop its
    # observable — see the observable-loss discussion; keep_patches is how the caller
    # prevents that by naming its outputs.)
    px = [_spec("C", 0, 0, "X_horizontal"), _spec("A", 2, 0, "X_horizontal"),
          _spec("T", 2, 2, "X_horizontal")]
    seq = [PPMStep([("C", "Z"), ("A", "Z")]), PPMStep([("A", "X"), ("T", "X")])]
    exp = SequentialPPMExperiment(px, seq, initial_states={"A": "X", "C": "X", "T": "X"},
                                  final_measure_states={"A": "Z", "C": "X", "T": "X"},
                                  rounds=D, rounds_init=1, noise_params=None,
                                  liveness=True, first_use_init=False,
                                  keep_patches={"C", "T"})
    c = _build(exp)
    det, obs = c.compile_detector_sampler().sample(2000, separate_observables=True)
    assert bool(np.all(det == det[0])), "detectors not deterministic"
    assert bool(np.all(obs == obs[0])), "observables not deterministic"
    assert c.num_observables == 2, \
        f"keep_patches failed to protect output C: num_observables={c.num_observables}"


def test_liveness_requires_keep_patches():
    # Retirement is destructive to un-declared outputs, so liveness=True must be given
    # keep_patches (even if empty). Omitting it is a hard error, not a silent risk.
    px = [_spec("Q1", 0, 0, "X_horizontal"), _spec("Q2", 2, 0, "X_horizontal")]
    seq = [PPMStep([("Q1", "Z"), ("Q2", "Z")])]
    st = {n: "X" for n in ("Q1", "Q2")}
    with pytest.raises(ValueError, match="keep_patches is required when liveness=True"):
        SequentialPPMExperiment(px, seq, initial_states=st, final_measure_states=st,
                                rounds=D, rounds_init=1, noise_params=None, liveness=True)


def _chain4(**kw):
    px = [_spec("Q1", 0, 0, "X_horizontal"), _spec("Q2", 2, 0, "X_horizontal"),
          _spec("Q3", 4, 0, "X_horizontal"), _spec("Q4", 6, 0, "X_horizontal")]
    seq = [PPMStep([("Q1", "Z"), ("Q2", "Z")]),
           PPMStep([("Q2", "Z"), ("Q3", "Z")]),
           PPMStep([("Q3", "Z"), ("Q4", "Z")])]
    st = {n: "X" for n in ("Q1", "Q2", "Q3", "Q4")}
    kw.setdefault("keep_patches", set())
    return SequentialPPMExperiment(px, seq, initial_states=st,
                                   final_measure_states=st, rounds=D,
                                   rounds_init=1, noise_params=NP,
                                   liveness=True, **kw)


def test_lifetime_override_free_layer_is_a_schedule():
    # Q1's derived window is (0, 0); free at layer 1 must retire it one
    # step late — a circuit-visible change, not a silent no-op.
    base = _build(_chain4())
    late = _build(_chain4(lifetime_overrides={"Q1": (0, 1)}))
    assert str(late) != str(base)
    assert _dist(late) == D
    # free at the FINAL layer = survives to the terminal readout = pinning
    kept = _build(_chain4(keep_patches={"Q1"}))
    end = _build(_chain4(lifetime_overrides={"Q1": (0, 2)}))
    assert str(end) == str(kept)


def test_lifetime_override_init_layer_is_a_schedule():
    # Q4's derived window is (2, 2); init at layer 1 must allocate it one
    # step early, and differ from both the default and up-front init.
    base = _build(_chain4())
    mid = _build(_chain4(lifetime_overrides={"Q4": (1, 2)}))
    up = _build(_chain4(lifetime_overrides={"Q4": (0, 2)}))
    assert str(mid) != str(base)
    assert str(up) != str(base)
    assert str(up) != str(mid)
    assert _dist(mid) == D and _dist(up) == D
