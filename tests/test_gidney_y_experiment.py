"""Task-3 integration: initial_states='Y' = Gidney in-place FT birth inside
SequentialPPMExperiment (degenerate patch -> mixed reset -> reversed rounds ->
tick-zipped transition via apply_relay_chunk -> declare_logical -> forward
'gidney' seam round), run as a prologue with the active stabilizer set masked
to the Y patches.  Acceptance: noiseless silence, closed census (build does
not throw), and full fault distance d with circuit-level noise everywhere —
the same bar the LightStim harness passed."""
import contextlib
import io

import pytest

from circls.core.sequential_ppm_ls import PPMStep, SequentialPPMExperiment
from circls.core.routed_multi_patch_ls import PatchSpec, origin_of
from lightstim.noise.config import NoiseConfig

D = 3
NP = NoiseConfig(p_1q=1e-3, p_2q=1e-3, p_meas=1e-3, p_reset=1e-3, p_idle=1e-3)


def _spec(nm, a, b, o="X_vertical"):
    return PatchSpec(nm, origin_of(a, b, D, seam=True), D, o)


def _build(exp):
    with contextlib.redirect_stdout(io.StringIO()):
        return exp.build()


def _gadget_exp(**kw):
    """One pi/4-gadget shape: data q0 + |Y> ancilla y0, joint Z-Z, X readout."""
    px = [_spec("q0", 0, 0), _spec("y0", 2, 0)]
    seq = [PPMStep([("q0", "Z"), ("y0", "Z")])]
    return SequentialPPMExperiment(
        px, seq, initial_states={"q0": "Z", "y0": "Y"},
        final_measure_states={"q0": "Z", "y0": "X"},
        rounds=D, rounds_init=1, **kw)


def test_y_birth_single_gadget_noiseless_silent():
    c = _build(_gadget_exp())
    det, obs = c.compile_detector_sampler(seed=0).sample(
        1024, separate_observables=True)
    assert not det.any(), "noiseless detectors must be silent"
    assert not obs.any(), "noiseless observables must be silent"
    assert c.num_observables == 1
    assert c.num_detectors > 0


def test_y_birth_single_gadget_full_distance():
    c = _build(_gadget_exp(noise_params=NP))
    c.detector_error_model(decompose_errors=True)   # detectors deterministic
    assert len(c.shortest_graphlike_error()) == D


def test_two_y_ancillas_zipped_transition():
    """Two |Y> patches born in ONE prologue (tick-zipped transition round),
    each consumed by its own gadget PPM."""
    px = [_spec("q0", 0, 0), _spec("y0", 2, 0), _spec("y1", 4, 0)]
    seq = [PPMStep([("q0", "Z"), ("y0", "Z")]),
           PPMStep([("q0", "Z"), ("y1", "Z")])]
    exp = SequentialPPMExperiment(
        px, seq, initial_states={"q0": "Z", "y0": "Y", "y1": "Y"},
        final_measure_states={"q0": "Z", "y0": "X", "y1": "X"},
        rounds=D, rounds_init=1)
    c = _build(exp)
    det, obs = c.compile_detector_sampler(seed=1).sample(
        1024, separate_observables=True)
    assert not det.any() and not obs.any()


def test_two_y_ancillas_full_distance():
    px = [_spec("q0", 0, 0), _spec("y0", 2, 0), _spec("y1", 4, 0)]
    seq = [PPMStep([("q0", "Z"), ("y0", "Z")]),
           PPMStep([("q0", "Z"), ("y1", "Z")])]
    exp = SequentialPPMExperiment(
        px, seq, initial_states={"q0": "Z", "y0": "Y", "y1": "Y"},
        final_measure_states={"q0": "Z", "y0": "X", "y1": "X"},
        rounds=D, rounds_init=1, noise_params=NP)
    c = _build(exp)
    c.detector_error_model(decompose_errors=True)
    assert len(c.shortest_graphlike_error()) == D


def test_y_final_state_still_rejected():
    px = [_spec("q0", 0, 0), _spec("y0", 2, 0)]
    with pytest.raises(ValueError, match="final_measure_states"):
        SequentialPPMExperiment(
            px, [PPMStep([("q0", "Z"), ("y0", "Z")])],
            initial_states={"q0": "Z", "y0": "Y"},
            final_measure_states={"q0": "Z", "y0": "Y"})


def test_y_patch_requires_x_vertical():
    px = [_spec("q0", 0, 0), _spec("y0", 2, 0, "X_horizontal")]
    exp = SequentialPPMExperiment(
        px, [PPMStep([("q0", "Z"), ("y0", "Z")])],
        initial_states={"q0": "Z", "y0": "Y"},
        final_measure_states={"q0": "Z", "y0": "X"})
    with pytest.raises(ValueError, match="X_vertical"):
        _build(exp)


# ------------------------------------------------- deferred birth (C1 redesign)
def test_c1_corridor_borrows_prebirth_y_cell():
    """The C1 reproduction layout: y0 sits BETWEEN the two step-0 targets and
    is first used at step 1.  Its cell is a routing obstacle from step
    fu-2 on (bus ban before the reset), the birth rides step 0's merged
    rounds, and the whole thing must build, stay silent and hold distance."""
    import stim as _stim
    px = [_spec("q0", 0, 0), _spec("y0", 1, 0), _spec("q1", 2, 0)]
    # step 0 must route AROUND y0's banned cell; step 1 is the adjacent
    # X-X gadget joint (vertical seam allows only the parallel X letters)
    seq = [PPMStep([("q0", "Z"), ("q1", "Z")]),
           PPMStep([("q0", "X"), ("y0", "X")])]
    states = {"q0": "Z", "q1": "Z", "y0": "Y"}
    finals = {"q0": "Z", "q1": "Z", "y0": "X"}
    exp = SequentialPPMExperiment(px, seq, initial_states=states,
                                  final_measure_states=finals,
                                  rounds=D, rounds_init=1)
    c = _build(exp)
    det, obs = c.compile_detector_sampler(seed=3).sample(
        1024, separate_observables=True)
    assert not det.any() and not obs.any()
    expn = SequentialPPMExperiment(px, seq, initial_states=states,
                                   final_measure_states=finals,
                                   rounds=D, rounds_init=1, noise_params=NP)
    cn = _build(expn)
    # distance on the deterministic step-0 closure (pre xor post records)
    pre, post = expn.step_joint_records_pre[0], expn.step_joint_records_post[0]
    assert pre is not None and post is not None
    recs = sorted(set(pre) ^ set(post))
    cn.append("OBSERVABLE_INCLUDE",
              [_stim.target_rec(r - cn.num_measurements) for r in recs],
              [cn.num_observables])
    cn.detector_error_model(decompose_errors=True)
    assert len(cn.shortest_graphlike_error()) == D


def test_absent_rule_for_y_patches():
    """Y cell is free ground only up to step fu-3; obstacle from fu-2 on."""
    px = [_spec("q0", 0, 0), _spec("q1", 2, 0), _spec("y0", 4, 0)]
    seq = [PPMStep([("q0", "Z"), ("q1", "Z")]),
           PPMStep([("q0", "Z"), ("q1", "Z")]),
           PPMStep([("q0", "Z"), ("q1", "Z")]),
           PPMStep([("q0", "Z"), ("y0", "Z")])]     # fu(y0) = 3
    exp = SequentialPPMExperiment(px, seq,
                                  initial_states={"q0": "Z", "q1": "Z",
                                                  "y0": "Y"},
                                  final_measure_states={"q0": "Z", "q1": "Z",
                                                        "y0": "X"},
                                  rounds=D, rounds_init=1)
    omap = {s.name: s.orientation for s in px}
    names = lambda i: {s.name for s in exp._specs_for_step(i, omap)}
    assert "y0" not in names(0)          # i < fu-2: borrowable free ground
    assert "y0" in names(1)              # i = fu-2: bus ban starts (reset
    assert "y0" in names(2)              # is one round after step fu-2 ends)
    assert "y0" in names(3)


def test_y_ancilla_liveness_retire():
    """liveness: the |Y> ancilla is measured out (its gadget X readout) and
    retired right after its last use; census stays closed."""
    px = [_spec("q0", 0, 0), _spec("y0", 2, 0), _spec("q1", 0, 2)]
    seq = [PPMStep([("q0", "Z"), ("y0", "Z")]),
           PPMStep([("q0", "Z"), ("q1", "Z")])]
    exp = SequentialPPMExperiment(
        px, seq, initial_states={"q0": "Z", "q1": "Z", "y0": "Y"},
        final_measure_states={"q0": "Z", "q1": "Z", "y0": "X"},
        rounds=D, rounds_init=1, liveness=True, keep_patches={"q0", "q1"})
    c = _build(exp)
    det, obs = c.compile_detector_sampler(seed=5).sample(
        1024, separate_observables=True)
    assert not det.any() and not obs.any()
    assert exp.tracker.expected_num_logicals == exp.tracker.logicals.count


# ---------------------------------------------------- C2: content-keyed cache
def test_route_cache_keys_on_step_content():
    """C2 regression: the route cache must key on step CONTENT (the route is
    a pure function of specs + step content), not on id() of throwaway
    replace() copies.  Repeated identical steps share ONE cache entry."""
    px = [_spec("A", 0, 0), _spec("B", 2, 0), _spec("C", 4, 0)]
    seq = ([PPMStep([("A", "Z"), ("B", "Z")]),
            PPMStep([("B", "Z"), ("C", "Z")])] * 6)   # 12 alternating steps
    exp = SequentialPPMExperiment(
        px, seq, initial_states={"A": "Z", "B": "Z", "C": "Z"},
        final_measure_states={"A": "Z", "B": "Z", "C": "Z"},
        rounds=D, rounds_init=1)
    c = _build(exp)
    det, obs = c.compile_detector_sampler(seed=9).sample(
        512, separate_observables=True)
    assert not det.any()
    # two distinct step contents -> at most a couple of cache entries, and
    # every key must be hashable content (no integers-that-were-addresses)
    assert len(exp._rr_cache) <= 4, len(exp._rr_cache)
    for key in exp._rr_cache:
        assert isinstance(key[0], tuple), key


def test_c2_double_build_shared_steps_pipeline():
    """The deterministic C2 reproduction: noiseless build + sampling + noisy
    build sharing the same PPMStep objects — pre-fix, heap-recycled replace()
    ids handed a step a stale route and the appended-observable DEM failed to
    decompose."""
    pytest.importorskip("nwqec")
    from circls.interop.nwqec.frontend import load_clifford_mpauli
    from circls.interop.ir.pauli_algebra import y_free_sweep
    from circls.interop.ir.gosc_gadgets import (expand_gadgets,
                                             to_experiment_inputs,
                                             append_program_observables)
    QASM = ('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\n'
            'cx q[0],q[1];\nh q[0];\nsx q[0];\n')
    prog = expand_gadgets(y_free_sweep(load_clifford_mpauli(QASM)))
    specs, steps, init, final, _ = to_experiment_inputs(prog, distance=3)
    exp = SequentialPPMExperiment(specs, steps, initial_states=init,
                                  final_measure_states=final,
                                  rounds=3, rounds_init=1)
    c = _build(exp)
    append_program_observables(c, exp, prog)
    c.compile_detector_sampler(seed=0).sample(1024, separate_observables=True)
    expn = SequentialPPMExperiment(specs, steps, initial_states=init,
                                   final_measure_states=final,
                                   rounds=3, rounds_init=1, noise_params=NP)
    cn = _build(expn)
    append_program_observables(cn, expn, prog)
    cn.detector_error_model(decompose_errors=True)
    assert len(cn.shortest_graphlike_error()) == D


# ------------------------------------------------------ M3: wall-step hooks
def test_wall_step_records_extracted():
    """M3: a wall-realised PPM step (adjacent pair, stretched-stabilizer
    wall, no corridor) must fill step_joint_records_pre/post like the routed
    path — post is extracted BEFORE the split (the kf rows leave the live
    span at deactivation).  Healthy row-2 config borrowed from
    test_rule_table_dispatch."""
    import stim as _stim
    from lightstim.noise.config import NoiseConfig
    px = [PatchSpec("A", origin_of(0, 0, D, seam=True), D, "X_horizontal"),
          PatchSpec("B", origin_of(0, 1, D, seam=True), D, "X_vertical")]
    states = {"A": "X", "B": "X"}
    exp = SequentialPPMExperiment(
        px, [PPMStep([("A", "X"), ("B", "X")])], initial_states=states,
        final_measure_states=states, rounds=D, rounds_init=1,
        colour_swapped={"B"})
    c = _build(exp)
    assert 0 in exp._walls                    # really the wall path
    pre = exp.step_joint_records_pre.get(0)
    post = exp.step_joint_records_post.get(0)
    assert post is not None, "wall step post-records missing (M3)"
    assert pre is not None, "X(x)X on |++> is deterministic: pre must solve"
    recs = sorted(set(pre) ^ set(post))
    c.append("OBSERVABLE_INCLUDE",
             [_stim.target_rec(r - c.num_measurements) for r in recs],
             [c.num_observables])
    det, obs = c.compile_detector_sampler(seed=11).sample(
        1024, separate_observables=True)
    assert not det.any()
    for j in range(c.num_observables):
        assert (obs[:, j] == obs[0, j]).all()
    # noisy: the closure observable holds full distance through the wall
    expn = SequentialPPMExperiment(
        px, [PPMStep([("A", "X"), ("B", "X")])], initial_states=states,
        final_measure_states=states, rounds=D, rounds_init=1,
        colour_swapped={"B"},
        noise_params=NoiseConfig(p_1q=1e-3, p_2q=1e-3, p_meas=1e-3,
                                 p_reset=1e-3, p_idle=1e-3))
    cn = _build(expn)
    pren, postn = expn.step_joint_records_pre[0], expn.step_joint_records_post[0]
    recsn = sorted(set(pren) ^ set(postn))
    cn.append("OBSERVABLE_INCLUDE",
              [_stim.target_rec(r - cn.num_measurements) for r in recsn],
              [cn.num_observables])
    cn.detector_error_model(decompose_errors=True)
    assert len(cn.shortest_graphlike_error()) == D


# ------------------------------------------- M4: retirement on wall/snake
def test_wall_step_last_use_retires_under_liveness():
    """M4: a consumed patch whose LAST use is a wall step must be measured
    out + retired like on the routed path (the wall branch used to
    `continue` past the retire block while the router's absent-rule assumed
    the cell was freed)."""
    px = [PatchSpec("A", origin_of(0, 0, D, seam=True), D, "X_horizontal"),
          PatchSpec("B", origin_of(0, 1, D, seam=True), D, "X_vertical"),
          PatchSpec("C", origin_of(4, 0, D, seam=True), D, "X_vertical"),
          PatchSpec("E", origin_of(6, 0, D, seam=True), D, "X_vertical")]
    seq = [PPMStep([("A", "X"), ("B", "X")]),          # wall (row 2)
           PPMStep([("C", "Z"), ("E", "Z")])]          # routed, final layer
    states = {"A": "X", "B": "X", "C": "Z", "E": "Z"}
    exp = SequentialPPMExperiment(
        px, seq, initial_states=states, final_measure_states=states,
        rounds=D, rounds_init=1, liveness=True, keep_patches={"C", "E"},
        colour_swapped={"B"})
    c = _build(exp)
    assert 0 in exp._walls
    owner = exp.system.index_to_owner_map
    retired = exp.tracker.retired_qubits
    for nm in ("A", "B"):
        dq = [q for q in exp.system.data_indices if owner.get(q) == nm]
        assert dq and all(q in retired for q in dq), \
            f"{nm} not retired after its last (wall) use"
    det, obs = c.compile_detector_sampler(seed=7).sample(
        1024, separate_observables=True)
    assert not det.any() and not obs.any()
    assert exp.tracker.expected_num_logicals == exp.tracker.logicals.count


# --------------------------------------------------- M5: build() re-run reset
def test_second_build_identical_after_rotation():
    """M5: a second build() on the SAME experiment object must produce a
    byte-identical circuit — _orient used to keep the post-rotation
    orientations of run 1, feeding wrong hook tables to run 2 (distance
    silently halves).  Config = the rotation-triggering corner case from
    test_wall_chirality."""
    px = [PatchSpec("q1", origin_of(0, 0, D, seam=True), D, "X_vertical"),
          PatchSpec("q2", origin_of(0, 2, D, seam=True), D, "X_vertical"),
          PatchSpec("q3", origin_of(4, 2, D, seam=True), D, "X_vertical"),
          PatchSpec("q5", origin_of(6, 2, D, seam=True), D, "X_vertical")]
    init = {p.name: "Z" for p in px}
    exp = SequentialPPMExperiment(
        px, [PPMStep([("q1", "Z"), ("q3", "X"), ("q5", "X")])],
        initial_states=init, final_measure_states=init,
        rounds=D, rounds_init=2, auto_rotate=True, rotation_kind="litinski",
        rotate_saving_threshold=1)
    c1 = _build(exp)
    log1 = list(exp.rotation_log)
    c2 = _build(exp)
    assert exp.rotation_log == log1, "rotation log must not accumulate"
    assert str(c1) == str(c2), "second build differs from the first"
