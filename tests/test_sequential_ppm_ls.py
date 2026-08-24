import contextlib
import io

import pytest
from circls.core.sequential_ppm_ls import PPMStep, compute_lifetimes, SequentialPPMExperiment
from circls.core.routed_multi_patch_ls import PatchSpec, origin_of
from circls.core.multi_patch_coupler import BentLayoutError
from lightstim.noise.config import NoiseConfig

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from conftest import assert_valid_circuit, assert_noiseless, assert_dem_valid

pytestmark = pytest.mark.smoke

D = 3

NP = NoiseConfig(p_1q=1e-3, p_2q=1e-3, p_meas=1e-3, p_reset=1e-3, p_idle=1e-3)


def _build(exp):
    with contextlib.redirect_stdout(io.StringIO()):
        return exp.build()


def _dist(c):
    c.detector_error_model(decompose_errors=True)   # detectors deterministic
    return len(c.shortest_graphlike_error())


def _spec(nm, a, b, o):
    return PatchSpec(nm, origin_of(a, b, D, seam=True), D, o)


def _exp(patches, seq, init, meas, **kw):
    return SequentialPPMExperiment(
        patches, seq, initial_states=init, final_measure_states=meas,
        rounds=D, rounds_init=1, **kw)


def test_non_xz_pauli_raises():
    px = [_spec("Q1", 0, 0, "X_horizontal"), _spec("Q2", 2, 0, "X_horizontal")]
    with pytest.raises(ValueError, match="X.*Z|Pauli"):
        SequentialPPMExperiment(px, [PPMStep([("Q1", "Y"), ("Q2", "Z")])],
                                initial_states={"Q1": "X", "Q2": "Z"},
                                final_measure_states={"Q1": "X", "Q2": "Z"})


def test_y_initial_state_accepted():
    # initial 'Y' = the fault-tolerant in-place Gidney birth (arXiv:2302.07395),
    # wired in 2026-08-03 — construction must NOT raise any more (the old
    # corner-injection rejection is history). Full behaviour is covered by
    # tests/test_gidney_y_experiment.py.
    px = [_spec("Q1", 0, 0, "X_horizontal"), _spec("Q2", 2, 0, "X_vertical")]
    SequentialPPMExperiment(px, [PPMStep([("Q1", "Z"), ("Q2", "Z")])],
                            initial_states={"Q1": "Z", "Q2": "Y"},
                            final_measure_states={"Q1": "Z", "Q2": "X"})


def test_bad_letter_initial_state_raises():
    px = [_spec("Q1", 0, 0, "X_horizontal"), _spec("Q2", 2, 0, "X_horizontal")]
    with pytest.raises(ValueError, match="initial_states"):
        SequentialPPMExperiment(px, [PPMStep([("Q1", "Z"), ("Q2", "Z")])],
                                initial_states={"Q1": "W", "Q2": "Z"},
                                final_measure_states={"Q1": "Z", "Q2": "Z"})


def test_y_final_state_raises():
    px = [_spec("Q1", 0, 0, "X_horizontal"), _spec("Q2", 2, 0, "X_horizontal")]
    with pytest.raises(ValueError, match="final_measure_states"):
        SequentialPPMExperiment(px, [PPMStep([("Q1", "Z"), ("Q2", "Z")])],
                                initial_states={"Q1": "Z", "Q2": "Z"},
                                final_measure_states={"Q1": "Y", "Q2": "Z"})


def test_missing_state_entry_raises():
    px = [_spec("Q1", 0, 0, "X_horizontal"), _spec("Q2", 2, 0, "X_horizontal")]
    with pytest.raises(ValueError):
        SequentialPPMExperiment(px, [PPMStep([("Q1", "Z"), ("Q2", "Z")])],
                                initial_states={"Q1": "X"},   # Q2 missing
                                final_measure_states={"Q1": "X", "Q2": "Z"})


def test_compute_lifetimes():
    seq = [
        PPMStep([("Q1", "Z"), ("Q2", "Z")]),   # layer 0
        PPMStep([("Q2", "X"), ("Q3", "X")]),   # layer 1
        PPMStep([("Q1", "X"), ("Q3", "X")]),   # layer 2
    ]
    lt = compute_lifetimes(seq)
    assert lt == {"Q1": (0, 2), "Q2": (0, 1), "Q3": (1, 2)}


def test_registration_all_majority_is_textbook():
    # CNOT-style: pure-Z then pure-X, no minority anywhere -> all textbook
    px = [_spec("C", 0, 0, "X_horizontal"),
          _spec("T", 2, 0, "X_horizontal")]
    seq = [PPMStep([("C", "Z"), ("T", "Z")]), PPMStep([("C", "X"), ("T", "X")])]
    exp = _exp(px, seq, {"C": "Z", "T": "Z"}, {"C": "Z", "T": "X"})
    reg = exp._resolve_registration()
    assert reg["C"][0] is False and reg["T"][0] is False


def test_registration_is_declared():
    # STRICT BIRTHS (design decision 2026-07-31): registration is EXACTLY the
    # declared input — no minority-role inference, no birth conjugation.
    px = [_spec("Q1", 0, 0, "X_vertical"),
          _spec("Q2", 2, 0, "X_horizontal")]
    seq = [PPMStep([("Q1", "X"), ("Q2", "Z")])]
    exp = _exp(px, seq, {"Q1": "X", "Q2": "Z"}, {"Q1": "X", "Q2": "Z"})
    reg = exp._resolve_registration()
    assert reg["Q1"] == (False, "X_vertical")
    assert reg["Q2"] == (False, "X_horizontal")


def test_role_switch_without_auto_rotate_wall_serves():
    # Q measures X in PPM0 and Z in PPM1.  Under STRICT BIRTHS the old
    # role-switch inference is gone; serving the letter mismatch without a
    # rotation needs a native-#6 wall.  Since the chirality-law envelope
    # (2026-08-02: both orientations, all sides, handedness-gated exits)
    # that wall hosts, so the sequence builds with ZERO rotations even
    # without auto_rotate.  Registration stays strictly the declared input.
    px = [_spec("Q", 0, 0, "X_vertical"),
          _spec("A", 2, 0, "X_horizontal"),
          _spec("B", 4, 0, "X_horizontal")]
    seq = [PPMStep([("Q", "X"), ("A", "Z")]),
           PPMStep([("Q", "Z"), ("B", "X")])]
    exp = _exp(px, seq, {"Q": "X", "A": "Z", "B": "X"},
               {"Q": "Z", "A": "Z", "B": "X"})
    reg = exp._resolve_registration()
    assert all(v == (False, s.orientation)
               for s, v in ((s, reg[s.name]) for s in px))
    c = _build(exp)
    assert exp.rotation_log == []
    det, obs = c.compile_detector_sampler(seed=0).sample(
        1024, separate_observables=True)
    assert not det.any() and not obs.any()


def test_boxed_target_repaired_by_rotation_fallback():
    # T measures Z but BOTH its parallel-law-legal faces (N/S) are occupied
    # by anchor patches — no corridor and no wall can host it in the birth
    # orientation.  Design decision 2026-08-05 (supersedes the earlier
    # no-fallback reading): even with the rotation PLANNER off, a blocked
    # patch gets a feasibility-only litinski rotation.  The 2026-07-31 iron
    # rule is untouched: it forbids retrying another CORRIDOR after a
    # failed construction; the fallback fires when selection is infeasible
    # wholesale, rotates, then selects fresh.
    px = [_spec("T", 2, 2, "X_vertical"),
          _spec("N_", 2, 1, "X_vertical"),
          _spec("S_", 2, 3, "X_vertical"),
          _spec("P", 6, 2, "X_vertical")]
    seq = [PPMStep([("T", "Z"), ("P", "X")])]
    init = {"T": "Z", "N_": "Z", "S_": "Z", "P": "Z"}
    exp = _exp(px, seq, init, init)
    c = _build(exp)
    assert ("T" in [nm for _, nm, kind in exp.rotation_log
                    if kind == "litinski"]), \
        "boxed patch was not repaired by the rotation fallback"
    assert exp.auto_rotate is False
    det, _ = c.compile_detector_sampler(seed=0).sample(
        1024, separate_observables=True)
    assert not det.any()


def test_pair_zz_then_xx_strict_births():
    # M(Z1Z2) then M(X1X2) on the same pair — the tracker regression pair
    # (2026-07-31): (a) the second joint anticommutes with the surviving
    # Z-string row but every lobe's Case-A pivot lands on a stabilizer
    # (stab-shadowed kill) — the dependent logical row must decrement the
    # budget; (b) a corridor split may remove a logical row (restore-credit
    # promotions live on corridor qubits) — the corridor-readout branch
    # must charge it.  Census alarms fire without either.
    px = [_spec("Q1", 0, 0, "X_vertical"), _spec("Q2", 2, 0, "X_vertical")]
    seq = [PPMStep([("Q1", "Z"), ("Q2", "Z")]),
           PPMStep([("Q1", "X"), ("Q2", "X")])]
    init = {"Q1": "Z", "Q2": "Z"}
    exp = _exp(px, seq, init, init, auto_rotate=True, rotation_kind='auto',
               rotate_saving_threshold=1)
    c = _build(exp)
    assert len(exp.rotation_log) == 4      # r90 pair in, r90 pair back

    det, obs = c.compile_detector_sampler(seed=0).sample(
        1024, separate_observables=True)
    assert not det.any() and not obs.any()


def test_length1_matches_single_ppm():
    # same setup as test_routed_multi_patch_ls.test_state_channel_full_distance
    px = [_spec("Q1", 0, 0, "X_horizontal"), _spec("Q2", 2, 0, "X_horizontal")]
    seq = [PPMStep([("Q1", "Z"), ("Q2", "Z")])]
    exp = _exp(px, seq, {"Q1": "X", "Q2": "Z"}, {"Q1": "X", "Q2": "Z"},
               noise_params=NP)
    c = _build(exp)
    assert c.num_observables == 1
    assert _dist(c) == D


def test_joint_closure_detector_emitted():
    # PR#68 review blocker #1 (ruling reversed 2026-08-09): the terminal
    # closure of the merge-promoted joint is a legitimate long-range detector —
    # under matched noise, emitting it measures ~30% better MWPM LER, a
    # nine-configuration sweep found none made undecomposable by it, and the
    # weight filter has been removed wholesale (back to upstream main
    # behaviour).  Guards: the closure must exist (long-range, record count
    # > 2D+2), be deterministic without noise, keep the DEM decomposable, and
    # keep full graphlike distance.
    px = [_spec("Q1", 0, 0, "X_horizontal"), _spec("Q2", 2, 0, "X_horizontal")]
    seq = [PPMStep([("Q1", "Z"), ("Q2", "Z")])]
    st = {"Q1": "Z", "Q2": "Z"}
    clean = _build(_exp(px, seq, st, st))
    longrange = [inst for inst in clean.flattened() if inst.name == "DETECTOR"
                 and len(inst.targets_copy()) > 2 * D + 2]
    assert longrange, "joint-closure long-range detector was not emitted"
    det, obs = clean.compile_detector_sampler(seed=0).sample(
        2048, separate_observables=True)
    assert not det.any() and not obs.any()
    noisy = _build(_exp(px, seq, st, st, noise_params=NP))
    assert _dist(noisy) == D


def _num_ticks(c):
    return sum(1 for inst in c.flattened() if inst.name == "TICK")


def test_two_disjoint_ppm_full_distance():
    # ZZ(Q1,Q2) then ZZ(Q3,Q4), disjoint pairs in a row; all four persist to the end.
    px = [_spec("Q1", 0, 0, "X_horizontal"), _spec("Q2", 2, 0, "X_horizontal"),
          _spec("Q3", 4, 0, "X_horizontal"), _spec("Q4", 6, 0, "X_horizontal")]
    seq = [PPMStep([("Q1", "Z"), ("Q2", "Z")]),
           PPMStep([("Q3", "Z"), ("Q4", "Z")])]
    exp = _exp(px, seq,
               {"Q1": "X", "Q2": "Z", "Q3": "X", "Q4": "Z"},
               {"Q1": "X", "Q2": "Z", "Q3": "X", "Q4": "Z"}, noise_params=NP)
    c = _build(exp)
    assert c.num_observables == 2
    assert _dist(c) == D


def test_idle_rounds_adds_standalone_se():
    px = [_spec("Q1", 0, 0, "X_horizontal"), _spec("Q2", 2, 0, "X_horizontal"),
          _spec("Q3", 4, 0, "X_horizontal"), _spec("Q4", 6, 0, "X_horizontal")]
    seq = [PPMStep([("Q1", "Z"), ("Q2", "Z")]),
           PPMStep([("Q3", "Z"), ("Q4", "Z")])]
    init = {"Q1": "X", "Q2": "Z", "Q3": "X", "Q4": "Z"}
    base = _build(_exp(px, seq, init, init, noise_params=NP, idle_rounds=0))
    more = _build(_exp(px, seq, init, init, noise_params=NP, idle_rounds=2))
    assert _num_ticks(more) > _num_ticks(base)
    assert more.num_observables == base.num_observables


def test_role_switch_sequence_builds_full_distance():
    # Q measures X in PPM0 and Z in PPM1 — a mid-sequence role switch.  B is
    # FIRST-USED in PPM1, so the planner's first-use re-allocation serves the
    # switch with ZERO rotations: the corridor paints Q's registered need (Z)
    # and B is born recoloured (X̄_B presented as Z) — cheaper than the
    # physical rotate_90 the old planner inserted (the user's rotation policy:
    # rotate only for geometric blockage or bus saving).
    px = [_spec("Q", 0, 0, "X_vertical"),
          _spec("A", 2, 0, "X_horizontal"),
          _spec("B", 4, 0, "X_horizontal")]
    seq = [PPMStep([("Q", "X"), ("A", "Z")]),
           PPMStep([("Q", "Z"), ("B", "X")])]
    # schedule stays 'auto': the Q-B corridor bends around A, so its merged
    # rounds switch to the diagonal schedule; the recoloured M checks are
    # scheduled by the _M_SLOTS variant tables (full distance verified)
    exp = _exp(px, seq, {"Q": "X", "A": "Z", "B": "X"},
               {"Q": "Z", "A": "Z", "B": "X"}, noise_params=NP,
               auto_rotate=True, rotation_kind='auto')
    c = _build(exp)
    # since the chirality-law wall envelope (2026-08-02) the wall
    # alternative hosts directly — ZERO rotations, cheaper than the
    # rotate_90 the old planner inserted; physics gates stay
    assert exp.rotation_log == []
    assert c.num_observables == 2
    assert _dist(c) == D


def _cnot_px_seq():
    # ZZ(C,A): C,A horizontally adjacent -> vertical seam, Z̄_A (vertical) parallel.
    # XX(A,T): T vertically offset from A -> horizontal seam, X̄_A (horizontal) parallel.
    px = [_spec("C", 0, 0, "X_horizontal"),
          _spec("A", 2, 0, "X_horizontal"),
          _spec("T", 2, 2, "X_horizontal")]
    seq = [PPMStep([("C", "Z"), ("A", "Z")]),
           PPMStep([("A", "X"), ("T", "X")])]
    return px, seq


@pytest.mark.parametrize("cb,tb", [("Z", "X"), ("X", "X"), ("Z", "Z")])
def test_cnot_via_sequence_noiseless_valid(cb, tb):
    # Three input-basis combos, all num_observables==2 with deterministic detectors.
    # ("X","X") exercises X_C -> X_C X_T and ("Z","Z") exercises Z_T -> Z_C Z_T
    # (the entangling sectors), not just the trivially-invariant ("Z","X") sector.
    # NOTE: assert_noiseless confirms detector determinism / DEM validity — it does
    # NOT by itself prove the logical action equals CNOT (the tracker constructs the
    # observables to be deterministic regardless). A full stim.Tableau operator-level
    # witness (spec section 6.2) remains follow-up work, at parity with the existing
    # CNOTLSExperiment test bar.
    px, seq = _cnot_px_seq()
    exp = SequentialPPMExperiment(
        px, seq,
        initial_states={"A": "X", "C": cb, "T": tb},
        final_measure_states={"A": "Z", "C": cb, "T": tb},
        rounds=D, rounds_init=1, noise_params=None)
    c = _build(exp)
    assert_valid_circuit(c)
    assert_noiseless(c)
    assert_dem_valid(c)
    assert c.num_observables == 2


def test_cnot_via_sequence_full_distance():
    px, seq = _cnot_px_seq()
    exp = SequentialPPMExperiment(
        px, seq,
        initial_states={"A": "X", "C": "Z", "T": "X"},
        final_measure_states={"A": "Z", "C": "Z", "T": "X"},
        rounds=D, rounds_init=1, noise_params=NP)
    c = _build(exp)
    assert c.num_observables == 2
    assert _dist(c) == D


def test_six_body_then_pair_absorb_census():
    # 40-patch notebook regression (2026-07-31): after the six-body joint X̄
    # kills the pivots of the six single-patch Z̄s, the live ledger keeps five
    # pairwise products Z̄ᵢZ̄ⱼ; the next step measures one of those live
    # logicals directly (Z̄₃₃Z̄₃₄).  Historical bug: the promotion budget only
    # looked at the pre-window absorbed rank, did not know this window was
    # about to absorb that live row, treated the freed slot as a deferred-init
    # vacancy and promoted one more record-less STAB row — the same degree of
    # freedom got counted once in the live ledger and once in the absorbed
    # ledger, so census expected +1.  Once the budget also subtracts the
    # "to be absorbed in this window" count (mirroring the three gates of
    # Fix C): live 4 + absorbed 1 = 5 ✓.
    px = [_spec("q33", 0, 8, "X_vertical"), _spec("q34", 2, 8, "X_vertical"),
          _spec("q35", 4, 8, "X_vertical"), _spec("q36", 6, 8, "X_vertical"),
          _spec("q37", 8, 8, "X_vertical"), _spec("q38", 10, 8, "X_vertical")]
    seq = [PPMStep([("q33", "X"), ("q34", "X"), ("q35", "X"),
                    ("q36", "X"), ("q37", "X"), ("q38", "X")]),
           PPMStep([("q33", "Z"), ("q34", "Z")])]
    init = {s.name: "Z" for s in px}
    exp = SequentialPPMExperiment(
        px, seq, initial_states=init, final_measure_states=init,
        rounds=D, rounds_init=1, auto_rotate=True, rotation_kind='auto', rotate_saving_threshold=1)
    c = _build(exp)
    det, obs = c.compile_detector_sampler(seed=0).sample(
        512, separate_observables=True)
    assert not det.any() and not obs.any()


def test_runtime_rotation_invalidates_downstream_preregistered_coupler():
    # Regression: with auto_rotate off, a routed step registered UP FRONT for
    # the birth orientation must be re-registered if an earlier step's
    # feasibility-only repair rotation physically turns one of its patches.
    # Layout: q0-q1 cell-adjacent, q2 routed; q2 lives from step 0 so the
    # routed step 2 IS pre-registered.  Step 1 (q0 X, q1 Z on the vertical
    # seam) owes q1 X_horizontal -> repair rotates q1 at runtime, between the
    # up-front registration and the activation of step 2's coupler.
    px = [_spec("q0", 0, 0, "X_vertical"),
          _spec("q1", 1, 0, "X_vertical"),
          _spec("q2", 3, 0, "X_vertical")]
    init = {"q0": "Z", "q1": "Z", "q2": "Z"}
    adj_ok = PPMStep([("q0", "X"), ("q1", "X")])
    adj_repair = PPMStep([("q0", "X"), ("q1", "Z")])   # forces q1 rotation
    routed = PPMStep([("q1", "X"), ("q2", "X")])       # touches rotated q1

    def _run(steps):
        exp = SequentialPPMExperiment(
            px, steps, initial_states=init, final_measure_states=init,
            rounds=D, rounds_init=1, rotation_kind='auto', auto_rotate=False,
            lifetime_overrides={"q2": (0, len(steps) - 1)})
        return _build(exp), exp

    # control B: routed BEFORE the repair -> registered pre-rotation, no stale
    # coupler; must build a correct (silent, deterministic) circuit.
    (cb, _) = _run([adj_ok, routed, adj_repair])
    det, obs = cb.compile_detector_sampler(seed=0).sample(
        1024, separate_observables=True)
    assert not det.any(), "control B (no stale route) must be silent at p=0"
    assert (obs.shape[1] == 0) or bool((obs == obs[0]).all()), \
        "control B observables must be deterministic"

    # variant A: routed AFTER the repair -> the up-front coupler is stale.
    # The fix invalidates it and re-registers post-rotation, so the outcome is
    # computed against q1's REAL orientation: either a correct build, or an
    # HONEST route-time BentLayoutError.  It must NEVER activate the stale
    # coupler, which crashed deep in hook-benign scheduling (RuntimeError)
    # before the fix.
    try:
        _run([adj_ok, adj_repair, routed])
    except BentLayoutError:
        pass          # honest post-rotation infeasibility — acceptable
    except RuntimeError as e:                         # pragma: no cover
        pytest.fail(f"stale pre-registered coupler was activated (bug "
                    f"regressed): {e}")
