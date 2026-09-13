"""Parity law (letter <=> colour lock in one corridor) and the zero-rotation
snake construction.

Inside a single corridor a recoloured mixed column is a "letter-flipping"
wall: every time a logical chain crosses one, X/Z flips once; and that column
only ever appears at the boundary where the red/blue convention goes out of
phase -- the letter flip and the colour flip are locked to the same boundary.
So for a corridor built from PLAIN stitches only, the two ends obey "same
measured letter <=> same colour convention" (topological, independent of the
route and of the access faces):

* same letter + different colour -- unconstructible with plain stitches
  (this file asserts the build fails);
* different letter + different colour -- constructible, including the
  decoupled configuration with conj in the **majority** (the old
  "letter != bus" inference cannot express it, so ``conj_names`` must be
  passed explicitly);
* same letter + same colour -- trivially constructible.

The piece that breaks the lock is the kf stretched wall (K&F §VI spatial
Hadamard interface: uniform bulk, flips the letter, keeps the colour), placed
**on the access seam of the colour-swapped target itself** (per-seam table
lookup: same/diff/diff = #4 pure wall, diff/same/diff = #6 mixed wall).
Layout rule (given by the user): the letter pair on the left/right legs of a
stretched weight-4 check is the opposite of the stabilizer of the square tile
adjacent on that side; the stretched weight-2 end cap goes on the end that is
not adjacent to the access patch's own weight-2 check. Colouring rule: for a
vertical-seam wall (patch and wall tile are left/right neighbours, horizontal
domino) the pure family (#4, uniform dominoes) has no solution inside the
compact-7 schedule template (4096-variant search, 0 solutions), so the whole
corridor is moved into the flipped gauge via ``flip_cells``, the wall lands on
the mixed row (#4 -> #6, i.e. the K&F Fig 4a family already validated,
'+'/'−' alternating) and the standard-side seam becomes a recoloured column
(#1 -> #7); a horizontal-seam wall (vertical domino) keeps the pure colouring
(the stacked-axis pure family is already validated). The parity-violating
Z⊗Z is still written as one step: one corridor, one coupler, one merge
window, zero rotations. The full d=3 criteria live in this file; d=5 was
measured on 2026-07-28 (rot=[], gl=5, obs=1, 158 ticks).
"""
import contextlib
import io

import pytest

from circls.core.multi_patch_coupler import (
    route_and_build, PatchSpec, origin_of, BentLayoutError)
from circls.core.sequential_ppm_ls import PPMStep, SequentialPPMExperiment
from lightstim.noise.config import NoiseConfig

pytestmark = pytest.mark.smoke

D = 3
NP = NoiseConfig(p_1q=1e-3, p_2q=1e-3, p_meas=1e-3, p_reset=1e-3, p_idle=1e-3)


def _spec(nm, a, b, o):
    return PatchSpec(nm, origin_of(a, b, D, seam=True), D, o)


def _corridor(oa, ob, ta, tb, conj, bus):
    """Minimal two-patch joint measurement over a one-tile straight corridor."""
    return route_and_build(
        [_spec("A", 0, 0, oa), _spec("B", 2, 0, ob)],
        [("A", ta), ("B", tb)], seam=True, route=[(1, 0)], bus=bus,
        conj_names=frozenset(conj))


def test_same_letter_diff_colour_unconstructible():
    # Parity law violated: A recoloured + B standard, both measured in Z --
    # unconstructible under either colouring with plain stitches only (breaking
    # the lock needs a kf stretched wall on the access seam, see snake tests)
    for bus in ('Z', 'X'):
        with pytest.raises(BentLayoutError):
            _corridor("X_horizontal", "X_horizontal", "Z", "Z", {"A"}, bus)


def test_diff_letter_diff_colour_minority_conj():
    # Parity law satisfied, conj in the minority (= the same set the legacy
    # letter inference produced)
    r = _corridor("X_vertical", "X_horizontal", "X", "Z", {"B"}, 'X')
    assert r.status == 'ok'
    assert all(r.layout.verify().values())


def test_diff_letter_diff_colour_majority_conj_decoupled():
    # Parity law satisfied, conj in the majority -- a configuration the old
    # inference cannot express, only an explicit conj_names can build it
    r = _corridor("X_vertical", "X_horizontal", "X", "Z", {"A"}, 'Z')
    assert r.status == 'ok'
    assert all(r.layout.verify().values())


def test_same_letter_same_colour_plain():
    r = _corridor("X_horizontal", "X_horizontal", "Z", "Z", set(), 'Z')
    assert r.status == 'ok'
    assert all(r.layout.verify().values())


def test_direct_row6_wall_mixed_letter_joint():
    # Direct #6 entry (user ruling 2026-07-30): a different-letter same-colour
    # seam -> mixed KF stretched wall, corridor not flipped. fig5 green-line
    # scenario: q1 measured in X on a Z-bus (majority letter), standard
    # registration -> the seam is natively #6, wall on q1's east seam;
    # q6 (birth-flipped Xh) and q4 (litinski state Xh) are both plain #1
    # seams. A 3-tile midline corridor replaces the earlier 11-tile detour.
    px = [_fig5_spec("q1", 0, 0, "X_vertical"),
          _fig5_spec("q2", 2, 0, "X_vertical"),
          _fig5_spec("q3", 0, 2, "X_vertical"),
          _fig5_spec("q4", 2, 2, "X_horizontal"),
          _fig5_spec("q5", 1, 3, "X_vertical"),
          _fig5_spec("q6", 2, 1, "X_horizontal")]
    with contextlib.redirect_stdout(io.StringIO()):
        r = route_and_build(
            px, [("q1", "X"), ("q6", "Z"), ("q4", "Z")], seam=True,
            route=[(1, 0), (1, 1), (1, 2)], bus="Z",
            conj_names=frozenset(),
            no_stitch=[("q2", (1, 0)), ("q3", (1, 2))])
    assert r.status == 'ok', getattr(r, 'message', '')
    assert all(r.layout.verify().values())
    # Wall records = d checks (d-1 stretched weight-4 + 1 end cap), each with
    # opposite letters on its left and right legs
    kf = [ch for ch in r.layout.checks if ch.get('kf')]
    assert len(kf) == D
    assert all(len(set(ch['pauli'].values())) == 2 for ch in kf)
    c0 = r.layout.build_circuit(rounds=D, p=0.0)
    det, obs = c0.compile_detector_sampler(seed=0).sample(
        1024, separate_observables=True)
    assert not det.any() and not obs.any()
    noisy = r.layout.build_circuit(rounds=D, p=1e-3)
    assert _hyper_distance(noisy) >= D


def _fig5_spec(nm, a, b, o):
    return PatchSpec(nm, origin_of(a, b, D, seam=True), D, o)


def test_snake_bad_route_raises_clearly():
    # a parity-violating step with an EXPLICIT route that cannot host the
    # snake (no route cell edge-adjacent to the colour-swapped target, so
    # there is no seam to put the stretched wall on) fails loudly at
    # planning — it must NOT fall back to the rotation repair with that
    # route (unvalidated path, used to crash deep inside the fixed-schedule
    # validator)
    px = [_spec("q1", 0, 0, "X_vertical"), _spec("q2", 1, 0, "X_vertical"),
          _spec("q3", 0, 1, "X_vertical")]
    seq = [PPMStep([("q1", "X"), ("q2", "Z")]),
           PPMStep([("q1", "Z"), ("q3", "Z")]),
           PPMStep([("q2", "Z"), ("q3", "Z")], route=[(1, 2), (0, 2)])]
    with pytest.raises(BentLayoutError, match="snake|no feasible"):
        SequentialPPMExperiment(
            px, seq,
            initial_states={"q1": "X", "q2": "Z", "q3": "Z"},
            final_measure_states={"q1": "X", "q2": "Z", "q3": "Z"},
            rounds=D, rounds_init=1, auto_rotate=True, rotation_kind='auto',
            rotate_saving_threshold=10, colour_swapped={"q2"}).build()


def test_zero_rotation_snake_sequence_d3():
    # Snake E2E: the Z⊗Z between recoloured q2 (born on row 4 of PPM1) and
    # standard q3 is written as one step; once the planner sees the parity
    # violation in segment 1 it takes the snake instead of rotating: a 5-tile
    # short corridor, wall on the q2|(2,0) vertical seam (colouring 2, the
    # corridor enters the flipped gauge, #6 mixed wall + a #7 recoloured
    # column on the q3 seam), q3|(0,2) horizontal seam; one coupler, one
    # merge window, zero rotations, full distance.
    # STRICT BIRTHS: q2's recolour is now the USER INPUT (declared
    # geometry X_vertical + colour_swapped) instead of the retired
    # birth-inference — physically the identical patch
    px = [_spec("q1", 0, 0, "X_vertical"), _spec("q2", 1, 0, "X_vertical"),
          _spec("q3", 0, 1, "X_vertical")]
    seq = [PPMStep([("q1", "X"), ("q2", "Z")]),
           PPMStep([("q1", "Z"), ("q3", "Z")]),
           PPMStep([("q2", "Z"), ("q3", "Z")],
                   route=[(2, 0), (2, 1), (2, 2), (1, 2), (0, 2)])]
    exp = SequentialPPMExperiment(
        px, seq,
        initial_states={"q1": "X", "q2": "Z", "q3": "Z"},
        final_measure_states={"q1": "X", "q2": "Z", "q3": "Z"},
        rounds=D, rounds_init=1, auto_rotate=True, rotation_kind='auto', rotate_saving_threshold=10,
        colour_swapped={"q2"})
    with contextlib.redirect_stdout(io.StringIO()):
        c = exp.build()
    assert exp.rotation_log == []
    assert sorted(exp._snake_plans) == [2]
    assert c.num_observables == 1
    # Wall records: the #6 mixed family -- d-1 stretched weight-4 checks + 1
    # stretched weight-2 end cap, each with opposite letters on its left and
    # right legs, alternating '+'/'−' variants
    kf = [ch for ch in exp._snake_plans[2]['route_result'].layout.checks
          if ch.get('kf')]
    assert len(kf) == D
    assert all(len(set(ch['pauli'].values())) == 2 for ch in kf)
    assert len({ch['kf']['orient'] for ch in kf}) == 2
    det, obs = c.compile_detector_sampler(seed=0).sample(
        1024, separate_observables=True)
    assert not det.any(), "detector fired at p=0"
    assert not obs.any(), "observable not deterministic at p=0"
    noisy = exp.builder.build_noisy_circuit(noise_params=NP,
                                            noise_model='circuit_level')
    noisy.detector_error_model(decompose_errors=True)
    assert len(noisy.shortest_graphlike_error()) == D


def test_three_target_one_step_all_standard():
    # Measure 3 patches in one step (all standard colour): a three-tile
    # corridor with every target hanging off its own tile; the standard
    # T-shaped merge measures two independent pairwise products (obs=2) at
    # full distance
    px = [_spec("q1", 0, 0, "X_horizontal"), _spec("q2", 4, 0, "X_horizontal"),
          _spec("q3", 2, 1, "X_vertical")]
    seq = [PPMStep([("q1", "Z"), ("q2", "Z"), ("q3", "Z")],
                   route=[(1, 0), (2, 0), (3, 0)])]
    exp = SequentialPPMExperiment(
        px, seq, initial_states={"q1": "Z", "q2": "Z", "q3": "Z"},
        final_measure_states={"q1": "Z", "q2": "Z", "q3": "Z"},
        rounds=D, rounds_init=1, auto_rotate=True, rotation_kind='auto', rotate_saving_threshold=10)
    with contextlib.redirect_stdout(io.StringIO()):
        c = exp.build()
    assert exp.rotation_log == []
    assert c.num_observables == 2
    det, obs = c.compile_detector_sampler(seed=0).sample(
        1024, separate_observables=True)
    assert not det.any() and not obs.any()
    noisy = exp.builder.build_noisy_circuit(noise_params=NP,
                                            noise_model='circuit_level')
    noisy.detector_error_model(decompose_errors=True)
    assert len(noisy.shortest_graphlike_error()) == D


def test_three_target_one_step_with_recoloured_member():
    # Measure 3 patches in one step where q2 was recoloured by the previous
    # step (an adjacent mixed PPM): the planner takes the generalized snake --
    # wall on q2's vertical seam (colouring 2, mixed), the q1/q3 seams become
    # recoloured columns; zero rotations, full distance
    px = [_spec("q1", 0, 0, "X_horizontal"), _spec("q2", 4, 0, "X_vertical"),
          _spec("q3", 2, 1, "X_vertical"), _spec("q4", 5, 0, "X_vertical")]
    seq = [PPMStep([("q4", "X"), ("q2", "Z")]),
           PPMStep([("q1", "Z"), ("q2", "Z"), ("q3", "Z")],
                   route=[(1, 0), (2, 0), (3, 0)])]
    exp = SequentialPPMExperiment(
        px, seq,
        initial_states={"q1": "Z", "q2": "Z", "q3": "Z", "q4": "X"},
        final_measure_states={"q1": "Z", "q2": "Z", "q3": "Z", "q4": "X"},
        rounds=D, rounds_init=1, auto_rotate=True, rotation_kind='auto', rotate_saving_threshold=10,
        colour_swapped={"q2"})
    with contextlib.redirect_stdout(io.StringIO()):
        c = exp.build()
    assert exp.rotation_log == []
    assert sorted(exp._snake_plans) == [1]
    assert [w for w, _ in exp._snake_plans[1]['walls']] == ['q2']
    det, obs = c.compile_detector_sampler(seed=0).sample(
        1024, separate_observables=True)
    assert not det.any() and not obs.any()
    noisy = exp.builder.build_noisy_circuit(noise_params=NP,
                                            noise_model='circuit_level')
    noisy.detector_error_model(decompose_errors=True)
    assert len(noisy.shortest_graphlike_error()) == D


def _hyper_distance(noisy):
    """True distance including hyperedges (the gauge-column parity detector of
    a two-wall block creates an essential hyperedge, so graphlike decomposition
    does not apply -- decode with an mwpf-style hypergraph decoder)."""
    return len(noisy.search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=6,
        dont_explore_edges_with_degree_above=8,
        dont_explore_edges_increasing_symptom_degree=False,
        canonicalize_circuit_errors=True))


def test_three_target_two_recoloured_members():
    # Both q1 and q2 among the 3 targets were recoloured by earlier mixed
    # steps -> one #6 wall at each end of the corridor. A two-wall block only
    # measures the triple product (no pairwise product at all, rank deficit 1);
    # the tracker books it through the "promotion quota guard + combined
    # absorb/fold" path. The gauge-column parity detector puts an essential
    # hyperedge in the fault structure -> the distance gate uses the true
    # hyperedge search (graphlike decomposition does not apply to this block).
    # q0/q4 use the Z convention: under the X convention, "a patch on the X̄
    # side of a mixed step idling inside somebody else's bent window" hits a
    # pre-existing hook defect unrelated to the wall (see the xfail test
    # below).
    px = [_spec("q0", 0, 0, "X_vertical"), _spec("q1", 1, 0, "X_vertical"),
          _spec("q2", 5, 0, "X_vertical"), _spec("q3", 3, 1, "X_vertical"),
          _spec("q4", 6, 0, "X_vertical")]
    seq = [PPMStep([("q0", "X"), ("q1", "Z")]),
           PPMStep([("q4", "X"), ("q2", "Z")]),
           PPMStep([("q1", "Z"), ("q2", "Z"), ("q3", "Z")],
                   route=[(2, 0), (3, 0), (4, 0)])]
    states = {"q0": "Z", "q1": "Z", "q2": "Z", "q3": "Z", "q4": "Z"}
    exp = SequentialPPMExperiment(
        px, seq, initial_states=states, final_measure_states=states,
        rounds=D, rounds_init=1, auto_rotate=True, rotation_kind='auto', rotate_saving_threshold=10,
        colour_swapped={"q1", "q2"})
    with contextlib.redirect_stdout(io.StringIO()):
        c = exp.build()
    assert exp.rotation_log == []
    assert [w for w, _ in exp._snake_plans[2]['walls']] == ['q1', 'q2']
    det, obs = c.compile_detector_sampler(seed=0).sample(
        1024, separate_observables=True)
    assert not det.any() and not obs.any()
    noisy = exp.builder.build_noisy_circuit(noise_params=NP,
                                            noise_model='circuit_level')
    assert _hyper_distance(noisy) == D


# Turned into a normal assertion on 2026-08-03: the M8 fix (the end cap now
# picks its end by the handbook 10.4 single-quota rule) changed the schedule
# geometry of this layout, the recorded Y-hook alignment disappeared and the
# true hyperedge distance is back to D -- locked in here. NOTE: Y-hook style
# defects are geometry dependent, so this test passing does not mean the class
# is gone globally; the mechanism described by the old xfail is kept in git
# history (during a bent merge window an idling patch can have the logical
# operator of one basis silently flipped by a Y-hook pair on two consecutive
# rounds; the graphlike search cannot see Y-hook pairs).
def test_idle_patch_during_bent_window_full_distance():
    px = [_spec("q0", 0, 0, "X_vertical"), _spec("q1", 1, 0, "X_horizontal"),
          _spec("q2", 5, 0, "X_horizontal"), _spec("q3", 3, 1, "X_vertical"),
          _spec("q4", 6, 0, "X_vertical")]
    seq = [PPMStep([("q0", "X"), ("q1", "Z")]),
           PPMStep([("q4", "X"), ("q2", "Z")])]
    states = {"q0": "X", "q1": "Z", "q2": "Z", "q3": "Z", "q4": "X"}
    exp = SequentialPPMExperiment(
        px, seq, initial_states=states, final_measure_states=states,
        rounds=D, rounds_init=1, auto_rotate=True, rotation_kind='auto', rotate_saving_threshold=10)
    with contextlib.redirect_stdout(io.StringIO()):
        exp.build()
    noisy = exp.builder.build_noisy_circuit(noise_params=NP,
                                            noise_model='circuit_level')
    assert _hyper_distance(noisy) == D


def test_fig5_six_patch_four_ppm_sequence():
    # show_fig5 sequence (green straight-line variant): PPM1-3 route fully
    # automatically (PPM1 a single straight tile, PPM3 a three-target mixed
    # letter step with q4 born in the minority); PPM4 uses first-use aware
    # colouring (colour X + q6 born recoloured) plus touch-without-stitch
    # (q4 south | corridor (2,1)). Attachment topology law: each target
    # appearing once in the joint <=> single-seam attachment (a two-seam
    # bridge squares away the logical of the middle patch). Orientation is a
    # resource shared by the whole sequence: this combination concentrates the
    # unstitch cost on PPM4.
    px = [_spec("q1", 0, 0, "X_vertical"), _spec("q2", 2, 0, "X_horizontal"),
          _spec("q3", 0, 2, "X_vertical"), _spec("q4", 2, 2, "X_horizontal"),
          _spec("q5", 1, 3, "X_vertical"), _spec("q6", 3, 1, "X_horizontal")]
    seq = [PPMStep([("q1", "X"), ("q2", "Z")]),
           PPMStep([("q1", "Z"), ("q5", "Z")]),
           PPMStep([("q1", "Z"), ("q3", "Z"), ("q4", "X")]),
           PPMStep([("q1", "X"), ("q6", "Z"), ("q4", "Z")],
                   route=[(1, 0), (1, 1), (1, 2), (2, 1)],
                   unstitch=[("q4", (2, 1))])]
    init = {f"q{i}": "Z" for i in range(1, 7)}
    exp = SequentialPPMExperiment(
        px, seq, initial_states=init, final_measure_states=init,
        rounds=D, rounds_init=1, auto_rotate=True, rotation_kind='auto', rotate_saving_threshold=10)
    with contextlib.redirect_stdout(io.StringIO()):
        c = exp.build()
    # STRICT BIRTHS: PPM2's q4^X minority no longer gets the free birth
    # conjugation; since the chirality-law wall envelope (2026-08-02, both
    # orientations, handedness-gated exits) q4's direct-#6 wall hosts
    # directly — zero rotations, the wall serves the whole sequence
    assert exp.rotation_log == []
    assert c.num_observables == 4
    det, obs = c.compile_detector_sampler(seed=0).sample(
        1024, separate_observables=True)
    assert not det.any() and not obs.any()
    noisy = exp.builder.build_noisy_circuit(noise_params=NP,
                                            noise_model='circuit_level')
    assert _hyper_distance(noisy) == D


def test_fig5_all_x_ppm4_exchange_restore():
    # The PPM4 = X̄1X̄6X̄4 variant of show_fig5: J4 anticommutes with the
    # already promoted joints (Z̄1Z̄5, Z̄1Z̄3X̄4) -- a "kill and restore"
    # exchange inside the same block, net change of freedom 0. Two historical
    # bugs met here: (1) the per-lobe greedy -1 of Case A cannot see a
    # block-level restore (fixed by booking it at the joint level in
    # LogicalLedger); (2) the WriteBack promotion budget counted "logicals
    # seen so far" -- in the return order of the independent bases a promotion
    # candidate can come before a surviving logical row, so the budget gets
    # claimed by the wrong row and the census over-counts (Expected 4 /
    # Found 5). Pre-counting the survivors first turned this sequence green.
    px = [_spec("q1", 0, 0, "X_vertical"), _spec("q2", 2, 0, "X_vertical"),
          _spec("q3", 0, 2, "X_vertical"), _spec("q4", 2, 2, "X_vertical"),
          _spec("q5", 1, 3, "X_vertical"), _spec("q6", 2, 1, "X_vertical")]
    seq = [PPMStep([("q1", "X"), ("q2", "Z")]),
           PPMStep([("q1", "Z"), ("q5", "Z")]),
           PPMStep([("q1", "Z"), ("q3", "Z"), ("q4", "X")]),
           PPMStep([("q1", "X"), ("q6", "X"), ("q4", "X")])]
    init = {f"q{i}": "Z" for i in range(1, 7)}
    exp = SequentialPPMExperiment(
        px, seq, initial_states=init, final_measure_states=init,
        rounds=D, rounds_init=1, auto_rotate=True, rotation_kind='auto', rotate_saving_threshold=1)
    with contextlib.redirect_stdout(io.StringIO()):
        c = exp.build()
    # The final readout retires the logicals into observables, so the internal
    # expected count is no longer 4 after build; the post-build census
    # assertion is the observable count
    assert c.num_observables == 4
    det, obs = c.compile_detector_sampler(seed=0).sample(
        1024, separate_observables=True)
    assert not det.any() and not obs.any()
    noisy = exp.builder.build_noisy_circuit(noise_params=NP,
                                            noise_model='circuit_level')
    # The final closure detector of the unwatched gauge direction (the pmm#8
    # quota-free row, a diagonal 4-qubit X string) is now suppressed by the
    # records == [] guard -- it used to attach an irreducible 6th symptom (the
    # long D937) to the Y⊗Y composite mechanism and broke strict
    # decomposition; with it gone, strict decomposition should pass
    noisy.detector_error_model(decompose_errors=True)
    assert _hyper_distance(noisy) == D


def test_strict_birth_mixed_pair_threshold_law():
    # Demo of STRICT BIRTHS + cost law L + threshold x r: both patches declare
    # X_vertical, mixed step [q1 X, q2 Z]. Two candidates: zero rotation = a
    # 3-tile detour to the south + a #6 mixed wall on q2; q2 rotate_90 = a
    # 1-tile direct link with a #7 column.
    #   threshold=10: 3 < 1+10 -> the zero-rotation wall detour wins;
    #   threshold=1 : 1+1 < 3 -> the rotated direct link wins.
    px = [_spec("q1", 0, 0, "X_vertical"), _spec("q2", 2, 0, "X_vertical")]
    seq = [PPMStep([("q1", "X"), ("q2", "Z")])]
    init = {"q1": "Z", "q2": "Z"}

    def _go(th):
        exp = SequentialPPMExperiment(
            px, seq, initial_states=init, final_measure_states=init,
            rounds=D, rounds_init=1, auto_rotate=True, rotation_kind='auto',
            rotate_saving_threshold=th)
        with contextlib.redirect_stdout(io.StringIO()):
            c = exp.build()
        assert exp.birth_reorientations == []
        det, obs = c.compile_detector_sampler(seed=0).sample(
            1024, separate_observables=True)
        assert not det.any() and not obs.any()
        noisy = exp.builder.build_noisy_circuit(noise_params=NP,
                                                noise_model='circuit_level')
        assert _hyper_distance(noisy) >= D
        return exp

    e10 = _go(10)
    assert e10.rotation_log == []
    # 2026-09-11: the same-length corridor on the other side is admissible
    # now that its wall end record is constructible (checks above hold)
    assert sorted(e10._routes[0].tree) == [(1, -1), (1, 0), (2, -1)]
    assert sum(1 for ch in e10._routes[0].layout.checks if ch.get('kf')) == D
    e1 = _go(1)
    assert e1.rotation_log == [(0, "q2", "rotate_90")]
    assert sorted(e1._routes[0].tree) == [(1, 0)]


def test_fig5_all_vertical_declarations_birth_repair():
    # All six patches declare X_vertical (lazy declaration) and no explicit
    # route: parallel-law stitching + borrowing not-yet-born tiles + the birth
    # freedom let the whole sequence complete fully automatically
    px = [_spec("q1", 0, 0, "X_vertical"), _spec("q2", 2, 0, "X_vertical"),
          _spec("q3", 0, 2, "X_vertical"), _spec("q4", 2, 2, "X_vertical"),
          _spec("q5", 1, 3, "X_vertical"), _spec("q6", 3, 1, "X_vertical")]
    seq = [PPMStep([("q1", "X"), ("q2", "Z")]),
           PPMStep([("q1", "Z"), ("q5", "Z")]),
           PPMStep([("q1", "Z"), ("q3", "Z"), ("q4", "X")]),
           PPMStep([("q1", "X"), ("q6", "Z"), ("q4", "Z")])]
    init = {f"q{i}": "Z" for i in range(1, 7)}
    exp = SequentialPPMExperiment(
        px, seq, initial_states=init, final_measure_states=init,
        rounds=D, rounds_init=1, auto_rotate=True, rotation_kind='auto', rotate_saving_threshold=10)
    with contextlib.redirect_stdout(io.StringIO()):
        c = exp.build()
    # STRICT BIRTHS: the birth knob is retired -- any orientation change that
    # is needed now shows up as a real rotation
    assert exp.birth_reorientations == []
    assert c.num_observables == 4
    det, obs = c.compile_detector_sampler(seed=0).sample(
        1024, separate_observables=True)
    assert not det.any() and not obs.any()
    noisy = exp.builder.build_noisy_circuit(noise_params=NP,
                                            noise_model='circuit_level')
    assert _hyper_distance(noisy) == D


@pytest.mark.xfail(reason="corner-touch defect family: in the 11-tile zero-"
                          "rotation detour corridor found by EMV a corridor "
                          "tile touches a target diagonally, true hyperedge "
                          "distance = 1. Unrelated to the eight-row access-"
                          "face table (a clean-geometry control shows such "
                          "detours keep full distance under that table, so "
                          "the loss comes from corner-touch); restore the "
                          "rotation-subset assertion once the corner-touch "
                          "family is characterized and candidates filtered",
                   strict=True)
def test_fig5_q6_original_placement_rotation_subset():
    # q6 back at (2,1) (its original placement) + all-X_vertical lazy
    # declarations + no explicit route: PPM4 needs "the rotate_90 subset
    # {q1,q4} of the registered targets + a q6 birth to balance it" (the
    # solution the user pointed out) -- the subset search of the uniform
    # colour repair finds it: colour Z, two SWAP-network rotations, a q6 birth
    # flip, corridor [(0,1),(1,1),(1,2)]
    px = [_spec("q1", 0, 0, "X_vertical"), _spec("q2", 2, 0, "X_vertical"),
          _spec("q3", 0, 2, "X_vertical"), _spec("q4", 2, 2, "X_vertical"),
          _spec("q5", 1, 3, "X_vertical"), _spec("q6", 2, 1, "X_vertical")]
    seq = [PPMStep([("q1", "X"), ("q2", "Z")]),
           PPMStep([("q1", "Z"), ("q5", "Z")]),
           PPMStep([("q1", "Z"), ("q3", "Z"), ("q4", "X")]),
           PPMStep([("q1", "X"), ("q6", "Z"), ("q4", "Z")])]
    init = {f"q{i}": "Z" for i in range(1, 7)}
    exp = SequentialPPMExperiment(
        px, seq, initial_states=init, final_measure_states=init,
        rounds=D, rounds_init=1, auto_rotate=True, rotation_kind='auto', rotate_saving_threshold=10)
    with contextlib.redirect_stdout(io.StringIO()):
        c = exp.build()
    assert exp.rotation_log == [(3, "q1", "rotate_90"), (3, "q4", "rotate_90")]
    assert (3, "q6") in exp.birth_reorientations
    assert sorted(exp._routes[3].tree) == [(0, 1), (1, 1), (1, 2)]
    # The SWAP networks of q1 and q4 run in parallel (same time span): the
    # total tick count is one network (6(d-1) = 12) below the sequential
    # version (182)
    assert c.num_ticks <= 175, c.num_ticks
    det, obs = c.compile_detector_sampler(seed=0).sample(
        1024, separate_observables=True)
    assert not det.any() and not obs.any()
    noisy = exp.builder.build_noisy_circuit(noise_params=NP,
                                            noise_model='circuit_level')
    assert _hyper_distance(noisy) == D
