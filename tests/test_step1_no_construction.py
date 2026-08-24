"""Iron-rule regression (user ruling 2026-07-31): while step 1 is searching for
a route, no stabilizer-building method may be called, not even once.

Method: mine the construction entries (_assemble_region /
rule_based_joint_checks / place_patch) inside the _plan_rotations execution
window — any call there fails an assertion; outside the window they pass
through as usual and the circuit must still build fine.  Covers the two paths
that historically violated (or were suspected of violating) the rule: the snake
gateway (which used to build the whole wall layout for real during planning)
and plain auto-routing (which used to run place_patch in the probe preamble to
get the obstacle ancilla).
"""
import contextlib
import io

import circls.core.sequential_ppm_ls as spl
import circls.core.multi_patch_coupler as mpc
from circls.core.routed_multi_patch_ls import PatchSpec, origin_of
from circls.core.sequential_ppm_ls import (PPMStep,
                                                   SequentialPPMExperiment)

D = 3


def _spec(nm, a, b, o):
    return PatchSpec(nm, origin_of(a, b, D, seam=True), D, o)


def _guarded_build(exp, monkeypatch):
    in_planning = {'on': False}
    orig_plan = spl.SequentialPPMExperiment._plan_rotations

    def plan(self, reg):
        in_planning['on'] = True
        try:
            return orig_plan(self, reg)
        finally:
            in_planning['on'] = False

    monkeypatch.setattr(spl.SequentialPPMExperiment, '_plan_rotations', plan)
    for name in ('_assemble_region', 'rule_based_joint_checks', 'place_patch'):
        orig = getattr(mpc, name)

        def trap(*a, __orig=orig, __name=name, **k):
            assert not in_planning['on'], (
                f"step-1 planning called construction entry {__name}")
            return __orig(*a, **k)

        monkeypatch.setattr(mpc, name, trap)
    with contextlib.redirect_stdout(io.StringIO()):
        return exp.build()


def test_snake_gateway_plans_without_construction(monkeypatch):
    # Snake gateway scenario (same as test_zero_rotation_snake_sequence_d3):
    # _plan_snake used to build for real during planning; planning may now only
    # probe, and construction happens at registration time.
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
    c = _guarded_build(exp, monkeypatch)
    assert exp.rotation_log == []
    assert sorted(exp._snake_plans) == [2]
    # real construction result back-filled at registration time (has a layout)
    assert exp._snake_plans[2]['route_result'].layout is not None
    det, obs = c.compile_detector_sampler(seed=0).sample(
        256, separate_observables=True)
    assert not det.any() and not obs.any()


def test_auto_route_plans_without_construction(monkeypatch):
    # Plain auto-routing (two threshold-law settings: th=10 detours around the
    # wall with zero rotations / th=1 rotates and connects directly); planning
    # probes heavily, yet must make zero construction calls throughout.
    for th, rot_n in ((10, 0), (1, 1)):
        px = [_spec("q1", 0, 0, "X_vertical"), _spec("q2", 2, 0, "X_vertical")]
        seq = [PPMStep([("q1", "X"), ("q2", "Z")])]
        init = {"q1": "Z", "q2": "Z"}
        exp = SequentialPPMExperiment(
            px, seq, initial_states=init, final_measure_states=init,
            rounds=D, rounds_init=1, auto_rotate=True, rotation_kind='auto',
            rotate_saving_threshold=th)
        c = _guarded_build(exp, monkeypatch)
        assert len(exp.rotation_log) == rot_n, (th, exp.rotation_log)
        det, obs = c.compile_detector_sampler(seed=0).sample(
            256, separate_observables=True)
        assert not det.any() and not obs.any()
