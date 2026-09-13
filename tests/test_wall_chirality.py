"""Envelope of the #6 wall chirality law (2026-08-02): the full
(orientation x side x exit) truth table.

24/24 measured matrix (each combination first located with a dispatch stub,
then re-checked through the real dispatch): the wall cell must be a terminal
cell (deg 1), and the bus may leave the wall cell only straight ahead or with
a turn that follows the scan chirality -- with s = wall - patch and
e = exit - wall, the bad chirality cross(s, e) is -1 for X_vertical and +1 for
X_horizontal (transposition is a reflection, so the bad chirality flips with
the family; ptype transpose symmetry makes everything else covariant, the same
argument as litinski +-x).  Legal combinations pass verify / p=0 / graphlike
distance; illegal ones are rejected by dispatch -> hard error (iron rule: if
step two cannot be built we do not backtrack for another route).
"""
import contextlib
import io

import pytest

from circls.core.multi_patch_coupler import route_and_build
from circls.core.routed_multi_patch_ls import PatchSpec, origin_of
from circls.core.sequential_ppm_ls import PPMStep, SequentialPPMExperiment
from lightstim.noise.config import NoiseConfig

pytestmark = pytest.mark.smoke

D = 3
NP = NoiseConfig(p_1q=1e-3, p_2q=1e-3, p_meas=1e-3, p_reset=1e-3, p_idle=1e-3)
FLIP = {'X': 'Z', 'Z': 'X'}
DIRS = {'N': (0, -1), 'S': (0, 1), 'E': (1, 0), 'W': (-1, 0)}


def _want(P, seam_ew):
    return 'X_horizontal' if (P == 'Z') == seam_ew else 'X_vertical'


def _scenario(d, fam, side, ex):
    """Target T at the centre, wall cell against the ``side`` face, corridor
    running two cells straight along ``ex``, partner P hooked on at the tail
    parallel to the seam."""
    tc = (4, 4)
    seam_ew = side in 'EW'
    P_t = next(P for P in 'ZX' if _want(P, seam_ew) == fam)
    bus = FLIP[P_t]
    wall = (tc[0] + DIRS[side][0], tc[1] + DIRS[side][1])
    exit_c = (wall[0] + DIRS[ex][0], wall[1] + DIRS[ex][1])
    attach = (exit_c[0] + DIRS[ex][0], exit_c[1] + DIRS[ex][1])
    p_home = (attach[0] + DIRS[ex][0], attach[1] + DIRS[ex][1])
    o_p = _want(bus, DIRS[ex][0] != 0)
    px = [PatchSpec("T", origin_of(*tc, d, seam=True), d, fam),
          PatchSpec("P", origin_of(*p_home, d, seam=True), d, o_p)]
    return px, [("T", P_t), ("P", bus)], [wall, exit_c, attach], bus


MATRIX = [(fam, side, ex)
          for fam in ('X_vertical', 'X_horizontal')
          for side in 'NSEW'
          for ex in 'NSEW'
          if DIRS[ex] != tuple(-v for v in DIRS[side])]


@pytest.mark.parametrize("fam,side,ex", MATRIX,
                         ids=[f"{f[2]}-{s}-{e}" for f, s, e in MATRIX])
def test_chirality_matrix(fam, side, ex):
    # 2026-09-11: every one of the 24 (family, side, exit) combinations is
    # constructible.  The eight "bad-handedness" cases used to be declined
    # because the stretched seam's end record was the bare weight-2 lobe
    # even when the corridor continued past the band end; the paper's
    # concave-corner rule (weight-3 there) hosts them, so the handedness
    # veto is gone and this test verifies all 24 the same way.
    px, tgt, tree, bus = _scenario(D, fam, side, ex)
    with contextlib.redirect_stdout(io.StringIO()):
        r = route_and_build(px, tgt, seam=True, route=tree, bus=bus,
                            conj_names=frozenset())
    assert r.status == 'ok', getattr(r, 'message', '')
    assert all(r.layout.verify().values())
    kf = [ch for ch in r.layout.checks if ch.get('kf')]
    assert len(kf) == D                  # d-1 stretched weight-4 + 1 end cap
    c0 = r.layout.build_circuit(rounds=D, p=0.0)
    det, obs = c0.compile_detector_sampler(seed=0).sample(
        512, separate_observables=True)
    assert not det.any() and not obs.any()
    noisy = r.layout.build_circuit(rounds=D, p=1e-3)
    noisy.detector_error_model(decompose_errors=True)
    assert len(noisy.shortest_graphlike_error()) >= D


def test_corner_mixed_step_litinski_only_example4():
    # the deadlock scenario of notebook Example 4: q1 sits in the (0,0) corner
    # measuring Z (X bus in the minority), q2 blocks the south.  Under
    # litinski-only the only way out = rotate q1 to X_horizontal + east seam
    # #6 wall + east exit -- exactly the combination the transposed family
    # unlocks (before the fix all 16 candidates died and it hard-errored).
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
    with contextlib.redirect_stdout(io.StringIO()):
        c = exp.build()
    # 2026-09-11: the east-seam #6 wall with the east exit is constructible
    # directly (concave-corner end record), so no rotation is needed
    assert exp.rotation_log == []
    kf = [ch for ch in exp._routes[0].layout.checks if ch.get('kf')]
    assert len(kf) == D
    det, obs = c.compile_detector_sampler(seed=0).sample(
        1024, separate_observables=True)
    assert not det.any() and not obs.any()
    noisy = exp.builder.build_noisy_circuit(noise_params=NP,
                                            noise_model='circuit_level')
    noisy.detector_error_model(decompose_errors=True)
    assert len(noisy.shortest_graphlike_error()) == D


def test_x_horizontal_wall_d5():
    # transposed family d=5 spot check: X_horizontal target, south wall, east
    # exit (the full transpose of the green-line case)
    d = 5
    px, tgt, tree, bus = _scenario(d, 'X_horizontal', 'S', 'E')
    with contextlib.redirect_stdout(io.StringIO()):
        r = route_and_build(px, tgt, seam=True, route=tree, bus=bus,
                            conj_names=frozenset())
    assert r.status == 'ok', getattr(r, 'message', '')
    assert all(r.layout.verify().values())
    assert sum(1 for ch in r.layout.checks if ch.get('kf')) == d
    c0 = r.layout.build_circuit(rounds=d, p=0.0)
    det, obs = c0.compile_detector_sampler(seed=0).sample(
        512, separate_observables=True)
    assert not det.any() and not obs.any()
    noisy = r.layout.build_circuit(rounds=d, p=1e-3)
    noisy.detector_error_model(decompose_errors=True)
    assert len(noisy.shortest_graphlike_error()) >= d
