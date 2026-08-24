"""Attach-face distance table on LEGAL constructions (regression guard).

HISTORY (2026-07-30): the original eight-cell table reported one
hook-UNSAFE face — (conjugate, B, X) attaching south lost a unit of
graphlike distance — and a +100 routing surcharge machinery was built
around it.  That measurement was made on ILLEGAL constructions: the
long L-shaped probe routes only built through the since-retired
unlocked ladder retry that silently repainted BOTH patches (a mid-life
retype the architecture forbids).  Re-measured on legal builds
(straight corridors, native lock unconditional, both d=3 and d=5,
short and long corridors), EVERY attach face reaches full distance —
there is no unsafe face, and the surcharge machinery was removed.

These tests pin that fact: all eight (type, letter, face) cells at
full distance on minimal legal geometry, plus a long-corridor pair on
the historically-suspected face.
"""
import contextlib
import io

import pytest

from circls.core.sequential_ppm_ls import PPMStep, SequentialPPMExperiment
from circls.core.routed_multi_patch_ls import PatchSpec, origin_of
from lightstim.noise.config import NoiseConfig

D = 3
NP = NoiseConfig(p_1q=1e-3, p_2q=1e-3, p_meas=1e-3, p_reset=1e-3, p_idle=1e-3)

#: the original suite's face naming: S = the corridor row at b - 1,
#: N = the row at b + 1, E/W = the columns at a ± 1
_FACE = {"E": (1, 0), "W": (-1, 0), "S": (0, -1), "N": (0, 1)}
_SPEC_OF = {"textbook": "X_vertical", "conjugate": "X_horizontal"}


def _dist(exp):
    noisy = exp.builder.build_noisy_circuit(noise_params=NP,
                                            noise_model='circuit_level')
    noisy.detector_error_model(decompose_errors=True)
    return len(noisy.shortest_graphlike_error())


def _build(px, letter, route, d=D):
    exp = SequentialPPMExperiment(
        px, [PPMStep([("Q", letter), ("P", letter)], route=route)],
        initial_states={"Q": letter, "P": letter},
        final_measure_states={"Q": letter, "P": letter},
        rounds=d, rounds_init=1, auto_rotate=False, schedule="diagonal")
    with contextlib.redirect_stdout(io.StringIO()):
        exp.build()
    return exp


def _spec(nm, a, b, o, d=D):
    return PatchSpec(nm, origin_of(a, b, d, seam=True), d, o)


_TABLE = [("textbook", "X", "E"), ("textbook", "X", "W"),
          ("textbook", "Z", "S"), ("textbook", "Z", "N"),
          ("conjugate", "X", "S"), ("conjugate", "X", "N"),
          ("conjugate", "Z", "E"), ("conjugate", "Z", "W")]


@pytest.mark.smoke
@pytest.mark.parametrize("typ,letter,face", _TABLE,
                         ids=[f"{t}-{l}-{f}" for t, l, f in _TABLE])
def test_attach_face_full_distance_d3(typ, letter, face):
    # minimal legal geometry: one corridor cell on the measured face,
    # the partner patch straight beyond it (zero elbows)
    dx, dy = _FACE[face]
    qc, cc, pc = (2, 2), (2 + dx, 2 + dy), (2 + 2 * dx, 2 + 2 * dy)
    seam_ew = (dx != 0)
    p_or = ("X_vertical" if letter == "X" else "X_horizontal") if seam_ew \
        else ("X_horizontal" if letter == "X" else "X_vertical")
    px = [_spec("Q", *qc, _SPEC_OF[typ]), _spec("P", *pc, p_or)]
    exp = _build(px, letter, [cc])
    assert _dist(exp) == D


@pytest.mark.smoke
@pytest.mark.parametrize("row", [0, 2], ids=["S", "N"])
def test_long_corridor_conjugate_x_face_full_distance(row):
    # the historically-suspected (conjugate, X, S) face with a LONG legal
    # corridor (3 straight cells) and its N mirror: both full distance
    px = [_spec("Q", 1, 1, "X_horizontal"), _spec("P", 4, row, "X_vertical")]
    exp = _build(px, "X", [(1, row), (2, row), (3, row)])
    assert _dist(exp) == D


@pytest.mark.slow
@pytest.mark.parametrize("row", [0, 2], ids=["S", "N"])
def test_long_corridor_conjugate_x_face_d5(row):
    d = 5
    px = [_spec("Q", 1, 1, "X_horizontal", d=d),
          _spec("P", 4, row, "X_vertical", d=d)]
    exp = _build(px, "X", [(1, row), (2, row), (3, row)], d=d)
    assert _dist(exp) == d
