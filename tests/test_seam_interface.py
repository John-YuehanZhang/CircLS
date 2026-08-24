"""Stage-C slice 1: the per-seam eight-row classifier (seam_interface).

Self-checks against the two verified production families:
  * the snake d3 step (q2 conj wall on a vertical seam, q3 standard on a
    horizontal seam) must classify to the VERIFIED rows #6 / #7 under the
    gauge rule;
  * a plain same-phase merge must classify both seams to row #1 at g = 0.
"""
from types import SimpleNamespace

import pytest

from circls.core.routed_multi_patch_ls import origin_of
from circls.core.multi_patch_coupler import \
    place_patch
from lightstim.qec_code.surface_code.rotated.seam_interface import (
    SeamInterface, choose_bus_conventions, choose_bus_gauge,
    choose_colour_reference, classify_attach_seam, patch_phase)

pytestmark = pytest.mark.smoke

D = 3


def _phase(a, b, orient):
    nat = place_patch(SimpleNamespace(origin=origin_of(a, b, D, seam=True),
                                      distance=D, orientation=orient))
    return patch_phase(nat["checks"])


def test_snake_d3_rows_match_verified_family():
    # M(q2^Z, q3^Z): q2 conj-registered (colour swap), wall on its VERTICAL
    # seam q2|(2,0); q3 standard through its HORIZONTAL seam q3|(0,2).
    # Production (verified full-distance): q2 = #6 mixed wall, q3 = #7
    # recoloured column, corridor in the mixed set.
    ph_q2 = _phase(1, 0, "X_horizontal") ^ 1      # conj = colour swap
    ph_q3 = _phase(0, 1, "X_vertical")
    ref = choose_colour_reference([ph_q2, ph_q3])
    i2 = classify_attach_seam("q2", (1, 0), (2, 0), "Z", "Z", ph_q2, ref)
    i3 = classify_attach_seam("q3", (0, 1), (0, 2), "Z", "Z", ph_q3, ref)
    assert i2.is_wall and i2.axis == 'vertical'
    assert not i3.is_wall and i3.axis == 'horizontal'
    g = choose_bus_gauge([i2, i3])
    assert g == 1                     # vertical wall -> mixed set
    assert i2.row(g) == 6
    assert i3.row(g) == 7


def test_plain_same_phase_merge_is_row1():
    # both textbook, same phase, measuring the bus letter: plain everywhere
    ph_bt = _phase(2, 0, "X_vertical")
    ph_c = _phase(4, 1, "X_vertical")
    ref = choose_colour_reference([ph_bt, ph_c])
    ib = classify_attach_seam("Bt", (2, 0), (3, 0), "X", "X", ph_bt, ref)
    ic = classify_attach_seam("C", (4, 1), (3, 1), "X", "X", ph_c, ref)
    g = choose_bus_gauge([ib, ic])
    assert g == 0
    assert ib.row(g) == 1 and ic.row(g) == 1


def test_gauge_toggle_pairs_rows():
    # the spec's wholesale toggle: #1<->#7 and #4<->#6
    for ld, cd, r0, r1 in [(False, False, 1, 7), (False, True, 4, 6),
                           (True, False, 6, 4), (True, True, 7, 1)]:
        i = SeamInterface("t", (0, 0), (1, 0), 'vertical', ld, cd)
        assert (i.row(0), i.row(1)) == (r0, r1)
        assert i.is_wall == (ld != cd)


def test_bus_convention_pair_is_fixed_up_front():
    # the pair (colour, letter): letter = majority measured (tie -> X),
    # explicit bus overrides; colour = majority phase (tie -> 0)
    tgt = [("Q", "X"), ("P", "Z")]
    assert choose_bus_conventions(tgt, [0, 0]) == ("X", 0)
    assert choose_bus_conventions(tgt, [0, 0], bus="Z") == ("Z", 0)
    assert choose_bus_conventions([("A", "Z"), ("B", "Z")], [1, 1]) \
        == ("Z", 1)
