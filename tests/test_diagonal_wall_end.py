"""Single-cell mixed checks at a WALL END — the ``_M_SLOTS`` variant key.

A colour domain wall ends where the narrower of the two regions it separates
ends.  The last mixed check of such a wall has no SW foot, and reading the
``_M_SLOTS`` variant off whatever foot happens to be (west,south)-most flips
the wall's polarity exactly there: the check's NW gate lands on slot 4, the
slot every plain plaquette already owns on its own SW corner (K&F Fig 4 gives
Z-SW = X-SW = 4), so the fixed compact-7 validator rejects the layout —

    fixed compact-7 schedule: slot 4 on qubit (111, 41) is used by both
    Z-check@(112, 42) and M-check@(112, 40)

The geometry is not exotic.  It appears wherever the data region NORTH of a
horizontal wall reaches further west than the region south of it: our own
seam-column router builds it for a T-junction whose corridor row is
segment-recoloured (``flip_cells``), and every compact DASCOT layout replayed
by ``experiments/dascot_bridge.py`` (ghz_n40, bv_n70) hits it at an attach
seam whose corridor continues past the attaching patch.
"""
import pytest

from circls.core.multi_patch_coupler import (_bent_plaquettes, route_and_build,
                                        PatchSpec, origin_of)
from lightstim.qec_code.surface_code.rotated.diagonal_se import (
    DiagonalSurfaceCodeExtractionBlock, FIXED_SLOTS, _M_SLOTS, _m_variant)

pytestmark = pytest.mark.smoke

_NE, _NW, _SE, _SW = (1, 1), (-1, 1), (1, -1), (-1, -1)
D = 3

#: the failing neighbourhood of the ghz_n40 replay, shifted to small coords:
#: a WIDE re-typed region on top of a NARROWER plain one, sharing one ancilla
#: row.  The wall's west end (ancilla (4, 4)) loses its SW foot.
_NORTH = {(x, y) for x in (1, 3, 5, 7) for y in (5, 7)}
_SOUTH = {(x, y) for x in (5, 7) for y in (1, 3)}


def _sgn(v):
    return (v > 0) - (v < 0)


def _corners(ch):
    """A layout check as {corner delta: foot Pauli}."""
    sx, sy = ch['syn']
    return {(_sgn(q[0] - sx), _sgn(q[1] - sy)): P
            for q, P in ch['pauli'].items()}


def _wall_ends(checks):
    """Mixed checks with no SW foot whose two north feet agree — the pattern
    the (west,south)-most-foot key used to mis-read."""
    out = []
    for ch in checks:
        if ch.get('kf') is not None or len(set(ch['pauli'].values())) < 2:
            continue
        p = _corners(ch)
        if _SW not in p and _NW in p and _NE in p and p[_NW] == p[_NE]:
            out.append((tuple(ch['syn']), p))
    return out


class _Sys:
    """The slice of ``QECSystem`` the diagonal SE block reads, over an
    explicit (data, checks) layout."""

    def __init__(self, data, checks):
        assert not any(ch.get('kf') for ch in checks)
        coords = sorted(data) + sorted({tuple(c['syn']) for c in checks})
        self.qubit_coords = dict(enumerate(coords))
        self.index_map = {c: i for i, c in self.qubit_coords.items()}
        self.stabilizers = {
            k: {'type': c['type'], 'syn_coord': tuple(c['syn']),
                'syn_idx': self.index_map[tuple(c['syn'])],
                'data_indices': [self.index_map[q] for q in c['pauli']],
                'pauli': {self.index_map[q]: P for q, P in c['pauli'].items()}}
            for k, c in enumerate(checks)}
        self.active_stabilizer_indices = sorted(self.stabilizers)
        self.num_qubits = len(coords)


def test_m_variant_keeps_the_sw_foot_when_it_is_there():
    # every foot present: the key is the SW foot, for both wall directions
    assert _m_variant({_SW: 'X', _NW: 'X', _NE: 'Z', _SE: 'Z'}) == 'X'   # |
    assert _m_variant({_SW: 'Z', _NW: 'Z', _NE: 'X', _SE: 'X'}) == 'Z'   # |
    assert _m_variant({_SW: 'X', _NW: 'Z', _NE: 'Z', _SE: 'X'}) == 'X'   # --
    assert _m_variant({_SW: 'Z', _NW: 'X', _NE: 'X', _SE: 'Z'}) == 'Z'   # --
    # a missing NW / NE / SE foot leaves the SW foot in charge
    assert _m_variant({_SW: 'X', _NW: 'Z', _SE: 'X'}) == 'X'
    assert _m_variant({_SW: 'X', _SE: 'Z'}) == 'X'
    # no west foot at all: the south row's SE foot carries the site's letter
    assert _m_variant({_NE: 'X', _SE: 'Z'}) == 'Z'
    # a vertical wall with no south feet: the west column is uniform
    assert _m_variant({_NW: 'Z', _NE: 'X'}) == 'Z'


def test_m_variant_at_a_wall_end_reads_the_site_not_the_foot():
    # horizontal wall, SW foot missing: both north feet agree, so the south
    # row (and with it the absent SW site) carries the other letter — NOT the
    # NW foot's letter, which is what the plain (west,south)-most rule read
    assert _m_variant({_NW: 'X', _NE: 'X', _SE: 'Z'}) == 'Z'
    assert _m_variant({_NW: 'Z', _NE: 'Z', _SE: 'X'}) == 'X'
    # ... and that is the variant whose NW slot is off the plaquettes' SW slot
    assert FIXED_SLOTS['Z'][_SW] == FIXED_SLOTS['X'][_SW] == 4
    assert _M_SLOTS[0][(_NW, 'X')] == 4                  # the old, colliding read
    assert _M_SLOTS[1][(_NW, 'X')] != 4


def test_wall_end_region_schedules():
    checks = _bent_plaquettes(_NORTH | _SOUTH, _NORTH, 0)
    ends = _wall_ends(checks)
    assert ends == [((4, 4), {_NW: 'X', _NE: 'X', _SE: 'Z'})]
    blk = DiagonalSurfaceCodeExtractionBlock(_Sys(_NORTH | _SOUTH, checks))
    assert blk.coupling_slots == 7
    # the wall-end check's NW gate must dodge the Z plaquette above it, whose
    # SW gate owns slot 4 on the very same data qubit
    gathered, kf = blk._gather()
    _T, slot_maps = blk._slot_maps(gathered, kf)
    tm = next(t for ch, t in zip(gathered, slot_maps) if ch['syn'] == (4, 4))
    nw = next(t for di, t in tm.items()
              if tuple(blk.system.qubit_coords[di]) == (3, 5))
    assert nw + 1 == _M_SLOTS[1][(_NW, 'X')] == 7


def test_our_router_builds_a_wall_end_check():
    # Not a DASCOT-only shape: our own seam-column router builds it for a
    # T-junction (A west, B east, C south of the corridor cell) whose corridor
    # row is moved into the flipped gauge.  The corridor reaches west past the
    # seam row that attaches C, so the wall between them ends on a check with
    # no SW foot — under the old key this layout raised
    # "slot 4 on qubit (7, 9) ... Z-check@(8, 10) and M-check@(8, 8)".
    def spec(nm, a, b, o):
        return PatchSpec(nm, origin_of(a, b, D, seam=True), D, o)

    r = route_and_build(
        [spec("A", 0, 1, "X_vertical"), spec("B", 2, 1, "X_vertical"),
         spec("C", 1, 0, "X_vertical")],
        [("A", "X"), ("B", "X"), ("C", "Z")], seam=True, route=[(1, 1)],
        bus="Z", conj_names=frozenset({"A", "B"}), flip_cells=[(1, 1)])
    assert r.status == 'ok'
    assert _wall_ends(r.layout.checks) == [((8, 8), {_NW: 'X', _NE: 'X',
                                                     _SE: 'Z'})]
    # the whole merged layout schedules under the fixed compact-7 table
    blk = DiagonalSurfaceCodeExtractionBlock(_Sys(r.layout.data,
                                                  r.layout.checks))
    assert blk.coupling_slots == 7
