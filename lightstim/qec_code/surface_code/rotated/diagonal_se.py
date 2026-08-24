"""Diagonal syndrome-extraction schedule (Kishony & Fowler, *Surface code
off-the-hook: diagonal syndrome-extraction scheduling*).

One globally uniform gate ORDER for every patch shape and orientation —
Z-plaquette corners coupled NW→SE→NE→SW, X-plaquette SW→NE→SE→NW (y-up frame) —
so every ancilla hook lands on a DIAGONAL data pair and never aligns with a
horizontal or vertical logical path.  This is what keeps boundary-deformation
steps (grow / move-corners / shrink) fault-tolerant: their logical
representatives bend, and an exhaustive sweep showed every N/Z zigzag variant
caps the Z-memory distance of the grow step at 2, while this schedule reaches
full distance d for both bases at d = 3 and 5 (``tests/test_grow_patch.py``).

Slot assignment: the FIXED compact-7 table of K&F Fig 4 — no dynamic search.

* Plain plaquettes (and boundary lobes, restricted to present corners) read
  ``FIXED_SLOTS`` at their corner deltas: Z at slots 1-4, X at slots 4-7 over
  the 7-slot coupling window.  Round depth: R | H | 7 | H | M = 11 ticks.
* Single-cell mixed checks (corridor/recoloured-column stabilizers; not in
  the paper) read TWO fixed variant tables (``_M_SLOTS``) keyed by the Pauli
  of the check's SOUTH-WEST corner SITE (``_m_variant``: the SW foot when it
  is there, else the letter that site must carry given the wall's direction
  through the cell) — the single-cell twin of the wall's '+'/'-' alternation,
  derived offline by constraint solving and selected by the distance ladder
  (see the table's comment).
* K&F relay (stretched) checks use the Fig 4(c) program verbatim via
  ``_kf_events``; adjacent dominoes alternate between the two Fig 4 variants
  (offset 0 / +2, i.e. the paper's slot-1 and slot-3 anchored circuits).

The K&F Sec. II constraint (two checks sharing two anticommuting corners keep
a consistent relative gate order) and (slot, qubit) collision-freedom are now
VALIDATED, not searched for: ``_validate_fixed`` raises with the offending
coordinates if a layout violates the table, instead of packing around it.

One exception to the uniform table: rounds that cover a live |Y>-born patch
must run that patch's interior checks on GIDNEY's order (SE_block contract:
the Y-transition chunk is a gate-for-gate port of Gidney's round, so its hook
errors only line up with neighbouring rounds that use his order).  Pass those
checks' uids as ``gidney_uids`` and the block embeds Gidney's 4-tick order in
the 7-slot window, deriving custom orders for the checks his order cannot
keep (see ``_gidney_slot_maps``).  A vertical relay wall resting its near
(smaller-y) side on such a patch additionally mirrors that side's foot
subslots (``y_coords``; see ``_gather_kf``).
"""
import itertools
import re
import time

import stim

from lightstim.qec_code.surface_code.rotated.SE_block import (
    RotatedSurfaceCodeExtractionBlock as _SE_BLOCK)

#: Gidney's 4-tick interaction order (arXiv:2302.07395), the (X, Z) canonical
#: corner delta per tick — single source of truth is the SE_block table the
#: Y-transition rounds run on.
_GIDNEY_ORDER = _SE_BLOCK.SCHEDULES['gidney']

_NE, _NW, _SE, _SW = (1, 1), (-1, 1), (1, -1), (-1, -1)

#: K&F diagonal gate orders (y-up corner deltas, data − syndrome)
DIAGONAL_ORDERS = {'Z': (_NW, _SE, _NE, _SW), 'X': (_SW, _NE, _SE, _NW)}

#: K&F Fig 4 fixed slots (1-indexed over the 7-slot window): Z gates at 1-4,
#: X gates at 4-7.  These ARE the numbers printed on the paper's plaquettes;
#: ``_slot_maps`` subtracts 1 to get 0-indexed circuit slots.
FIXED_SLOTS = {
    'Z': {_NW: 1, _NE: 3, _SW: 4, _SE: 2},
    'X': {_NW: 7, _NE: 5, _SW: 4, _SE: 6},
}


#: K&F relay-template program (Fig 39 / interface plaquette): per-check slot
#: offsets for the two-aux + shared-relay realization of a wide mixed check.
#: A = flag aux (RX -> MZ, couples the Z feet via CZ), B = syndrome aux
#: (RZ -> MX, couples the X feet via CX), S = shared relay (RZ -> MZ).
#: p0 CX(A,S); p1 CZ(A,fz1)+CX(S,B); p2 CZ(A,fz2)+CX(B,fx1);
#: p3 CX(S,A)+CX(B,fx2); p4 CX(B,S).  This IS the Fig 4(c) circuit: offset 0
#: reproduces the paper's slot-1 anchored variant (feet at slots 2-4,
#: 1-indexed) and offset +2 the slot-3 anchored one (feet at 4-6) — the two
#: stretched-stabilizer schedules that alternate along the interface in
#: Fig 4(a,b).  Adjacent dominoes share two anticommuting feet, so a uniform
#: offset would violate the K&F Sec. II consistency constraint; the builders
#: therefore alternate '+'/'-' along the wall.
_KF_OFFSET = {'+': 0, '-': 2}

#: Single-cell mixed (corridor/recoloured-column) checks: TWO fixed variant
#: tables keyed by the Pauli of the check's SOUTH-WEST corner SITE (see
#: ``_m_variant``) — the single-cell twin of the wall's '+'/'-' alternation.  A
#: recoloured column's checks alternate their foot Paulis row by row and
#: adjacent checks share two anticommuting feet with each other AND with
#: the neighbouring bulk plaquettes; one per-own-Pauli table cannot order
#: all those pairs consistently (K&F Sec. II).  These tables were derived
#: OFFLINE by constraint solving over five verified corridor layouts
#: (straight/stacked row-4, an obstacle-detour bent corridor, an L
#: corridor) and selected by the distance ladder: all scenarios reach
#: p=0-clean full graphlike distance under the diagonal schedule.
#: (Entries never exercised by a real check pattern carry the solver's
#: canonical values.)  Keys: (corner, foot Pauli) -> 1-indexed slot.
_M_SLOTS = (
    # variant 0: the check's SOUTH-WEST corner site is X-support
    {(_NW, 'X'): 4, (_NW, 'Z'): 1, (_NE, 'X'): 1, (_NE, 'Z'): 5,
     (_SW, 'X'): 3, (_SW, 'Z'): 1, (_SE, 'X'): 4, (_SE, 'Z'): 2},
    # variant 1: it is Z-support
    {(_NW, 'X'): 7, (_NW, 'Z'): 4, (_NE, 'X'): 3, (_NE, 'Z'): 1,
     (_SW, 'X'): 1, (_SW, 'Z'): 5, (_SE, 'X'): 6, (_SE, 'Z'): 4},
)

_FLIP_P = {'X': 'Z', 'Z': 'X'}

#: The ``<type>-check@(x, y)`` labels ``_validate_fixed`` embeds in its
#: messages, read back by the gidney-override drop cascade to learn WHICH
#: checks a violation involves (apparatus labels say ``apparatus@`` and
#: qubit coordinates carry no ``check@`` prefix, so neither can match).
_OVERRIDE_LABEL_RE = re.compile(r"check@\((-?\d+), (-?\d+)\)")


def _m_variant(pauli_at_corner):
    """Which ``_M_SLOTS`` table a single-cell mixed check reads: 0 when its
    SOUTH-WEST corner SITE carries X-support, 1 when it carries Z.

    ``pauli_at_corner`` maps present corner deltas to that foot's Pauli.  The
    key is the *site*, not merely the foot: a check on the edge of the data
    region can be missing its SW foot, and the two tables are a wall-polarity
    alternation — reading the variant off whatever foot happens to be
    (west,south)-most instead flips the polarity at such a wall end and puts
    the check's NW gate on slot 4, which every plain plaquette already owns on
    its own SW corner (K&F Fig 4: Z-SW = X-SW = 4).

    A foot's Pauli is the cell's base letter, flipped on the re-typed side of
    the wall, so the absent SW site's letter follows from the wall's direction
    through the cell:

    * both NORTH feet present and EQUAL — the wall runs east-west between the
      rows, the whole south row carries the other letter, so the SW site is
      the flip of the north feet (equivalently the SE foot's letter);
    * otherwise the west column is uniform across the cell (a vertical or
      corner wall), so the NW foot's letter already is the SW site's letter.

    With every foot present this is exactly the old (west,south)-most-foot
    lookup, and so are the remaining boundary cases (the fallback order
    SW, NW, SE, NE is the lexicographic order of the corner coordinates).
    """
    p = pauli_at_corner
    if _SW in p:
        return p[_SW]
    if _NW in p and _NE in p and p[_NW] == p[_NE]:
        return _FLIP_P.get(p[_NW], p[_NW])
    for c in (_NW, _SE, _NE):
        if c in p:
            return p[c]
    # no diagonal corner at all: pick deterministically and let the per-corner
    # lookup raise the informative "non-diagonal corner delta" error
    return p[min(p)]


def _kf_events(ch):
    """The (slot, gate, pair) events of one K&F relay check.  ``ch`` needs
    'offset', flag/shared/syn coords 'A'/'S'/'B', and per-side feet lists
    'fa' (A's side) / 'fb' (B's side), each a list of (subslot, coord, gate)
    with subslot in {0, 1} (the foot's POSITIONAL slot within the side's
    pair — like the paper's boundary lobes, a weight-1 side keeps the slot
    its corner position dictates, it does not slide to the first one) and
    gate 'CX' (X-support foot, aux is control) or 'CZ' (Z-support).
    The aux split is GEOMETRIC (each aux couples the feet on its own row),
    not Pauli-based: back-propagation gives MX(B) = the full check and keeps
    MZ(A)/MZ(S) deterministic for ANY foot-Pauli pattern, so the same
    template serves mixed dominoes (2+2 split) and uniform same-type
    dominoes (all-X or all-Z)."""
    o = ch['offset']
    A, S, B = ch['A'], ch['S'], ch['B']

    def foot(aux, coord, gate):
        return (gate, (aux, coord) if gate == 'CX' else (coord, aux))

    ev = [(o + 0, 'CX', (A, S)), (o + 1, 'CX', (S, B))]
    for sub, coord, gate in ch['fa']:
        ev.append((o + 1 + sub,) + foot(A, coord, gate))
    for sub, coord, gate in ch['fb']:
        ev.append((o + 2 + sub,) + foot(B, coord, gate))
    ev.extend([(o + 3, 'CX', (S, A)), (o + 4, 'CX', (B, S))])
    return ev


def _validate_fixed(placements):
    """Assert the fixed table produced a legal schedule — raise, don't search.

    ``placements``: list of ``(label, pauli_at, {coord: slot})`` covering every
    check's data couplings AND the K&F relay apparatus gates.  Checks the two
    invariants the old backtracking packer used to guarantee:

    * no (slot, qubit) cell is used twice;
    * K&F Sec. II — two checks sharing >= 2 anticommuting corners keep a
      consistent relative gate order on them (apparatus cells carry no Pauli
      and are exempt: ``pauli_at`` simply has no entry for them).
    """
    occupied = {}
    for label, _, cm in placements:
        for q, t in cm.items():
            if not 0 <= t < 7:
                raise RuntimeError(
                    f"fixed compact-7 schedule: {label} places qubit {q} at "
                    f"slot {t + 1}, outside the 7-slot window")
            other = occupied.setdefault((t, q), label)
            if other != label:
                raise RuntimeError(
                    f"fixed compact-7 schedule: slot {t + 1} on qubit {q} is "
                    f"used by both {other} and {label} — this layout violates "
                    f"the K&F Fig 4 table (fix the wall spec, do not repack)")
    by_corner = {}
    for k, (_, pauli_at, cm) in enumerate(placements):
        for q in cm:
            if q in pauli_at:
                by_corner.setdefault(q, []).append(k)
    pairs = {}
    for q, ks in by_corner.items():
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                a, b = ks[i], ks[j]
                if placements[a][1][q] != placements[b][1][q]:
                    pairs.setdefault((a, b), []).append(q)
    for (a, b), qs in pairs.items():
        if len(qs) < 2:
            continue
        la, _, ca = placements[a]
        lb, _, cb = placements[b]
        rels = {ca[q] < cb[q] for q in qs}
        if len(rels) > 1:
            raise RuntimeError(
                f"fixed compact-7 schedule: K&F Sec. II consistency violated "
                f"between {la} and {lb} on shared anticommuting corners {qs}")


class DiagonalSurfaceCodeExtractionBlock:
    """Drop-in alternative to :class:`RotatedSurfaceCodeExtractionBlock` using the
    diagonal schedule.  Same contract: ``self.circuit`` is one noiseless SE round,
    last instruction is the syndrome measurement.

    ``gidney_uids`` (optional): stabilizer uids that must couple their corners
    in GIDNEY's 4-tick order instead of the K&F table — the interior checks of
    a live |Y>-born patch (SE_block contract: the rounds surrounding a
    Y-transition round must run his order or the transition costs one unit of
    fault distance).  Empty/None leaves the block byte-identical to the plain
    table.  See ``_gidney_slot_maps`` for the derivation rule.

    ``y_coords`` (optional): the coordinates owned by the |Y>-born patches —
    scopes the kf-wall NEAR-SIDE MIRROR (see ``_gather_kf``): a vertical
    relay domino whose lo (smaller-y) side rests BOTH feet on a Y wedge
    couples that pair in the mirrored subslot order.  Empty/None keeps every
    wall on the verbatim Fig 4 order.
    """

    def __init__(self, system, gidney_uids=None, y_coords=None):
        self.system = system
        self.gidney_uids = frozenset(gidney_uids) if gidney_uids else frozenset()
        self.y_coords = frozenset(y_coords) if y_coords else frozenset()
        self.circuit = stim.Circuit()
        self.coupling_slots = None    # set by _build_circuit: always 7 (K&F Fig 4)
        self._build_circuit()

    def _gather(self):
        """Gather ALL active checks — pure X/Z plaquettes AND single-cell mixed
        checks (type 'MIXED'/'M', e.g. corridor stabilizers).  Corners are
        classified by the SIGN of their offset from the syn coord and then read
        the fixed table: a plain check its own type's table, a mixed check each
        FOOT's own Pauli's table (gates stay per-foot: CX for X-support / CZ
        for Z-support).  A check whose feet do not sit on four distinct
        diagonal corners cannot be expressed in the table — raise, don't guess."""
        system = self.system

        def sgn(v):
            return (v > 0) - (v < 0)

        checks, kf_checks = [], []
        for uid in sorted(system.active_stabilizer_indices):
            st = system.stabilizers[uid]
            if st.get('kf') is not None:
                kf_checks.append(self._gather_kf(st))
                continue
            t = st.get('type')
            typ = t if t in ('X', 'Z') else 'M'
            sc = st['syn_coord']
            deltas = {}
            pauli_of = {}
            for di in st['data_indices']:
                dc = system.qubit_coords[di]
                dxy = (sgn(dc[0] - sc[0]), sgn(dc[1] - sc[1]))
                if dxy in deltas:
                    raise RuntimeError(
                        f"fixed compact-7 schedule: check at {tuple(sc)} has "
                        f"two data qubits ({tuple(system.qubit_coords[deltas[dxy]])} "
                        f"and {tuple(dc)}) on the same {dxy} corner direction — "
                        f"a wide check must carry 'kf' relay metadata, the "
                        f"plain table cannot express it")
                deltas[dxy] = di
                pauli_of[di] = (st['pauli'][di] if typ == 'M' else typ)
            pauli_at = {tuple(system.qubit_coords[di]): p
                        for di, p in pauli_of.items()}
            checks.append({'type': typ, 'syn': tuple(sc), 'syn_idx': st['syn_idx'],
                           'deltas': deltas, 'pauli_of': pauli_of,
                           'pauli_at': pauli_at})
        return checks, kf_checks

    def _gather_kf(self, st):
        """One K&F relay check: resolve the apparatus coords (flag A, shared S,
        syn B = ``syn_coord``) and split the feet GEOMETRICALLY — each foot
        belongs to the aux one diagonal step away (its own gap side).  Both
        Fig 4 orientations are supported and told apart by the apparatus
        axis: A and B in the same COLUMN = vertical domino (feet above and
        below the aux band), same ROW = horizontal domino (feet left and
        right).  Each foot gets the POSITIONAL subslot of the Fig 4 table —
        vertical: Z side east first, X side west first (Fig 4b); horizontal:
        both sides north first on the '-' variant, south first on '+'
        (Fig 4a/c) — and carries its own gate: CX for X support, CZ for Z
        support.  One exception: a vertical domino's lo side MIRRORS its
        subslot order when both its feet rest on a |Y>-born wedge patch
        (``y_coords``; see the near-side-mirror comment in ``side``)."""
        system = self.system

        def sgn(v):
            return (v > 0) - (v < 0)

        kf = st['kf']
        B = tuple(st['syn_coord'])
        A = tuple(kf['flag'])
        S = tuple(kf['shared'])
        if A[0] == B[0] and A[1] != B[1]:
            horizontal = False
        elif A[1] == B[1] and A[0] != B[0]:
            horizontal = True
        else:
            raise RuntimeError(
                f"K&F relay check at {B}: apparatus A={A} and B={B} share "
                f"neither a row nor a column — malformed wall spec")
        axis = 0 if horizontal else 1     # feet split along this coordinate
        feet = {A: {}, B: {}}      # aux -> {delta-from-aux: (coord, pauli)}
        for di in st['data_indices']:
            dc = tuple(system.qubit_coords[di])
            p = st['pauli'][di]
            ax = A if abs(dc[axis] - A[axis]) == 1 else B
            if dc in (A, S, B) or abs(dc[0] - ax[0]) != 1 or abs(dc[1] - ax[1]) != 1:
                raise RuntimeError(
                    f"K&F relay check at {B}: foot {dc} is not one diagonal "
                    f"step from its aux {ax} (apparatus A={A}, S={S}, B={B}) — "
                    f"malformed wall spec")
            dxy = (sgn(dc[0] - ax[0]), sgn(dc[1] - ax[1]))
            if dxy in feet[ax]:
                raise RuntimeError(
                    f"K&F relay check at {B}: feet {feet[ax][dxy][0]} and {dc} "
                    f"occupy the same {dxy} corner of aux {ax} — malformed "
                    f"wall spec")
            feet[ax][dxy] = (dc, p)

        offset = _KF_OFFSET[kf['orient']]

        def side(ax):
            # K&F Fig 4 foot subslots for the stretched stabilizers.  The
            # subslot is POSITIONAL (which end of the side's short edge), so
            # a weight-1 side keeps the slot its corner dictates — exactly
            # like the paper's boundary lobes.
            # * vertical dominoes (Fig 4b): an X-support side couples its
            #   WEST foot first (time runs west -> east on the red halves),
            #   a Z-support side its EAST foot first.  Sides are
            #   Pauli-uniform in every wall family we build; a hypothetical
            #   mixed side gets the X rule.
            # * horizontal dominoes (Fig 4a/c): the order is keyed by the
            #   VARIANT, not the Pauli — the slot-3-anchored '-' circuit
            #   couples both sides north first (Fig 4c: tl=4,bl=5 / tr=5,
            #   br=6), the slot-1-anchored '+' couples both south first.
            # * NEAR-SIDE MIRROR (vertical dominoes only): when the side is
            #   the LO aux (smaller y) and BOTH its feet are data qubits of
            #   a |Y>-born wedge patch (``y_coords``), the pair couples in
            #   the MIRRORED subslot order — Z side WEST first, X side EAST
            #   first.  This is the unique uniform wall re-parametrization
            #   whose gidney solve puts the wedge-corner Z-check's NW foot
            #   mid-window (risk tier 0), removing the last weight-4
            #   single-foot-hook chain class; the far side and all offsets
            #   stay Fig 4b verbatim, so the fixed-corridor environment the
            #   ``_M_SLOTS`` tables were derived in is untouched.  The
            #   Y-wedge scoping is MANDATORY: an unscoped mirror collides
            #   where the wall abuts plain-K&F checks (measured, e.g.
            #   X-check@(38, 56) vs kf-check@(38, 60) on (39, 57)).
            other = B if ax == A else A
            mirror = (not horizontal and ax[1] < other[1]
                      and len(feet[ax]) == 2
                      and all(c in self.y_coords
                              for c, _ in feet[ax].values()))
            out = []
            for coord, p in feet[ax].values():
                if horizontal:
                    north = coord[1] > ax[1]
                    north_first = (offset == 2)
                    sub = (0 if north else 1) if north_first else (1 if north else 0)
                else:
                    paulis = {q for _, q in feet[ax].values()}
                    east_first = (paulis == {'Z'}) ^ mirror
                    east = coord[0] > ax[0]
                    sub = (0 if east else 1) if east_first else (1 if east else 0)
                out.append((sub, coord, 'CX' if p == 'X' else 'CZ'))
            return sorted(out)

        return {'A': A, 'S': S, 'B': B, 'fa': side(A), 'fb': side(B),
                'offset': offset,
                'syn_idx': st['syn_idx'],
                'flag_idx': system.index_map[A],
                'shared_idx': system.index_map[S],
                'pauli_at': {tuple(system.qubit_coords[di]): st['pauli'][di]
                             for di in st['data_indices']}}

    def _slot_maps(self, checks, kf_checks=()):
        """Fixed compact-7 assignment (K&F Fig 4) — a table lookup, no search.

        Plain X/Z checks (any corner subset) read ``FIXED_SLOTS[type]``;
        single-cell mixed checks read each foot's OWN Pauli at that foot's
        corner, out of the ``_M_SLOTS`` variant their SW corner site keys
        (:func:`_m_variant`); K&F relay checks are already fully determined by
        ``_kf_events``.  Everything is then validated once.

        When ``gidney_uids`` is set the checks it names are overridden onto
        Gidney's order instead (``_gidney_slot_maps``); without it this path
        is untouched.
        """
        if self.gidney_uids:
            return self._gidney_slot_maps(checks, kf_checks)
        return self._finish_slot_maps(
            checks, kf_checks, [self._table_tm(ch) for ch in checks])

    def _table_tm(self, ch):
        """One gathered check's ``{data_index: 0-indexed slot}`` from the
        fixed K&F tables (the per-check body of the plain ``_slot_maps``)."""
        system = self.system
        tm = {}
        # single-cell mixed checks read the _M_SLOTS variant table keyed
        # by the SOUTH-WEST corner SITE's Pauli (see _m_variant)
        if ch['type'] == 'M':
            key = _m_variant({dxy: ch['pauli_of'][di]
                              for dxy, di in ch['deltas'].items()})
            m_tab = _M_SLOTS[0 if key == 'X' else 1]
        for dxy, di in ch['deltas'].items():
            pauli = ch['pauli_of'][di] if ch['type'] == 'M' else ch['type']
            if pauli not in FIXED_SLOTS:
                raise RuntimeError(
                    f"fixed compact-7 schedule: check at {ch['syn']} has "
                    f"{pauli!r} support on qubit "
                    f"{tuple(system.qubit_coords[di])} — the K&F table "
                    f"only covers X and Z")
            if dxy not in FIXED_SLOTS[pauli]:
                raise RuntimeError(
                    f"fixed compact-7 schedule: check at {ch['syn']} has a "
                    f"non-diagonal corner delta {dxy} (qubit "
                    f"{tuple(system.qubit_coords[di])}) — the K&F table "
                    f"only covers diagonal corners")
            if ch['type'] == 'M':
                tm[di] = m_tab[(dxy, pauli)] - 1
            else:
                tm[di] = FIXED_SLOTS[pauli][dxy] - 1
        return tm

    def _finish_slot_maps(self, checks, kf_checks, slot_maps):
        """Register every placement (data couplings, ancilla busy-slots, K&F
        apparatus), validate the whole round once, and return the
        ``(coupling_slots, slot_maps)`` pair (the tail of the plain
        ``_slot_maps``)."""
        system = self.system
        placements = []
        for ch, tm in zip(checks, slot_maps):
            placements.append(
                (f"{ch['type']}-check@{ch['syn']}", ch['pauli_at'],
                 {tuple(system.qubit_coords[di]): t for di, t in tm.items()}))
            # the check's own ancilla is busy at every one of its coupling
            # slots: register those cells too, so a collision against a K&F
            # apparatus qubit (or anything else) on the ancilla coord raises
            for t in tm.values():
                placements.append(
                    (f"{ch['type']}-check@{ch['syn']}:anc", {}, {ch['syn']: t}))
        for ch in kf_checks:
            feet_cm = {}
            for slot, gate, (u, v) in _kf_events(ch):
                for q in (u, v):
                    if q in (ch['A'], ch['S'], ch['B']):
                        # apparatus qubits recur across slots: one placement
                        # per event, carrying no Pauli (collision check only)
                        placements.append(
                            (f"kf-apparatus@{ch['B']}:t{slot}", {}, {q: slot}))
                    else:
                        feet_cm[q] = slot
            placements.append(
                (f"kf-check@{ch['B']}", ch['pauli_at'], feet_cm))
        _validate_fixed(placements)

        return 7, slot_maps

    def _gidney_slot_maps(self, checks, kf_checks):
        """Compact-7 slot maps with the ``gidney_uids`` checks on GIDNEY's
        order — the live-|Y> override.  The RULE, deterministic end to end
        (validated on toffoli/bell/fredkin/simon at d = 5: all four recover
        full graphlike distance, and the no-override path is byte-identical):

        1. Try each base offset b = 0..3 in turn: every override check
           couples its corners over slots b..b+3 in Gidney's tick order
           (``_GIDNEY_ORDER``, read through the owner patch's orientation).
        2. At one base, the override checks Gidney's order cannot keep are
           found by a DROP CASCADE: validate the round, drop the override
           checks named by the violation, re-validate — so only checks that
           actually appear in a conflict lose the pure Gidney order.
        3. The dropped ("forced") checks get SOLVED custom orders: a
           backtracking search over injective 7-slot corner assignments,
           consistent (slot/qubit collision-freedom + K&F Sec. II) with the
           kept Gidney checks, the plain-table checks and the K&F relay
           apparatus.  Candidates per check are ranked, best first, by

           (a) hook benignness (weight-4 checks; hard filter): the two
               LAST-coupled corners — the weight-2 ancilla hook — must form
               a DIAGONAL pair, the same property the K&F table guarantees;
           (b) the NW-hook risk tier of weight-4 Z checks (a RANKING, not a
               filter — a boxed-in check falls back a tier instead of going
               infeasible): coupling the (-1, -1) corner LAST puts the
               measured fatal single-foot Z hook on it (Y on the ancilla
               after the third coupling) -> tier 2; coupling it FIRST is
               that hook's stabilizer equivalent (Z on that corner + a
               measurement flip) -> tier 1; mid-order -> tier 0;
           (c) Gidney similarity: most matched relative corner-pair orders
               first, exact relative-order matches ahead of the rest,
               smaller slot span ahead, then the lexicographically first
               slot tuple — a total, deterministic order.

           Forced checks are assigned fewest-candidates-first.
        4. A base whose cascade or solve fails falls through to the next.
           The four bases are swept TWICE: first with the plain cascade
           (every case that pass can schedule stays byte-identical to the
           earlier rule), then with ``skip_probe`` — a cascade stuck ONLY
           on an already-dropped check's K&F table fallback (an artifact
           of the PROBING fallback, not a real constraint: the final build
           replaces that check with a SOLVED order) probes on with that
           check's placements skipped (see ``_gidney_cascade``).  No pass
           admitting a schedule is an error, never a repack.

        This is the same offline constraint derivation the ``_M_SLOTS``
        tables came from; it runs online here because the conflict set
        depends on the merged layout around the Y patch (which seam and
        corridor checks the patch abuts), not on the patch alone.
        """
        best_err = None
        for skip_probe, base in itertools.product((False, True), range(4)):
            full = self._gidney_base_map(base)
            dropped, out, err = self._gidney_cascade(checks, kf_checks, full,
                                                     skip_probe=skip_probe)
            if out is not None and not dropped:
                return out
            if err is not None:
                best_err = err
                continue
            kept = {k: v for k, v in full.items() if k not in dropped}
            solved = self._solve_forced(checks, kf_checks, kept, dropped, full)
            if solved is None:
                continue
            try:
                return self._finish_slot_maps(
                    checks, kf_checks,
                    self._override_tms(checks, {**kept, **solved}))
            except RuntimeError as e:
                if 'fixed compact-7 schedule' not in str(e):
                    raise
                best_err = e
        if best_err is not None:
            raise best_err
        raise RuntimeError(
            f"gidney override: no base offset admits a schedule for "
            f"checks {sorted(self.gidney_uids)}")

    def _gidney_base_map(self, base):
        """Every override check's Gidney slot map at base offset ``base``:
        the corner coupled at tick t of ``_GIDNEY_ORDER`` sits at slot
        base + t, through the owner patch's orientation.  Returns
        ``{syn_coord: (type, {data_index: slot})}``."""
        system = self.system
        ov = {}
        for uid in sorted(self.gidney_uids):
            st = system.stabilizers[uid]
            typ = st.get('type')
            if typ not in ('X', 'Z'):
                raise RuntimeError(
                    f"gidney override: check {uid} has type {typ!r} — "
                    f"Gidney's order only covers plain X/Z plaquettes")
            sc = tuple(st['syn_coord'])
            patch = system.patches[system.coord_to_owner_map[sc]][0]
            tm = {}
            for t in range(4):
                dcan = _GIDNEY_ORDER[t][0 if typ == 'X' else 1]
                dg = patch.transform_vector(dcan)
                key = patch.get_grid_key((sc[0] + dg[0], sc[1] + dg[1]))
                di = system.grid_map.get(key)
                if di is not None and di in st['data_indices']:
                    tm[di] = base + t
            missing = set(st['data_indices']) - set(tm)
            if missing:
                raise RuntimeError(
                    f"gidney override: check {uid} at {sc}: data qubits "
                    f"{sorted(missing)} are not reachable by Gidney corner "
                    f"deltas")
            ov[sc] = (typ, tm)
        return ov

    def _override_tms(self, checks, ov, skip=frozenset()):
        """Per-check slot maps with the override map applied: an overridden
        check uses its assigned map, everyone else the fixed tables.  A check
        in ``skip`` (drop-cascade PROBING only, never a final build) places
        nothing at all: its K&F table fallback is known to collide, and the
        final build replaces it with a SOLVED order under full validation."""
        tms = []
        for ch in checks:
            if ch['syn'] in skip:
                tms.append({})
            elif ch['syn'] in ov:
                typ, tm = ov[ch['syn']]
                if typ != ch['type']:
                    raise RuntimeError(
                        f"gidney override: type mismatch at {ch['syn']} "
                        f"({typ!r} overridden, {ch['type']!r} gathered)")
                tms.append(dict(tm))
            else:
                tms.append(self._table_tm(ch))
        return tms

    def _gidney_cascade(self, checks, kf_checks, full, skip_probe=False):
        """Drop cascade: the minimal override subset that cannot keep pure
        Gidney order at this base.  A validation failure names the offending
        checks (``_validate_fixed`` embeds their ``check@(x, y)`` labels);
        the override checks among them are dropped and the round is
        re-validated, until it passes or no override check explains the
        failure.  Returns ``(dropped, ok_output_or_None, error_or_None)``.

        With ``skip_probe``, a cascade stuck ONLY on already-dropped checks
        — their K&F table PROBING fallback colliding, an artifact of the
        fallback rather than a real constraint (the final build replaces
        those checks with SOLVED orders under full validation) — skips
        those checks' probing placements (empty slot map, validation-
        exempt) instead of failing."""
        dropped = set()
        skipped = set()
        for _ in range(2 * len(full) + 2):
            ov = {k: v for k, v in full.items() if k not in dropped}
            try:
                out = self._finish_slot_maps(
                    checks, kf_checks,
                    self._override_tms(checks, ov, skip=skipped))
                return dropped, out, None
            except RuntimeError as e:
                msg = str(e)
                if 'fixed compact-7 schedule' not in msg:
                    raise
                cs = [(int(x), int(y))
                      for x, y in _OVERRIDE_LABEL_RE.findall(msg)]
                over = [c for c in cs if c in full and c not in dropped]
                if over:
                    dropped.update(over)
                    continue
                if skip_probe:
                    stale = [c for c in cs
                             if c in dropped and c not in skipped]
                    if stale:
                        skipped.update(stale)
                        continue
                return dropped, None, e
        return dropped, None, RuntimeError(
            "gidney override: drop cascade exhausted")

    def _solve_forced(self, checks, kf_checks, kept_ov, forced_syns, gid_full):
        """Backtracking: custom injective 7-slot maps for the forced checks,
        consistent (collision + K&F Sec. II) with everything else, candidates
        ranked as documented on ``_gidney_slot_maps`` (Gidney-most-similar
        within a risk tier).  Returns ``{syn: (type, {data_index: slot})}``
        or None when some forced check has no legal assignment."""
        system = self.system
        coords = {di: tuple(system.qubit_coords[di]) for ch in checks
                  for di in ch['deltas'].values()}
        bych = {ch['syn']: ch for ch in checks}
        # fixed placements: kept-gidney + fixed-table others + kf events
        occ = {}                       # (slot, coord) -> label
        fixed_cm = {}                  # syn -> (pauli_at, {coord: slot})
        for ch in checks:
            syn = ch['syn']
            if syn in forced_syns:
                continue
            if syn in kept_ov:
                tm = kept_ov[syn][1]
            else:
                tm = self._table_tm(ch)
            cm = {coords[di]: t for di, t in tm.items()}
            fixed_cm[syn] = (ch['pauli_at'], cm)
            for c, t in cm.items():
                occ[(t, c)] = syn
            for t in cm.values():
                occ[(t, syn)] = syn
        for ch in kf_checks:
            feet_cm = {}
            for slot, gate, (u, v) in _kf_events(ch):
                for q in (u, v):
                    if q in (ch['A'], ch['S'], ch['B']):
                        occ[(slot, q)] = ('kf', ch['B'])
                    else:
                        feet_cm[q] = slot
                        occ[(slot, q)] = ('kf', ch['B'])
            fixed_cm[('kf', ch['B'])] = (ch['pauli_at'], feet_cm)

        def shared_anti(pa_a, cm_a, pa_b, cm_b):
            qs = [q for q in cm_a if q in cm_b
                  and q in pa_a and q in pa_b and pa_a[q] != pa_b[q]]
            return qs if len(qs) >= 2 else None

        forced = sorted(forced_syns)
        cand = {}
        for syn in forced:
            ch = bych[syn]
            feet = [coords[di] for di in sorted(ch['deltas'].values())]
            g_tm = gid_full[syn][1]
            g_cm = {coords[di]: t for di, t in g_tm.items()}
            opts = []
            for slots in itertools.permutations(range(7), len(feet)):
                cm = dict(zip(feet, slots))
                # collision vs fixed (feet + own ancilla busy-slots)
                if any((t, c) in occ for c, t in cm.items()):
                    continue
                if any((t, syn) in occ for t in cm.values()):
                    continue
                # Sec. II vs every fixed check
                ok = True
                for os_, (pa_b, cm_b) in fixed_cm.items():
                    qs = shared_anti(ch['pauli_at'], cm, pa_b, cm_b)
                    if qs and len({cm[q] < cm_b[q] for q in qs}) > 1:
                        ok = False
                        break
                if not ok:
                    continue
                # hook benignness (hard filter): the two LAST-coupled feet
                # (the weight-2 ancilla hook) must form a DIAGONAL pair
                tier = 0
                if len(feet) == 4:
                    lo = sorted(feet, key=lambda c: cm[c])
                    a2, b2 = lo[2], lo[3]
                    if a2[0] == b2[0] or a2[1] == b2[1]:
                        continue
                # NW-hook risk tier for solved Z-checks (ranking, not
                # filter): Y-on-anc after the 3rd coupling with the (-1, -1)
                # corner last is the measured fatal single-foot Z hook
                # -> tier 2; that corner first is its stabilizer equivalent
                # (Z(corner) + measurement flip) -> tier 1; mid-order -> 0.
                if ch['type'] == 'Z' and len(feet) == 4:
                    sx, sy = syn
                    nwf = (sx - 1, sy - 1)
                    if nwf in cm:
                        lo = sorted(feet, key=lambda c: cm[c])
                        if lo[3] == nwf:
                            tier = 2
                        elif lo[0] == nwf:
                            tier = 1
                # gidney-similarity score: matched relative orders of pairs
                score = sum(1 for a, bq in itertools.combinations(feet, 2)
                            if (cm[a] < cm[bq]) == (g_cm[a] < g_cm[bq]))
                exact = all((cm[a] < cm[bq]) == (g_cm[a] < g_cm[bq])
                            for a, bq in itertools.combinations(feet, 2))
                span = max(cm.values()) - min(cm.values())
                opts.append((tier, -score, 0 if exact else 1, span,
                             tuple(slots), cm))
            if not opts:
                return None
            opts.sort()
            cand[syn] = [o[5] for o in opts]

        # order forced checks by fewest candidates first
        forced.sort(key=lambda s: len(cand[s]))
        assign = {}
        t0 = time.time()

        def compatible(syn, cm):
            ch = bych[syn]
            for os_, cm_b in assign.items():
                chb = bych[os_]
                for q, t in cm.items():
                    if cm_b.get(q) == t:
                        return False
                # ancilla busy-slot collisions between two forced checks'
                # ancillas only matter if one's ancilla is the other's foot
                # (never: ancilla coords are not data coords) — skip.
                qs = [q for q in cm if q in cm_b and q in ch['pauli_at']
                      and q in chb['pauli_at']
                      and ch['pauli_at'][q] != chb['pauli_at'][q]]
                if len(qs) >= 2 and len({cm[q] < cm_b[q] for q in qs}) > 1:
                    return False
            return True

        def dfs(i):
            if time.time() - t0 > 120:
                raise RuntimeError("gidney override: solver timeout")
            if i == len(forced):
                return True
            syn = forced[i]
            for cm in cand[syn]:
                if compatible(syn, cm):
                    assign[syn] = cm
                    if dfs(i + 1):
                        return True
                    del assign[syn]
            return False

        if not dfs(0):
            return None
        out = {}
        for syn, cm in assign.items():
            ch = bych[syn]
            di_of = {coords[di]: di for di in ch['deltas'].values()}
            out[syn] = (ch['type'], {di_of[c]: t for c, t in cm.items()})
        return out

    def _build_circuit(self):
        system = self.system
        checks, kf_checks = self._gather()
        T, slot_maps = self._slot_maps(checks, kf_checks)
        self.coupling_slots = T

        idx = system.index_map
        # K&F relay apparatus: A = flag (RX -> MZ: H after reset only),
        # B = syndrome (RZ -> MX: H before measure only), S = relay (RZ -> MZ:
        # no H).  Plain X/M checks keep the H sandwich on both sides.
        kf_gates = {}                 # slot -> list of ('CX'|'CZ', u_idx, v_idx)
        for ch in kf_checks:
            for slot, gate, (u, v) in _kf_events(ch):
                kf_gates.setdefault(slot, []).append((gate, idx[u], idx[v]))

        plain_h = [ch['syn_idx'] for ch in checks if ch['type'] in ('X', 'M')]
        h_pre = sorted(plain_h + [ch['flag_idx'] for ch in kf_checks])
        h_post = sorted(plain_h + [ch['syn_idx'] for ch in kf_checks])
        syn = sorted(
            [ch['syn_idx'] for ch in checks]
            + [i for ch in kf_checks
               for i in (ch['syn_idx'], ch['flag_idx'], ch['shared_idx'])])

        self.circuit.append("R", syn)
        self.circuit.append("TICK", tag="SE_start")
        if h_pre:
            self.circuit.append("H", h_pre)
        self.circuit.append("TICK")
        for t in range(T):
            cx, cz = [], []
            for gate, u, v in kf_gates.get(t, ()):
                (cx if gate == 'CX' else cz).extend([u, v])
            for ch, smap in zip(checks, slot_maps):
                si = ch['syn_idx']
                for di, slot in smap.items():
                    if slot != t:
                        continue
                    if ch['type'] == 'X':
                        cx.extend([si, di])       # X plaquette: CNOT(anc -> data)
                    elif ch['type'] == 'Z':
                        cx.extend([di, si])       # Z plaquette: CNOT(data -> anc)
                    elif ch['pauli_of'][di] == 'X':
                        cx.extend([si, di])       # mixed, X-support: CNOT(anc -> data)
                    else:
                        cz.extend([di, si])       # mixed, Z-support: CZ(data, anc)
            if cx:
                self.circuit.append("CNOT", cx)
            if cz:
                self.circuit.append("CZ", cz)
            self.circuit.append("TICK")
        if h_post:
            self.circuit.append("H", h_post)
        self.circuit.append("TICK")
        self.circuit.append("M", syn)
