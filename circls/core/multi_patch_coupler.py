"""Rotated multi-patch coupling layer — geometry, routing and stabilizer rules.

Consolidation of the former ``bent_layout`` / ``multi_patch`` /
``deterministic_checks`` / ``subset_routing`` modules into one file, mirroring
``unrotated/multi_patch_coupler.py``.  Contents, in order:

* SECTION bent_layout          — PatchSpec, place_patch, the bent-XZ two-patch
  layout generator and the shared GF(2)/symplectic helpers.
* SECTION multi_patch          — N-patch joint layout (MultiPatchLayout), the
  geometry-agnostic physics oracle (re-typing, mixed walls, verify()).
* SECTION deterministic_checks — rule-based stabilizer construction
  (rule 0 native lock, corners-first sweeps, convex-corner cuts).
* SECTION subset_routing       — obstacle-aware corridor routing on the coarse
  cell grid (abutting pitch 2d / seam-column pitch 2d+2), acceptance oracle,
  route_and_build.

The classic two-patch coupler (``RotatedTwoPatchCoupler``) and the SE engine
(``bent_joint_se``) remain separate, as in the unrotated package.
"""



# =============================================================================
# SECTION: bent_layout
# =============================================================================

# Parameterized rotated bent (XZ) joint-measurement layout generator.
#
# Given the two logical patches that participate in a joint measurement M(X̄₁·Z̄₂),
# auto-build the data qubits, ordinary CSS stabilizers, mixed (XZ) domain-wall
# stabilizers, boundary trim/replacement, readout chain, and the no-MPP gate-level
# syndrome-extraction circuit — reusing the existing rotated-code coordinate /
# stabilizer convention (data on (odd,odd); placement via shift; orientation
# via transpose_coords).

from dataclasses import dataclass, field

import numpy as np

from lightstim.qec_code.surface_code.rotated.code_patch import RotatedSurfaceCode


@dataclass(frozen=True)
class PatchSpec:
    """A logical patch placed on the coarse routing grid.

    origin:           bus-facing corner data coord, in the library (1,1)-corner convention.
    distance:         odd code distance.
    orientation:      "X_horizontal" (X̄ runs along a row) | "X_vertical" (X̄ runs up a column).

    Which logical each patch contributes to the joint ``M(∏ᵢ P̄ᵢ)`` is given
    separately by the ``target`` / ``interaction_type`` list (e.g.
    ``[("Q1", "Z"), ("Q2", "Z")]``), not by the patch itself.
    """
    name: str
    origin: tuple
    distance: int
    orientation: str


def place_patch(spec):
    """Build one rotated patch placed/oriented per ``spec``; return coord-keyed dicts.

    Returns dict(data=set[(col,row)], checks=list[checkdict], x_support, z_support).
    Reuses RotatedSurfaceCode: build at the (1,1) corner, transpose for X_horizontal,
    then shift so the corner lands at ``origin``.
    """
    if spec.orientation not in ("X_horizontal", "X_vertical"):
        raise ValueError(f"orientation must be X_horizontal|X_vertical, got {spec.orientation!r}")
    code = RotatedSurfaceCode(distance=spec.distance)
    if spec.orientation == "X_horizontal":
        code.transpose_coords()                       # default X̄ vertical -> horizontal
    code.shift_coords(spec.origin[0] - 1, spec.origin[1] - 1)
    qc = code.qubit_coords
    data = {tuple(qc[i]) for i in code.data_indices}
    checks = []
    for s in code.stabilizers:
        pauli = {tuple(qc[q]): P for q, P in s["pauli"].items()}
        checks.append({"syn": tuple(s["syn_coord"]), "type": s["type"],
                       "pauli": pauli, "corners": sorted(pauli)})
    x_support = next(sorted(tuple(qc[q]) for q in lo["pauli"])
                     for lo in code.logical_ops if lo["type"] == "X")
    z_support = next(sorted(tuple(qc[q]) for q in lo["pauli"])
                     for lo in code.logical_ops if lo["type"] == "Z")
    return dict(data=data, checks=checks, x_support=x_support, z_support=z_support)


def conjugate_patch_records(code):
    """Typeswap a built patch IN PLACE: every stabilizer/logical record swaps
    X<->Z (type and per-qubit Paulis) and the syndrome-role index sets swap.

    The result is the CONJUGATE-CONVENTION patch: textbook-shaped on the same
    qubits with the checkerboard colours exchanged — exactly the structure the
    seam-column painting assigns to a minority patch.  A minority patch
    registered this way matches the routed merged check set natively (rule-0
    lock), so the protocol contains no transversal-H morph anywhere; this is
    how ``RoutedMultiPatchLSExperiment`` registers its minority patches."""
    for rec in code.stabilizers:
        rec["type"] = _FLIP.get(rec["type"], rec["type"])
        rec["pauli"] = {q: _FLIP.get(P, P) for q, P in rec["pauli"].items()}
    for rec in code.logical_ops:
        rec["type"] = _FLIP.get(rec["type"], rec["type"])
        rec["pauli"] = {q: _FLIP.get(P, P) for q, P in rec["pauli"].items()}
    xs = getattr(code, "syndrome_indices_x", None)
    zs = getattr(code, "syndrome_indices_z", None)
    if xs is not None and zs is not None:
        code.syndrome_indices_x, code.syndrome_indices_z = zs, xs
    return code


def conjugate_layout(lay):
    """The colour-swapped COPY of a built :class:`MultiPatchLayout`.

    Every check swaps X<->Z (type and per-qubit Paulis; an 'M'/'MIXED' check
    keeps its tag while its feet flip individually), every logical's measured
    Pauli label flips, the bus flips, the recorded specs/target flip to the
    live convention, and every orientation-domain value flips WITH the
    colours (``BENIGN_TABLES`` pairs orientation and Pauli — flipping colours
    without domains would silently change the hook-benign gate order).

    Colour swapping is an exact symplectic relabeling, so a verified layout
    stays verified: route + verify in the STANDARD colours, then emit the
    swapped copy.  This is the corridor construction a same-Pauli joint
    between two SWAP-rotated (colour-swapped) patches needs — their weight-2
    positions are unchanged, so the rule table still says "plain merge", and
    the merge is simply the standard corridor under a global colour swap."""
    from dataclasses import replace as _dc_replace
    _FLIP_O = {"X_horizontal": "X_vertical", "X_vertical": "X_horizontal"}
    checks = [dict(ch, type=_FLIP.get(ch.get('type'), ch.get('type')),
                   pauli={q: _FLIP.get(P, P) for q, P in ch['pauli'].items()})
              for ch in lay.checks]
    logicals = [(nm, _FLIP.get(P, P), sup) for nm, P, sup in lay.logicals]
    target = [(nm, _FLIP.get(P, P)) for nm, P in (lay.target or [])]
    specs = [_dc_replace(s, orientation=_FLIP_O[s.orientation])
             for s in (lay.specs or [])]
    domains = {q: _FLIP_O[o] for q, o in (lay.domains or {}).items()}
    bus = _FLIP.get(lay.bus, lay.bus)
    return _dc_replace(lay, checks=checks, logicals=logicals, target=target,
                       specs=specs, domains=domains, bus=bus)


# -----------------------------------------------------------------------------
# Bent (XZ) joint-measurement layout
# -----------------------------------------------------------------------------

_FLIP = {"X": "Z", "Z": "X"}


class BentLayoutError(ValueError):
    """Raised when the requested patch placement cannot host a bent XZ joint measurement
    (e.g. the two patches are misaligned, or their parity/seam endpoints are incompatible).
    The message states the concrete geometric reason — it is NOT a generator hard-coding limit."""







#: Post-routing seam-table dispatch (route first, then classify every attach
#: seam on the path by rule-table v3).  Same-letter type-split joints:
#: the minority-type target is REGISTERED in the recolour convention and its
#: seam carries the verified mixed wall family — the rule-table row algebra
#: row2 = row4 o row3 (user-confirmed, measured full-distance end-to-end on
#: straight, L-bend and T-junction geometries, 2026-07-29).
TABLE_WALL_DISPATCH = True



def _bent_plaquettes(data, retype, phase):
    """Every (even,even) plaquette center with >=2 data corners, typed by the re-typing model.

    Base type at center ``(cx, cy)`` is 'X' if ``((cx+cy)//2 + phase)`` is odd else 'Z'.
    Each corner's Pauli equals the base type, flipped (X<->Z) if the corner lies in the
    re-typed set ``retype`` (the X-side arm).  All-equal corner Paulis -> a pure check of
    that type; mixed -> a domain-wall check of type 'M'.  ``phase`` (0/1) absorbs the
    position-dependent parity of ``(cx+cy)//2`` so the construction is translation-correct.
    """
    xs = [c for c, _ in data]
    ys = [r for _, r in data]
    out = []
    for cx in range(min(xs) - 1, max(xs) + 2):
        for cy in range(min(ys) - 1, max(ys) + 2):
            if cx % 2 or cy % 2:
                continue
            present = [(cx + a, cy + b) for a in (-1, 1) for b in (-1, 1)
                       if (cx + a, cy + b) in data]
            if len(present) < 2:
                continue
            base = "X" if ((cx + cy) // 2 + phase) % 2 == 1 else "Z"
            pauli = {q: (_FLIP[base] if q in retype else base) for q in sorted(present)}
            types = set(pauli.values())
            t = next(iter(types)) if len(types) == 1 else "M"
            out.append({"syn": (cx, cy), "type": t, "pauli": pauli,
                        "corners": sorted(pauli)})
    return out




def _commute(a, b, n):
    return int((a[:n] & b[n:]).sum() + (a[n:] & b[:n]).sum()) % 2 == 0






def _symplectic(data):
    idx = {c: i for i, c in enumerate(sorted(data))}
    n = len(idx)

    def sv(pauli):
        v = np.zeros(2 * n, np.uint8)
        for c, P in pauli.items():
            if P in ("X", "Y"):
                v[idx[c]] ^= 1
            if P in ("Z", "Y"):
                v[n + idx[c]] ^= 1
        return v
    return sv, n


def _gf2_rank(rows):
    if not len(rows):
        return 0
    M = np.array([r.copy() for r in rows], np.uint8)
    r = 0
    for c in range(M.shape[1]):
        piv = next((k for k in range(r, len(M)) if M[k, c]), None)
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        for k in range(len(M)):
            if k != r and M[k, c]:
                M[k] ^= M[r]
        r += 1
    return r






def _no_tick_collision(circuit):
    """True iff no qubit is touched by two operations within the same TICK window."""
    skip = {"QUBIT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS", "TICK",
            "DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Z_ERROR", "Y_ERROR"}
    seen = set()
    for op in circuit.flattened():
        if op.name == "TICK":
            seen = set()
            continue
        if op.name in skip:
            continue
        for t in op.targets_copy():
            if t.is_qubit_target:
                if t.value in seen:
                    return False
                seen.add(t.value)
    return True






# =============================================================================
# SECTION: multi_patch
# =============================================================================

# Rotated **N-patch** joint lattice-surgery measurement — ``M(∏ᵢ P̄ᵢ)``.
#
# A genuinely N-patch generalization of the two-/three-patch bent joint measurements.  The user
# gives one :class:`.PatchSpec` per logical patch (origin / distance / orientation) and an
# optional ``target`` (which logical each patch contributes); the generator
# **derives the whole geometry from the patch origins** — routed bus, mixed (XZ) walls, boundary
# trim/replacement and the readout chain all follow from where the patches sit.  There is **no
# ``offset`` parameter and no hard-coded p1/p2/p3 topology**: any vertical/horizontal stagger is an
# *internal* quantity read off the origins.  Changing any origin regenerates the layout; an
# incompatible placement raises :class:`.BentLayoutError` with a concrete geometric reason.
#
# **Routing layer (coordinate-derived, not a template).**  The geometry is built in two layers.
# The *routing* layer turns the patch coordinates into a connected data region; the *physics* layer
# (re-typing + mixed walls + selection + verification) then works on whatever region it is handed —
# it is geometry-agnostic, so all of the cleverness is in routing.
#
# * **Pairwise fusion corridor** (Phase 1) — for two same-type (X-side) patches that share a row or a
#   column, :func:`_xx_fuse` derives a ``d``-wide corridor straight from their boxes: it extends the
#   lower patch's columns up to the upper patch (or the right patch's rows left to the left patch).
#   The corridor is **computed from the coordinates** (any even stagger works); there is no
#   ``p1.right == p3.left`` template to match and no ``offset`` parameter.  A narrow corridor (not a
#   wide shared band) is what keeps the two X-logicals independent.
# * **Routing graph over the X-patches** (Phase 2) — fusing a *set* of X-patches is a graph: each
#   edge is one pairwise corridor, and the X-region is the union of the patches and their corridors.
#   The graph comes from ``routing``:
#   * ``routing="auto"`` — the **bounded auto-router** (:func:`_candidate_edge_sets`) computes the
#     fusable X-patch pairs (those sharing a row/column) and takes a spanning tree (greedy MST by
#     centre distance) over them.  It is *bounded*: it tries that one coordinate-derived candidate
#     tree, not an exhaustive search over all topologies — hence "not a general auto-router".
#   * an **explicit graph** — ``routing={"type": "tree", "edges": [("p1","p3"), ...]}`` or a bare
#     list of edges — uses exactly the X–X fusions you name.
# * **Each** Z-side patch is reached by a band (:func:`_zband`) tapping the X-region (a vertical bus,
#   :func:`_add_bus`, grows it to span every Z-patch's rows); the **mixed (XZ) domain wall** falls at
#   each Z-patch's boundary automatically (``retype`` = everything except the Z-patches).  Multiple
#   Z-side patches are supported — each gets its own wall.
#
# Each candidate routing is **oracle-gated**: it is accepted only if the selection finds a boundary
# set that puts the full joint in the stabilizer span with every proper sub-product excluded.  A
# disconnected X-region, an unfusable explicit edge, bad parity/overlap, or an unreachable/overlapping
# Z all raise :class:`.BentLayoutError` with a concrete reason — the generator never silently
# mis-routes or mis-measures.
#
# ``M(X̄₁X̄₃Z̄₂)`` (one Z), ``M(Z̄₁Z̄₂X̄₃)`` (two Z onto one X-bus), and ``M(Z̄₁Z̄₂X̄₃X̄₄)`` (the XXZZ
# cross: X̄₃/X̄₄ on a vertical trunk, Z̄₁/Z̄₂ on the horizontal arm) are all **examples** of this one
# API, not special cases.
#
# **Scope (the real constraint is *separation*, not patch counts).**  Both multiple X-side and
# multiple Z-side patches work.  The genuine limit is homological: **two same-type patches whose
# logicals run parallel collapse to a measured pair-product when they fuse into one region with no
# mixed wall / junction separating them.**  So two X-patches *adjacent* on a bare trunk give
# ``X̄ᵢX̄ⱼ`` in the span (a sub-product), but two X-patches at opposite ends of a trunk with the
# Z-walls crossing *between* them (the XXZZ cross) stay independent and the 4-body joint is clean.
# Separately-walled Z-patches do not collapse (X-side material sits between their walls).  When a
# routing would collapse a pair, the oracle **detects it** (no boundary selection measures the full
# joint) and **raises** — it never silently mis-measures.  The auto-router only tries one
# coordinate-derived candidate, so a collapsing placement is rejected; reposition so same-type
# logicals are separated, or pass an explicit routing.
#
# The construction realizes **only the full joint** ``∏ᵢ P̄ᵢ`` — no single logical and **no proper
# sub-product** is measured — leaving ``N-1`` logical degrees of freedom.  Selection forbids every
# proper sub-product from the stabilizer span, forces the full joint into it, and avoids leaving a
# weight-1 leftover logical.  A bit-packed-int GF(2) search makes this fast (3-patch ``d=5`` and the
# 4-patch build in ~1-2 s).  ``verify()`` is the oracle (eleven checks).
#

import itertools
from dataclasses import dataclass, field

import numpy as np



# -----------------------------------------------------------------------------
# bit-packed GF(2) machinery — Pauli vectors as Python ints (X in bits 0..n-1, Z in n..2n-1).
# Int XOR / int.bit_count() are C-level, so the boundary search scales to N=5/6 and d=5.
# -----------------------------------------------------------------------------

def _int_symplectic(data):
    """Return ``(isv, n)`` where ``isv({coord: pauli})`` packs a Pauli into a 2n-bit int."""
    idx = {c: i for i, c in enumerate(sorted(data))}
    n = len(idx)

    def isv(pauli):
        v = 0
        for c, P in pauli.items():
            i = idx[c]
            if P in ("X", "Y"):
                v |= 1 << i
            if P in ("Z", "Y"):
                v |= 1 << (n + i)
        return v
    return isv, n


def _icommute(a, b, n):
    mask = (1 << n) - 1
    return (((a & mask) & (b >> n)).bit_count() + ((a >> n) & (b & mask)).bit_count()) & 1 == 0


class _IntBasis:
    """An incremental GF(2) row basis over bit-packed ints (reduction by highest set bit)."""
    __slots__ = ("piv",)

    def __init__(self):
        self.piv = {}                      # highest-set-bit -> reduced row int

    def copy(self):
        b = _IntBasis()
        b.piv = dict(self.piv)
        return b

    def reduce(self, v):
        # Canonical (fixed-pivot) reduction: eliminate EVERY pivot bit,
        # high to low, not just the leading run.  The old form returned at
        # the first leading bit without a pivot, leaving lower pivot bits
        # set -- a non-canonical residue that made reduce NON-linear
        # (reduce(a)^reduce(b) != reduce(a^b)).  contains()/add() only test
        # ==0 and were unaffected, but verify()'s no_subjoint check XORs the
        # residues assuming linearity, so the early return let a proper
        # subset product that lies in span(B) escape the gate (fix
        # 2026-08-24).  A canonical residue has no pivot bit set, so add()'s
        # `piv[r.bit_length()-1] = r` still lands on a fresh leading bit.
        for b in sorted(self.piv, reverse=True):
            if (v >> b) & 1:
                v ^= self.piv[b]
        return v

    def add(self, v):                      # returns True iff v is independent (and adds it)
        r = self.reduce(v)
        if r:
            self.piv[r.bit_length() - 1] = r
            return True
        return False

    def contains(self, v):
        return self.reduce(v) == 0

    @property
    def rank(self):
        return len(self.piv)


# -----------------------------------------------------------------------------
# routing layer: coordinate-derived corridors (fusion / band), an explicit routing
# graph, and a bounded auto-router.  This REPLACES the old fixed staircase template:
# corridors are computed from the patch coordinates (any even offset works), and the
# old corner-touch staircase is just one possible edge set of corner fusions.
# -----------------------------------------------------------------------------

def _cols(d):
    return sorted({c for c, _ in d})


def _rows(d):
    return sorted({r for _, r in d})
















def _connected(region):
    """True iff ``region`` (a set of (odd,odd) coords) is connected via the plaquette graph."""
    region = set(region)
    if not region:
        return True
    start = next(iter(region))
    seen = {start}; stack = [start]
    while stack:
        x, y = stack.pop()
        for nx, ny in ((x + 2, y), (x - 2, y), (x, y + 2), (x, y - 2),
                       (x + 2, y + 2), (x + 2, y - 2), (x - 2, y + 2), (x - 2, y - 2)):
            if (nx, ny) in region and (nx, ny) not in seen:
                seen.add((nx, ny)); stack.append((nx, ny))
    return len(seen) == len(region)




def _logical_direction(pauli, orientation):
    """The direction the measured ``pauli`` logical string must run, per the patch ``orientation``.

    ``orientation`` names the X̄ direction (``"X_horizontal"`` ⇒ X̄ runs along a row, ``"X_vertical"``
    ⇒ X̄ runs up a column); Z̄ is perpendicular to X̄.  So a measured ``X`` on an ``X_horizontal``
    patch runs ``"horizontal"``, a measured ``Z`` on an ``X_horizontal`` patch runs ``"vertical"``,
    and vice-versa for ``X_vertical``.
    """
    x_is_horizontal = orientation == "X_horizontal"
    if pauli == "X":
        return "horizontal" if x_is_horizontal else "vertical"
    return "vertical" if x_is_horizontal else "horizontal"


def _patch_rep(patch, pauli, direction, F, sv, n):
    """A patch-spanning ``pauli`` string **in the requested ``direction``** commuting with bulk ``F``.

    The orientation declared on the ``PatchSpec`` is honoured: an ``X_horizontal`` X-logical is
    represented by a horizontal string, an ``X_vertical`` X-logical by a vertical one (and Z̄
    perpendicular).  Returns the support, or ``None`` if no string in that direction commutes — the
    caller tries the other parity phase, and if both fail the placement is rejected rather than the
    logical being silently re-oriented.
    """
    c, r = _cols(patch), _rows(patch)
    cands = ([[(x, y) for y in r] for x in c] if direction == "vertical"
             else [[(x, y) for x in c] for y in r])
    for sup in cands:
        v = sv({q: pauli for q in sup})
        if all(_commute(v, f, n) for f in F):
            return sup
    return None
def _readout_chain(data, checks, logicals):
    """Check syndromes whose GF(2) product equals the joint ``∏ᵢ P̄ᵢ`` (for highlighting/readout)."""
    from lightstim.utils.linear_algebra import solve_linear_decomposition
    sv, n = _symplectic(data)
    basis = np.array([sv(ch["pauli"]) for ch in checks], np.uint8)
    target = np.zeros(2 * n, np.uint8)
    for P, sup in logicals:
        target ^= sv({c: P for c in sup})
    co, dep, _ = solve_linear_decomposition(basis=basis, targets=target.reshape(1, -1),
                                             reduce_weight=True)
    if not dep[0]:
        return set()
    return {checks[i]["syn"] for i in np.where(co[0])[0]}


# -----------------------------------------------------------------------------
# layout + builder
# -----------------------------------------------------------------------------

@dataclass
class MultiPatchLayout:
    """A rotated N-patch joint-measurement layout (``M(∏ᵢ P̄ᵢ)``).

    ``logicals`` is a list of ``(name, measured_pauli, support)`` — one per patch.
    ``x_observable`` is the X-type logical the X-memory circuit reads.
    """
    distance: int
    data: list
    checks: list
    logicals: list
    x_observable: list
    readout_chain: set = field(default_factory=set)
    specs: list = field(default_factory=list)
    target: list = field(default_factory=list)
    #: orientation-domain map (data coord -> 'X_horizontal' | 'X_vertical') used
    #: by the hook-benign scheduler; empty means "all default (X_horizontal)".
    domains: dict = field(default_factory=dict)
    #: the native (corridor) Pauli basis this layout was built for.  ``None`` means
    #: "majority target basis (tie -> X)"; an explicit 'X'/'Z' is a PLANNED bus that
    #: forces which patches attach through mixed walls / are conjugated.
    bus: str = None
    #: bus data qubits on the RECOLOURED side of a corridor-internal colour wall
    #: (segment recolouring, flip_cells): they are initialized/read out in the
    #: OPPOSITE basis of the rest of the bus.  Empty for uniform corridors.
    retyped: frozenset = frozenset()

    @property
    def N(self):
        return len(self.logicals)

    def build_circuit(self, rounds=None, p=0.0):
        """No-MPP gate-level syndrome-extraction circuit.

        The memory experiment runs in the **bus basis** — the majority target basis
        (tie -> X), matching the corridor's native type.  ``x_observable`` holds a
        bus-basis logical representative (see ``_construct_phase``), so basis and
        tracked observable always agree; running the opposite-basis experiment on a
        bus layout is ill-posed (a single measurement flip would flip the observable
        undetected).

        WARNING: standalone builder (wraps ``RotatedBentJointMeasurement.circuit``);
        its observable is a single bus-basis logical, NOT the ``SyndromeTracker``
        joint m.  Good for single-patch memory + determinism/collision checks, but
        its MULTI-PATCH graphlike distance is NOT the real joint distance -- use
        :class:`RoutedMultiPatchLSExperiment` for joint distance / LER."""
        from lightstim.qec_code.surface_code.rotated.bent_joint_se import RotatedBentJointMeasurement
        if rounds is None:
            rounds = self.distance
        if self.bus is not None:
            basis = self.bus
        else:
            n_x = sum(1 for _, P, _ in self.logicals if P == "X")
            basis = "X" if n_x >= len(self.logicals) - n_x else "Z"
        return RotatedBentJointMeasurement(
            list(self.data), self.checks, list(self.x_observable),
            domains=self.domains,
        ).circuit(rounds=rounds, p=p, basis=basis)

    def verify(self, rounds=None):
        """The eleven N-patch acceptance checks; returns a dict of booleans (all True == valid)."""
        isv, n = _int_symplectic(self.data)
        S = [isv(ch["pauli"]) for ch in self.checks]
        singles = [isv({c: P for c in sup}) for _, P, sup in self.logicals]
        joint = 0
        for v in singles:
            joint ^= v
        N = len(self.logicals)
        B = _IntBasis()
        for v in S:
            B.add(v)
        # no_subjoint, polynomially: the subset-products lying in span(B)
        # form the NULL SPACE of the residues {B.reduce(single_k)} —
        # B.reduce is linear (fixed-pivot XOR elimination), so combination
        # masks ride along a second elimination.  The old loop enumerated
        # all 2^N-2 proper subsets: exponential in the patch count (N=16 =
        # 65534 big-int reduces, dj_32's entire route timeout, 2026-08-05).
        # Accept iff the null space is {0} or {0, full}: the ONLY dependent
        # combination allowed is the full joint product.  Any violating
        # dependent surfaces as a mask != full (two full masks cannot both
        # occur: dependents' masks are linearly independent).
        full_mask = (1 << N) - 1
        piv2 = {}
        no_subjoint_ok = True
        for k in range(N):
            r = B.reduce(singles[k])
            m = 1 << k
            while r:
                b = r.bit_length() - 1
                if b not in piv2:
                    piv2[b] = (r, m)
                    break
                pr, pm = piv2[b]
                r ^= pr
                m ^= pm
            if not r and m != full_mask:
                no_subjoint_ok = False
        commute = all(_icommute(S[i], S[j], n) for i in range(len(S)) for j in range(i + 1, len(S)))
        twist = any(P == "Y" for ch in self.checks for P in ch["pauli"].values())

        def w1():
            for q in self.data:
                for P in "XZ":
                    v = isv({q: P})
                    if B.reduce(v) != 0 and all(_icommute(v, b, n) for b in B.piv.values()):
                        return True
            return False

        out = dict(
            commute=commute,
            joint=B.contains(joint),
            no_single=not any(B.contains(v) for v in singles),
            no_subjoint=no_subjoint_ok,
            no_twist=not twist,
            logical_count=len(self.data) - B.rank == N - 1,
            no_weight1_logical=not w1(),
        )
        if any(ch.get('kf') for ch in self.checks):
            # a layout hosting stretched (kf) checks cannot run the bent
            # standalone builder — the circuit-level items are covered by the
            # end-to-end diagonal pipeline instead; the algebraic oracle above
            # is the acceptance decision here
            return out
        circuit = self.build_circuit(rounds=rounds)
        out["no_mpp"] = "MPP" not in str(circuit)
        out["no_tick_collision"] = _no_tick_collision(circuit)
        try:
            dem = circuit.detector_error_model(decompose_errors=True)
            dem_ok = (dem.num_detectors == circuit.num_detectors
                      and dem.num_observables == circuit.num_observables)
        except Exception:
            dem_ok = False
        det, obs = circuit.compile_detector_sampler(seed=0).sample(200, separate_observables=True)
        out["dem_valid"] = dem_ok and not det.any() and not obs.any()
        return out








# -----------------------------------------------------------------------------
# diagnostic mode: build the candidate geometry and report WHY it (in)validates,
# WITHOUT raising — so a failing layout can still be drawn and explained.
# -----------------------------------------------------------------------------








# =============================================================================
# SECTION: deterministic_checks
# =============================================================================

# Deterministic mixed-stabilizer construction for N-patch subset joint measurements.
#
# Implements the documented construction rules **as rules** —
# no randomized search anywhere.  Per checkerboard phase the DIRECT construction is:
#
# 1.  **Bulk** — every complete (weight-4) plaquette of the merged region is forced:
#     patch bulk, bus bulk and the mixed (XZ) walls alike.  Patch-edge weight-2s at the
#     bus interface are gone automatically (those plaquettes gained bus corners and became
#     forced bulk).
# 2.  **Logical representatives** — one patch-spanning representative per measured logical
#     is fixed from the forced bulk (deterministically).
# 3.  **Boundary scan (letter rule + spacing)** — pool candidates (the boundary-clipped
#     weight-2 semicircles and the weight-3 concave-corner plaquettes of Fig 34 labels
#     1/3) are visited in anchored-walk order; a slot is placed iff its letter equals its
#     REGION's letter (bus letter; conjugate inside a recoloured region) and it shares no
#     conflicting data foot with an already-placed stabilizer — sharing a foot IS the
#     paper's "one lattice spacing" skip.  On straight edges this reproduces the
#     add-one-skip-one alternation; patch-native lobes are rule-0 anchors, wall (kf)
#     records own their feet.
# 4.  **Convex outer corners, LOCAL cuts** (Fig 34 label 2; design decision 2026-07-31) — a
#     bus-corner data qubit whose corner plaquette has a boundary stabilizer at one
#     lattice spacing (sharing a foot), or a weight-1 leftover on a corner, is cut IN
#     PLACE: that plaquette drops the foot (weight-4 -> weight-3) and nothing else moves.
#     A bounded second variant relays the boundary ONCE on the cut-reduced region (some
#     wall-adjacent layouts only form the joint in the relaid pattern).  No rebuild
#     iteration exists — the old cut-and-rebuild ladder manufactured phantom corners at
#     the notch and cascaded.  (The legacy greedy engine with corner-first anchoring and
#     rule-5 re-anchoring is deleted; suite census showed zero reachable customers.)
#

import itertools
from types import SimpleNamespace


_ALL_DETERMINISTIC = ["rule_based_joint_checks"]

_ORTH = ((2, 0), (-2, 0), (0, 2), (0, -2))


def _corner_qubits(dset):
    """Exposed corner data qubits: ≤2 orthogonal data neighbours."""
    return {q for q in dset
            if sum(((q[0] + dx, q[1] + dy) in dset) for dx, dy in _ORTH) <= 2}


class _Builder:
    """One deterministic greedy construction on a fixed region and phase.

    ``forbidden`` is a set of syndrome sites the boundary may NOT use — the ancilla
    qubits already owned by neighbouring idle patches; vetoing them makes the
    alternating pattern interleave with the neighbour on a shared ancilla line, so
    edge-adjacent placement is physically collision-free whenever the parity allows.
    """

    def __init__(self, placed, target, orient, data, retype, phase, forbidden=frozenset(),
                 native_lock=False, bus=None, conj_names=None, extra_forced=(),
                 retire_lobes=frozenset()):
        self.data = sorted(data)
        self.forbidden = frozenset(forbidden)
        self.target = target
        self.patchq = set().union(*[placed[nm] for nm, _ in target])
        self._placed = placed
        self._orient = orient
        self._retype = set(retype)
        if bus is None:
            n_x = sum(1 for _, P in target if P == "X")
            bus = "X" if n_x >= len(target) - n_x else "Z"
        self._bus = bus
        self.plaqs = _bent_plaquettes(self.data, retype, phase)
        self.forced = [p for p in self.plaqs if len(p["pauli"]) >= 4]
        # injected forced checks (e.g. a corridor-internal stretched wall's kf
        # records): they join the forced set verbatim, so the rank/joint
        # acceptance metric and the logical representatives account for them;
        # their feet are protected from the convex-corner cut rule like patch
        # qubits (a cut foot would orphan the wall record)
        self.forced = self.forced + [dict(e) for e in extra_forced]
        self.patchq |= {q for e in extra_forced for q in e["pauli"]}
        self.pool = sorted((p for p in self.plaqs if len(p["pauli"]) < 4),
                           key=lambda p: p["syn"])
        if retire_lobes:
            # the wall band's two ancilla lines belong to the wall (same
            # retire semantics as the patch|patch wall coupler): no rule tile
            # may sit there
            self.pool = [p for p in self.pool if p["syn"] not in retire_lobes]
        if extra_forced:
            # the injected records' apparatus sites (syn + kf flag/shared) are
            # occupied — expel pool tiles there; anything else (e.g. the corner
            # lobe at the wall band's free end) stays available to the rules
            occ = set()
            for e in extra_forced:
                occ.add(tuple(e["syn"]))
                kf = e.get("kf") or {}
                for k in ("flag", "shared"):
                    if k in kf:
                        occ.add(tuple(kf[k]))
            self.pool = [p for p in self.pool if p["syn"] not in occ]
        # rule 0 -- native lock: a target patch whose interior matches the global
        # checkerboard (not retyped, phase-compatible) keeps its TEXTBOOK standalone
        # construction verbatim: its native boundary semicircles are seeded as forced
        # picks, and non-native tiles fully inside the patch are expelled from the
        # pool.  The one exception is a facing semicircle whose forced-bulk extension
        # (superset whose extra corners lie OUTSIDE the patch, i.e. seam/corridor
        # qubits) exists -- the extension replaces it, per joint-measurement rank.
        self.native_seed, self.locked_cells = [], set()
        if native_lock:
            forced_by_sup = {frozenset(p["pauli"]): p for p in self.forced}
            pool_by_key = {(frozenset(p["pauli"]), p["type"]): p for p in self.pool}
            if bus is None:
                n_x = sum(1 for _, P in target if P == "X")
                bus = "X" if n_x >= len(target) - n_x else "Z"
            _FLIP_O = {"X_horizontal": "X_vertical", "X_vertical": "X_horizontal"}
            _FLIP_P = {"X": "Z", "Z": "X"}
            for nm, _P in target:
                cells = placed[nm]
                # conjugate-registered patch: its lock target is the CONJUGATE-
                # CONVENTION construction (transposed native, types swapped).
                # conj_names is the explicit registration set; None infers it
                # from the measured Pauli (minority basis), the historical rule.
                flipped = (nm in conj_names) if conj_names is not None \
                    else (_P != bus)
                dd = len({x for x, _ in cells})
                o = (min(x for x, _ in cells), min(y for _, y in cells))
                onm = _FLIP_O[orient[nm]] if flipped else orient[nm]
                try:
                    nat = place_patch(SimpleNamespace(
                        origin=o, distance=dd, orientation=onm))["checks"]
                except Exception:
                    continue
                if flipped:
                    nat = [dict(c, type=_FLIP_P[c["type"]],
                                pauli={q: _FLIP_P[P] for q, P in c["pauli"].items()})
                           for c in nat]
                seeds, ok = [], True
                for c in nat:
                    sup = frozenset(c["pauli"])
                    if len(sup) >= 4:              # interior: must equal a forced tile
                        f = forced_by_sup.get(sup)
                        if f is None or f["type"] != c["type"]:
                            ok = False
                            break
                    else:                          # boundary semicircle / corner tile
                        if tuple(c["syn"]) in retire_lobes:
                            continue               # facing an ARM WALL: the lobe is
                                                   # retired, the wall records replace it
                        if any(sup < frozenset(f["pauli"])
                               and not ((frozenset(f["pauli"]) - sup) & cells)
                               for f in self.forced):
                            continue               # facing: replaced by its extension
                        cand = pool_by_key.get((sup, c["type"]))
                        if cand is None:
                            ok = False
                            break
                        seeds.append(cand)
                if not ok:                         # rule 0 is HARD: a phase that cannot
                    self.fail = (f"native lock: patch {nm} textbook construction is "
                                 f"incompatible with checkerboard phase {phase}")
                    return                         # host a native patch is rejected
                self.native_seed.extend(seeds)
                self.locked_cells |= cells
            if self.locked_cells:
                nset = {(frozenset(p["pauli"]), p["type"]) for p in self.native_seed}
                self.pool = [p for p in self.pool
                             if not (set(p["pauli"]) <= self.locked_cells
                                     and (frozenset(p["pauli"]), p["type"]) not in nset)]
        self.seed_syns = {p["syn"] for p in self.native_seed}
        self.isv, self.n = _int_symplectic(self.data)
        sv, _ = _symplectic(self.data)
        F = [sv(p["pauli"]) for p in self.forced]
        self.reps = {}
        for nm, P in target:
            sup = _patch_rep(placed[nm], P, _logical_direction(P, orient[nm]), F, sv, self.n)
            if sup is None:
                self.fail = (f"no {P}-representative for patch {nm} commutes with the "
                             f"forced bulk (checkerboard phase {phase})")
                return
            self.reps[nm] = (P, sup)
        self.fail = None
        self.repv = [self.isv({q: P for q in sup}) for P, sup in self.reps.values()]
        self.fvecs = [self.isv(p["pauli"]) for p in self.forced]

    def _seg_letter(self, p):
        """Boundary LETTER for candidate ``p``, or None to fall back to
        parity.  Measured ground truth (engine dump, docs §2.6): the merged
        complex's free boundary carries the BUS letter everywhere outside
        the patches — bus-letter lobes forbid dual-letter chains from
        terminating across the corridor, which is what protects the joint
        measurement.  The patches keep their native lobes verbatim (rule-0
        anchors); any other all-inside-patch slot belongs to the native
        construction, not to this rule.  Mixed (domain-wall) tiles fall back
        to the parity scan."""
        if p["type"] not in "XZ":
            return None                       # mixed/domain-wall tile
        for nm, _P in self.target:
            if all(q in self._placed[nm] for q in p["pauli"]):
                if self.native_seed:
                    return None               # locked: seeds own the patches
                # UNLOCKED ladder fallback (native constructions cannot share
                # the attempt's phase): the patch boundary is re-derived like
                # an ordinary patch at this phase — each side keeps its
                # orientation-determined letter (X̄-horizontal: E/W carry X,
                # N/S carry Z; X̄-vertical swaps), measured from the engine's
                # unlocked successes (docs §2.6)
                cells = self._placed[nm]
                xs = [x for x, _ in cells]
                ys = [y for _, y in cells]
                fx = {x for x, _ in p["pauli"]}
                fy = {y for _, y in p["pauli"]}
                if fx == {min(xs)} or fx == {max(xs)}:
                    side_ew = True
                elif fy == {min(ys)} or fy == {max(ys)}:
                    side_ew = False
                else:
                    return None               # corner-spanning tile: undecided
                o = self._orient[nm]
                # OPEN (docs §2.6): in the unlocked fallback ONE of the
                # patches (the parity law's recoloured minority) follows the
                # FLIPPED side map — the selection criterion is not yet
                # transcribed; until it is, the orientation map is used for
                # all patches and the residual per-patch lobe deltas are the
                # known gap (cross-check harness tracks them).
                return ("X" if side_ew else "Z") if o == "X_horizontal" \
                    else ("Z" if side_ew else "X")
        if self._bus not in ("X", "Z"):
            return self._bus                  # legacy abutting: no bus letter
        feet = p["pauli"]
        inside = sum(1 for q in feet if q in self._retype)
        if inside == len(feet):
            # recoloured corridor: the M column flips letter AND colour
            # together (component ledger, docs §2.6), so the region's own
            # colour dictates the conjugate bus letter here
            return "X" if self._bus == "Z" else "Z"
        if inside:
            return None                       # straddles the recolour boundary
        return self._bus

    def direct_build(self):
        """DIRECT boundary construction — zero search, zero
        per-tile commutation tests.

        The patches' native weight-2 stabilizers (rule-0 seeds) anchor the
        boundary pattern; wall records' feet count as occupied.  Every other
        boundary candidate is visited outward from the anchors (multi-source
        BFS over the foot-sharing chain graph, lexicographic within each
        layer) and placed iff it shares NO data foot with an already-placed
        boundary stabilizer.  Sharing a foot IS the paper's "one lattice
        spacing" (skip); disjoint is "two spacings" (place): straight-edge
        alternation and the concave weight-3 rule are local instances of this
        single test, and the convex-corner criterion stays with rule 6 in the
        caller.  The engine's sub-product exclusion is deliberately absent —
        the rules are supposed to make it unreachable (cross-checked against
        :meth:`build` by the regression harness).

        Returns the same ``(selected, corner_picks, metric, w1)`` tuple as
        :meth:`build`; the metric is computed ONCE at the end (it gates the
        caller's phase/cut ladders, it steers nothing here)."""
        n, isv = self.n, self.isv
        selected = []
        placed_feet = set()
        placed_pauli = {}                          # qubit -> letter already laid
        for p in self.native_seed:                 # rule 0: anchors, verbatim
            selected.append(p)
            placed_feet.update(p["pauli"])
            placed_pauli.update(p["pauli"])
        for e in self.forced:                      # wall/kf records own their feet
            if len(e["pauli"]) < 4:
                placed_feet.update(e["pauli"])
                placed_pauli.update(e["pauli"])

        # parity-propagating BFS over the whole boundary graph.  Nodes: every
        # pool position (anchors included; forbidden positions participate as
        # TRANSIT nodes — a neighbouring patch's own lobe sits there, so the
        # alternation parity flows through but nothing of ours is placed).
        # Edges: sharing a data foot (one lattice spacing), with a Chebyshev-2
        # fallback so the wave crosses seam endpoints where no common foot
        # survives.  A node at even chain-distance from an anchor is a "place"
        # slot, odd is a "skip" slot; a foot conflict with anything already
        # placed vetoes regardless (waves meeting out of phase — the parity
        # law's edge case — resolve to skip, like the paper's spacing rule).
        # ---- transient chain graph (the user's anchored-walk counting) ----
        # nodes: integer slots (pool candidates, untouched records) plus ONE
        # node per wall band (the single-slot rule).  Edges: sharing a data
        # foot — physical one-lattice-spacing adjacency ONLY (unstitched gaps
        # break chains; no geometric shortcuts).  The count walks outward
        # from the patch-native anchors: even = place slot, odd = skip slot;
        # a wall band advances the count by exactly one.
        nodes = {p["syn"]: p for p in self.pool}
        for p in self.native_seed:
            nodes.setdefault(p["syn"], p)
        # wall band -> one node PER FOOT-LINE (the pair of feet at one
        # along-wall coordinate).  A crossing chain pays exactly one slot
        # (enter 1, leave 0); consecutive lines of the same wall are one
        # slot apart along the wall.  A single blob node is wrong twice
        # over: symmetric costs make a crossing cost two slots, and any
        # shared node lets same-side tiles far apart along the wall borrow
        # a one-slot shortcut through it.
        band_feet = {}                       # line id -> foot pair
        for e in self.forced:
            if e.get('kf'):
                feet = set(map(tuple, e["pauli"]))
                xs = frozenset(x for x, _ in feet)
                ys = frozenset(y for _, y in feet)
                if len(xs) == 2:             # vertical wall: lines by y
                    for yv in sorted(ys):
                        band_feet.setdefault(('bl', 'v', xs, yv), set()).update(
                            (x, yv) for x in xs)
                else:                        # horizontal wall: lines by x
                    for xv in sorted(xs):
                        band_feet.setdefault(('bl', 'h', ys, xv), set()).update(
                            (xv, y) for y in ys)
        by_foot = {}
        for s, p in nodes.items():
            for q in p["pauli"]:
                by_foot.setdefault(q, []).append(s)
        for bid, bf in band_feet.items():
            for q in bf:
                by_foot.setdefault(q, []).append(bid)

        def _feet(nid):
            return band_feet[nid] if nid in band_feet else set(nodes[nid]["pauli"])

        def _line_neighbours(nid):
            # consecutive foot-lines of the SAME wall: one slot apart
            _tag, ax, span, v = nid
            for dv in (-2, 2):
                nb = ('bl', ax, span, v + dv)
                if nb in band_feet:
                    yield nb

        # 0-1 BFS: a STRAIGHT neighbour step advances the count by one slot;
        # a DIAGONAL corner step keeps it ((cx+cy)/2 unchanged — corner slots
        # share the corner foot at the SAME parity, the corner-w3 lesson);
        # entering a wall band costs one slot (the single-slot rule)
        from collections import deque
        def _w(s, t):
            if t in band_feet:
                return 1                  # entering the line: the single slot
            if s in band_feet:
                return 0                  # leaving is free - one slot in total
            return 0 if (abs(s[0] - t[0]) == 2 and abs(s[1] - t[1]) == 2) else 1
        depth = {}
        dq = deque()
        for s in sorted(s for s in self.seed_syns if s in nodes):
            depth[s] = 0
            dq.append(s)
        while dq:
            s = dq.popleft()
            nbrs = [t for q in _feet(s) for t in by_foot.get(q, ()) if t != s]
            if s in band_feet:
                nbrs.extend(_line_neighbours(s))
            for t in nbrs:
                w = 1 if (s in band_feet and t in band_feet) else _w(s, t)
                nd = depth[s] + w
                if t not in depth or nd < depth[t]:
                    depth[t] = nd
                    (dq.appendleft if w == 0 else dq.append)(t)
        order = sorted((s for s in depth if s not in band_feet),
                       key=lambda s: (depth[s], s))
        order += sorted(set(nodes) - set(depth))        # anchorless: fallback

        walked = set(depth)
        for s in order:
            if s in self.seed_syns:
                continue                            # anchors already placed
            if s in self.forbidden:
                continue                            # neighbour-owned site
            p = nodes[s]
            if s in walked:
                # letter-driven placement (slice-3 bus-boundary rule): the
                # slot's letter is the REGION's letter (bus letter, conjugate
                # inside a recoloured region — the M column flips letter and
                # colour together, so the colour shortcut tracks it), XOR'd
                # by the kf-crossing parity (a kf wall flips the letter only,
                # invisible to colour).  The walk count sequences the scan
                # and decides only the letter-undecided slots (recolour-
                # boundary straddlers, mixed tiles).
                if p["type"] in "XZ":
                    letter = self._seg_letter(p)
                    if letter is None:
                        if depth[s] % 2 == 1:
                            continue            # letter-undecided: count rules
                    else:
                        if p["type"] != letter:
                            continue
                        if any(placed_pauli.get(q, P) != P
                               for q, P in p["pauli"].items()):
                            continue            # region-junction guard: a
                                                # shared foot with a DIFFERENT
                                                # letter anticommutes; corner
                                                # slots legitimately share a
                                                # same-letter foot
                else:
                    if depth[s] % 2 == 1:
                        continue
                    if not placed_feet.isdisjoint(p["pauli"]):
                        continue                # mixed tile foot veto
            else:
                # anchorless chain: wall-free by construction — the
                # checkerboard-letter shortcut is exact there
                if p["type"] in "XZ":
                    letter = self._seg_letter(p)
                    if letter is None or p["type"] != letter:
                        continue
                else:
                    if not placed_feet.isdisjoint(p["pauli"]):
                        continue
            selected.append(p)
            placed_feet.update(p["pauli"])
            placed_pauli.update(p["pauli"])

        # acceptance metric, computed once (same contract as build())
        B = _IntBasis()
        svecs = list(self.fvecs)
        for v in self.fvecs:
            B.add(v)
        for p in selected:
            v = isv(p["pauli"])
            svecs.append(v)
            B.add(v)
        w1 = []
        for q in self.data:
            for P in "XZ":
                v = isv({q: P})
                if B.reduce(v) != 0 and all(_icommute(v, s, n) for s in svecs):
                    w1.append(q)
        N = len(self.target)
        deficit = (len(self.data) - B.rank) - (N - 1)
        joint = 0
        for v in self.repv:
            joint ^= v
        metric = (max(deficit, 0), len(w1), 0 if B.contains(joint) else 1)
        return selected, [], metric, w1


def _direct_metric(bl, selected):
    """Acceptance metric over the CURRENT forced+selected sets — vectors
    only, no re-scan (the local corner-cut semantics modifies a forced
    plaquette in place and re-checks acceptance without rebuilding)."""
    B = _IntBasis()
    svecs = []
    for p in bl.forced:
        v = bl.isv(p["pauli"])
        svecs.append(v)
        B.add(v)
    for p in selected:
        v = bl.isv(p["pauli"])
        svecs.append(v)
        B.add(v)
    w1 = []
    for q in bl.data:
        for P in "XZ":
            v = bl.isv({q: P})
            if B.reduce(v) != 0 and all(_icommute(v, s, bl.n) for s in svecs):
                w1.append(q)
    N = len(bl.target)
    deficit = (len(bl.data) - B.rank) - (N - 1)
    joint = 0
    for v in bl.repv:
        joint ^= v
    return (max(deficit, 0), len(w1), 0 if B.contains(joint) else 1), w1


def _convex_corner_cuts_local(bl, selected, w1):
    """Fig 34 label 2, the user's EXACT scope (2026-07-31): a convex bus
    corner is cut iff a NEIGHBOURING boundary stabilizer — one sharing a
    data foot with the corner plaquette, i.e. at ONE lattice spacing —
    exists; two spacings (no shared foot) means no cut.  A weight-1
    leftover on a bus corner also cuts.  (The retired Chebyshev-2 test
    was wider: in dense geometries it counted unrelated pieces at
    diagonal distance as neighbours and over-cut.)"""
    corner_q = _corner_qubits(set(bl.data)) - bl.patchq
    cuts = {q for q in w1 if q in corner_q}
    for q in corner_q:
        w4 = next((p for p in bl.forced if q in p["pauli"]), None)
        if w4 is None:
            continue
        w4_feet = set(w4["pauli"])
        for p in selected:
            feet = set(p["pauli"])
            if q in feet:
                continue
            if w4_feet & feet:
                cuts.add(q)      # one lattice spacing: shares a foot
                break
    return cuts


def _construct_phase(placed, target, orient, dset, retype, phase, forbidden=frozenset(),
                     native_lock=False, bus=None, conj_names=None, extra_forced=(),
                     retire_lobes=frozenset()):
    """One deterministic construction attempt (direct scan + LOCAL corner
    cuts).  Returns ``("ok", checks, logicals, x_obs, applied_cuts)`` or
    ``("fail", reason)``."""
    bl = _Builder(placed, target, orient, dset, retype, phase, forbidden,
                  native_lock=native_lock, bus=bus, conj_names=conj_names,
                  extra_forced=extra_forced, retire_lobes=retire_lobes)
    if bl.fail is not None:
        return ("fail", bl.fail)

    applied_cuts = set()
    # 10.4 direct construction, slice-3 letter rule: a slot's letter is
    # its REGION's letter (bus letter, conjugate inside a recoloured
    # region), so the placement is a lookup, not a propagated count; the
    # walk count sequences the scan and decides only letter-undecided
    # slots.  Wall-carrying builds count each kf band foot-line as ONE
    # slot (enter 1 / leave 0); retired-lobe reuse builds go through the
    # same rules.  (The legacy greedy engine and its rule-5 re-anchoring
    # are deleted: census showed zero reachable customers.)
    selected, picks, metric, w1 = bl.direct_build()
    if metric != (0, 0, 0):
        # rule 6, LOCAL semantics (design decision 2026-07-31): a convex
        # corner that fails the neighbour-spacing test is cut IN PLACE
        # - the corner plaquette drops that foot (w4 -> w3), nothing
        # else moves, no region rebuild.  The old rebuild ladder
        # re-clipped the outline after every cut, manufactured phantom
        # corners at the notch and cascaded (the 'corner-cut
        # avalanche' was an artifact of that loop, not geometry).
        cuts = _convex_corner_cuts_local(bl, selected, w1)
        for q in sorted(cuts):
            if any(q in p["pauli"] for p in selected):
                continue          # a laid boundary piece owns this foot
            w4 = next((p for p in bl.forced
                       if q in p["pauli"] and len(p["pauli"]) >= 4), None)
            if w4 is None:
                continue
            del w4["pauli"][q]    # w4 -> w3, in place
            applied_cuts.add(q)
        if applied_cuts:
            bl.data = [q for q in bl.data if q not in applied_cuts]
            metric, w1 = _direct_metric(bl, selected)
        if metric != (0, 0, 0) and applied_cuts:
            # variant 2 - cut-then-RELAY: the region is FIRST reduced by
            # the round-one cut demands, then the boundary is laid ONCE
            # on the final region (fresh scan; no further cuts, so no
            # iteration and no cascade).  Some wall-adjacent layouts
            # only form the JOINT product in the relaid pattern
            # (measured: the role-switch south wall), while the flanked
            # corner family needs the kept pattern (variant 1).  Both
            # variants are bounded rule evaluations.
            bl2 = _Builder(placed, target, orient,
                           set(bl.data), {q for q in retype
                                          if q in set(bl.data)},
                           phase, forbidden, native_lock=native_lock,
                           bus=bus, conj_names=conj_names,
                           extra_forced=extra_forced,
                           retire_lobes=retire_lobes)
            if bl2.fail is None:
                sel2, picks2, met2, w12 = bl2.direct_build()
                if met2 == (0, 0, 0):
                    bl, selected, metric, w1 = bl2, sel2, met2, w12
        if metric != (0, 0, 0):
            import os as _os
            if _os.environ.get('LIGHTSTIM_DEBUG_JOINT'):
                import sys as _sys
                _B = _IntBasis()
                for _p2 in bl.forced:
                    _B.add(bl.isv(_p2["pauli"]))
                for _p2 in selected:
                    _B.add(bl.isv(_p2["pauli"]))
                _joint = 0
                for _v in bl.repv:
                    _joint ^= _v
                _res = _B.reduce(_joint)
                _n2 = bl.n
                _terms = []
                for _k, _q in enumerate(bl.data):
                    _x = (_res >> _k) & 1
                    _z = (_res >> (_n2 + _k)) & 1
                    if _x or _z:
                        _terms.append(
                            f"{'Y' if _x and _z else ('X' if _x else 'Z')}"
                            f"@{_q}")
                print(f"[joint] phase={phase} residual({len(_terms)}): "
                      f"{' '.join(_terms[:14])}", file=_sys.stderr)
            return ("fail", f"direct-scan acceptance shortfall {metric} "
                            f"(rank deficit, weight-1 count, joint "
                            f"missing) after {len(applied_cuts)} local "
                            f"corner cuts (phase {phase})")


    checks = []
    for p in bl.forced + selected:
        c = dict(p)
        c["corners"] = sorted(c["pauli"])
        checks.append(c)
    logicals = [(nm, bl.reps[nm][0], bl.reps[nm][1]) for nm, _ in target]
    # tracked observable: a BUS-basis representative — the memory experiment runs in the
    # bus basis, so the tracked logical must be of that type.  The bus is the PLANNED
    # ``bus`` when given, else the majority target basis (tie -> X).
    if bus is None:
        n_x = sum(1 for _, P, _ in logicals if P == "X")
        bus = "X" if n_x >= len(logicals) - n_x else "Z"
    x_obs = next((s for nm, P, s in logicals if P == bus), logicals[0][2])
    return ("ok", checks, logicals, x_obs, frozenset(applied_cuts))


def rule_based_joint_checks(placed, target, orient, data, retype, d, max_cut=4,
                            forbidden=frozenset(), native_lock=False, bus=None,
                            conj_names=None, extra_forced_fn=None,
                            retire_lobes=frozenset()):
    """Deterministic construction on the routed region (phases 0/1; corner
    cuts are applied LOCALLY inside each attempt per the user's 2026-07-31
    ruling — no rebuild ladder).
    Returns ``dict(checks, data, cut, phase, logicals, x_observable, reason="")`` on
    success, else ``dict(checks=None, reason=<why per phase>)``.  Pure function of the
    geometry — no randomness.

    ``native_lock`` (rule 0, default OFF in the abutting pipeline — in that
    geometry the greedy's alternation is sometimes load-bearing for mixed-joint
    distance laws; the seam-column pipeline passes True): every target patch whose interior is
    compatible with the global checkerboard (not retyped, phase-matched) keeps its
    TEXTBOOK standalone construction verbatim — its native boundary semicircles are
    forced picks and non-native tiles inside the patch are expelled, so a boundary-
    alternation tie can never relocate an observable's guard.  The facing semicircle
    is the sole exception: when its forced-bulk extension into seam/corridor qubits
    exists, the extension replaces it (joint-measurement rank demands this).  If the
    lock is unsatisfiable in every phase/strategy, the ladder is retried without it
    (noted in ``reason``).
    """
    reasons = []

    def attempt(phase, lock=True):
        rt = {q for q in retype if q in set(data)}
        ef = extra_forced_fn(phase) if extra_forced_fn is not None else ()
        res = _construct_phase(placed, target, orient, set(data), rt, phase,
                               forbidden, native_lock=lock, bus=bus,
                               conj_names=conj_names, extra_forced=ef,
                               retire_lobes=retire_lobes)
        if res[0] == "ok":
            lcut = set(res[4]) if len(res) > 4 else set()
            return dict(checks=res[1],
                        data=sorted(set(data) - lcut),
                        cut=tuple(sorted(lcut)),
                        phase=phase, logicals=res[2], x_observable=res[3],
                        reason="")
        reasons.append(f"phase {phase}: {res[1]}")
        return None

    # phases 0/1 with the native lock HARD (an unsatisfiable lock is a
    # failed build); corner cuts happen LOCALLY inside the attempt, so no
    # cut ladder exists any more.  ``max_cut`` is kept in the signature
    # for API compatibility only.
    lock = bool(native_lock)
    for phase in (0, 1):
        out = attempt(phase, lock=lock)
        if out is not None:
            return out
    return dict(checks=None, data=sorted(data), cut=(), phase=None,
                logicals=None, x_observable=None, reason="; ".join(reasons))


# =============================================================================
# SECTION: subset_routing
# =============================================================================

# Subset joint lattice-surgery measurement with obstacle-aware, ``d``-wide corridor routing.
#
# Given many placed patches, measure the joint ``M(∏ᵢ P̄ᵢ)`` of only a chosen **subset**; the
# non-target patches are **obstacles** the routed ancilla bus must not overlap.  This module holds the
# **core routing logic** — the coordinate/coarse-cell model, the ``networkx`` corridor graph, the
# ``d``-wide channel + obstacle handling, the shortest-path-tree routing, and the path→corridor
# conversion — so the demo notebook only imports and calls.
#
# The router is **propose-and-verify**: it builds an obstacle-aware ``d``-wide corridor graph,
# enumerates candidate shortest-path trees connecting the targets, and the **existing GF(2) oracle**
# (the :mod:`.multi_patch` physics layer, reused unchanged) gates every candidate — returning the first
# **verified** :class:`.MultiPatchLayout`, or an honest ``"no_verified_route"`` (it never silently
# mis-measures, and it never routes a corridor through an obstacle).
#
# **Scope.**  The current physics layer verifies a **straight X-bus (trunk) with Z-targets attached
# perpendicular through mixed (XZ) walls (any side)** plus simple end-bends.  It **rejects** most
# *arbitrary bent* trunks (they fail ``joint ∈ span``).  So the router routes within that family; a
# general bent-bus / rectilinear-Steiner router is future work.
#
# Geometry model: a coarse grid of ``d×d`` cells at pitch ``2d``.  Each patch occupies one cell;
# corridors run through **empty** cells.  The joint code's ``data`` = target cells + corridor-tree
# cells — the obstacles are **not** in ``data`` (they are separate idle patches, only forbidden
# coordinates for the bus), which is exactly why the code measures the *subset* joint and not the
# all-patch joint.
#

import itertools
from dataclasses import dataclass, field

import networkx as nx


_ALL_ROUTING = ["PatchSpec", "cell", "origin_of", "route_and_build",
           "acceptance", "acceptance_of_layout", "ACCEPTANCE_ITEMS", "collision_report",
           "SubsetRoute"]

NEIGH = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _xor_ints(vs):
    x = 0
    for v in vs:
        x ^= v
    return x
_LR = [(1, 0), (-1, 0)]          # X-faces of an X_horizontal patch
_TB = [(0, 1), (0, -1)]          # X-faces of an X_vertical patch


def _pitch(d, seam=False):
    """Coarse-grid pitch: ``2d`` for the abutting design, ``2d+2`` for the SEAM-COLUMN
    design (standard-LS style: one freshly-initialized data column/row between every
    pair of edge-adjacent occupied cells; progress §11)."""
    return 2 * d + 2 if seam else 2 * d


def cell(a, b, d, seam=False):
    """The ``d×d`` data-qubit footprint of coarse cell ``(a, b)`` (rotated frame)."""
    P = _pitch(d, seam)
    x0, y0 = 1 + P * a, 1 + P * b
    return {(x, y) for x in range(x0, x0 + 2 * d, 2) for y in range(y0, y0 + 2 * d, 2)}


def origin_of(a, b, d, seam=False):
    """The data-qubit **origin** (bus-facing ``(1,1)``-corner) of coarse cell ``(a, b)`` —
    i.e. the :attr:`PatchSpec.origin` that places a distance-``d`` patch on that cell.
    Inverse of :func:`cell_index`.  Use it to write subset-routing patches in the same
    ``PatchSpec(name, origin, distance, orientation)`` form as the N-patch API.
    ``seam=True`` uses the seam-column grid (pitch ``2d+2``)."""
    P = _pitch(d, seam)
    return (1 + P * a, 1 + P * b)


def cell_index(origin, d, seam=False):
    """The coarse cell ``(a, b)`` a distance-``d`` patch placed at ``origin`` occupies.  Inverse of
    :func:`origin_of`.  Raises ``ValueError`` if ``origin`` is off the pitch grid (the coarse
    router places patches only on that grid; use :func:`origin_of` to land on it)."""
    P = _pitch(d, seam)
    ox, oy = origin
    if (ox - 1) % P or (oy - 1) % P:
        raise ValueError(f"origin {origin} is not on the pitch-{P} coarse grid for d={d}")
    return ((ox - 1) // P, (oy - 1) // P)


def _seam_qubits(occupied, d):
    """The seam data qubits joining every pair of edge-adjacent occupied coarse cells
    (seam-column grid, pitch ``2d+2``): a vertical column of ``d`` qubits between
    horizontal neighbours, a horizontal row between vertical neighbours."""
    P = 2 * d + 2
    out = set()
    for (a, b) in occupied:
        if (a + 1, b) in occupied:
            out |= {(P * (a + 1) - 1, P * b + 1 + 2 * j) for j in range(d)}
        if (a, b + 1) in occupied:
            out |= {(P * a + 1 + 2 * i, P * (b + 1) - 1) for i in range(d)}
    return out


def _specs_to_cells(patches, target=None, seam=False):
    """Map the user-facing :class:`PatchSpec` list onto the internal coarse-cell model.

    Returns ``(patch_at, orient, d)`` where ``patch_at[name] = (a, b)`` is the coarse cell whose
    ``d×d`` footprint equals the patch's placed data qubits, ``orient[name]`` is the declared
    orientation, and ``d`` is the (shared) code distance.  This is the single adapter every public
    entry point runs first, so the whole subset API speaks ``PatchSpec`` while the routing internals
    keep working in coarse cells.

    Raises ``TypeError`` / ``ValueError`` with a concrete reason when the specs can't be placed on the
    coarse grid — a non-``PatchSpec`` element (e.g. the old ``{name:(a,b)}`` cell dict), mixed
    distances, duplicate names, an origin off the pitch-``2d`` grid, or a bad orientation — or when
    ``target`` names an unknown patch, repeats one, or uses a non-``X``/``Z`` Pauli.
    """
    patches = list(patches)
    if not patches:
        raise ValueError("need at least one PatchSpec")
    if not all(isinstance(s, PatchSpec) for s in patches):
        raise TypeError("patches must be a list of PatchSpec(name, origin, distance, "
                        "orientation); the old {name: (a, b)} cell-dict form is no "
                        "longer accepted — build a PatchSpec per patch (origin_of(a, b, d) places "
                        "one on coarse cell (a, b)).")
    d = patches[0].distance
    if any(s.distance != d for s in patches):
        raise ValueError(f"all patches must share one distance, got {sorted({s.distance for s in patches})}")
    names = [s.name for s in patches]
    if len(set(names)) != len(names):
        raise ValueError(f"patch names must be unique, got {names}")
    patch_at, orient = {}, {}
    for s in patches:
        if s.orientation not in ("X_horizontal", "X_vertical"):
            raise ValueError(f"patch {s.name!r}: orientation must be 'X_horizontal'|'X_vertical', "
                             f"got {s.orientation!r}")
        try:
            patch_at[s.name] = cell_index(s.origin, d, seam)
        except ValueError:
            raise ValueError(
                f"patch {s.name!r}: origin {s.origin} is not on the coarse routing grid (pitch "
                f"{_pitch(d, seam)} for d={d}, seam={seam}).  Subset routing places patches on cells "
                f"whose bus-facing corner is origin_of(a, b, d, seam={seam}).")
        orient[s.name] = s.orientation
    if target is not None:
        tnames = [nm for nm, _ in target]
        unknown = [nm for nm in tnames if nm not in patch_at]
        if unknown:
            raise ValueError(f"target names {unknown} are not among the patches {names}")
        if len(set(tnames)) != len(tnames):
            raise ValueError(f"target lists a patch more than once: {tnames}")
        if any(P not in ("X", "Z") for _, P in target):
            raise ValueError(f"target paulis must be 'X' or 'Z', got {[P for _, P in target]}")
    return patch_at, orient, d




# -----------------------------------------------------------------------------
# corridor graph  (d-wide channel search + obstacle handling)
# -----------------------------------------------------------------------------

def _cheb(ab, cd):
    """King-move (Chebyshev) distance between two coarse cells."""
    return max(abs(ab[0] - cd[0]), abs(ab[1] - cd[1]))


def _legal_attach_groups(patch_at, target, orient, corridor):
    """Per-target group of corridor cells adjacent through a PARALLEL-LAW
    legal seam (the same rule the stitching policy applies).  Returns
    (groups, missing) — ``missing`` lists targets with no legal cell."""
    groups, missing = [], []
    for nm, P in target:
        pa = tuple(patch_at[nm])
        cells = []
        for da, db in NEIGH:
            c = (pa[0] + da, pa[1] + db)
            if c not in corridor:
                continue
            seam_ew = (da != 0)
            want = 'X_horizontal' if (P == 'Z') == seam_ew else 'X_vertical'
            if orient[nm] == want:
                cells.append(c)
        groups.append(cells)
        if not cells:
            missing.append(nm)
    return groups, missing


def path_to_corridor(tree_cells, placed, target, d, seam=False, patch_at=None, bus=None,
                     conj_names=None, flip_cells=None, skip_seam_pairs=None):
    """Convert a set of corridor **cells** into the joint code's ``(data, retype)``.

    ``data`` = the target patch cells ∪ the corridor cells' footprints.

    **Bus-basis rule**: the corridor's native basis is ``bus`` when given, else the
    *majority* target basis (a tie keeps the historical X-bus).  Only the
    non-bus-basis patches attach through mixed (XZ) domain walls — so a pure-X or
    pure-Z joint uses NO mixed stabilizer at all (the pure-Z corridor is the exact CSS
    dual of the pure-X one).  Passing an explicit ``bus`` forces the native basis to a
    PLANNED choice (used to keep a shared patch's conjugation status consistent across
    a PPM sequence); the physics is valid for either basis (conjugating the non-bus
    patches reduces the joint to a pure-bus merge), the minority basis merely raises
    more mixed 'M' walls.

    ``conj_names``: the target patches whose LIVE construction is the colour-conjugate
    registration.  A patch's region is retyped iff it is conjugate-registered — its
    physical checks really are colour-flipped, and the wall the retype boundary raises
    is what reconciles them with the corridor.  ``None`` (the historical default)
    infers conjugation from the measured Pauli: minority-basis patches are the
    conjugate-registered ones (first-use minority allocation), which reproduces the
    original letter rule byte for byte.  Passing the set explicitly decouples the two:
    a conjugate patch measuring the BUS basis (its recoloured seam still needs the
    wall) routes without any rotation.

    ``retype`` encodes this for :func:`._bent_plaquettes`: for EITHER bus letter,
    exactly the conjugate-registered cells (plus ``flip_cells``) are flipped.  The
    complement is an exact STATIC dual — flipping *all* of ``data`` equals flipping
    none at the opposite checkerboard ``phase`` — but patches carry measurement
    history, so the dual is not free to choose; the direct convention is the
    only valid one.  Both checkerboard phases are tried downstream.
    """
    tnames = [nm for nm, _ in target]
    data = set()
    for nm in tnames:
        data |= placed[nm]
    for c in tree_cells:
        data |= cell(*c, d, seam)
    if seam:                    # seam-column design: join every adjacent occupied pair
        if patch_at is None:
            raise ValueError("path_to_corridor(seam=True) needs patch_at (coarse cells)")
        occupied = set(map(tuple, tree_cells)) | {tuple(patch_at[nm]) for nm in tnames}
        data |= _seam_qubits(occupied, d)
        # a walled junction gets NO seam qubits (the stretched-stabilizer band
        # lives in the gap; the wall records bridge the two segments)
        for pair in (skip_seam_pairs or ()):
            data -= _seam_qubits({tuple(pair[0]), tuple(pair[1])}, d)
    xnames = [nm for nm, P in target if P == "X"]
    znames = [nm for nm, P in target if P == "Z"]
    if bus is None:
        bus = "X" if len(xnames) >= len(znames) else "Z"
    if conj_names is None:      # historical inference: conjugate ⟺ minority basis
        conj_names = frozenset(znames if bus == "X" else xnames)
    flip = set().union(*[placed[nm] for nm in tnames if nm in conj_names]) \
        if conj_names else set()
    if flip_cells:
        # segment recolouring: these corridor cells join the flipped-gauge side,
        # WITH the seam qubits interior to the flipped region (flipped cell to
        # flipped cell / flipped cell to conjugate patch) — a plain stitch inside
        # one colour region flips as a whole
        fcs = {tuple(c) for c in flip_cells}
        for c in fcs:
            flip |= cell(*c, d, seam)
        if seam:
            fents = fcs | {tuple(patch_at[nm]) for nm in tnames
                           if nm in conj_names}
            flip |= _seam_qubits(fents, d) & data
    # DIRECT convention for both bus letters (stage-A hardening): retype is
    # exactly the flipped side - conj-registered footprints plus flip_cells.
    # The old X-bus complement shortcut realized the colour-conjugate DUAL of
    # the intended layout; statically equivalent, but a patch with measurement
    # history must keep its own letters, so the dual is not free to choose.
    retype = {q for q in data if q in flip}
    return data, retype


# -----------------------------------------------------------------------------
# physics-layer reuse: a routed (data, retype) region -> a verified MultiPatchLayout
# -----------------------------------------------------------------------------

def _assemble_region(placed, target, orient, data, retype, d, seed=0, max_trials=5000, max_cut=4,
                     forbidden=frozenset(), native_lock=False, bus=None, conj_names=None,
                     extra_forced_fn=None, extra_connect=frozenset(),
                     retyped_bus=frozenset(), retire_lobes=frozenset()):
    """Hand a routed region to the **deterministic rule-based** physics layer.

    The stabilizers are constructed by :func:`.deterministic_checks.rule_based_joint_checks`
    (the documented construction rules: forced bulk, alternating-spacing
    boundary, concave/convex corner rules with rule-driven corner cuts) — no randomized
    search.  ``seed`` / ``max_trials`` are accepted for backward compatibility and ignored.
    Returns a :class:`.MultiPatchLayout` (whose ``data`` may be smaller than the input when
    the convex-corner rule cut bus-corner qubits), or ``None`` if the rules cannot host this
    geometry.  The oracle decision itself is ``MultiPatchLayout.verify()``.
    """
    data = sorted(data)
    # a walled junction leaves a data gap; its would-be seam qubits are handed
    # in as extra_connect so the two segments count as one region
    if not _connected(set(data) | set(extra_connect)):
        return None
    rb = rule_based_joint_checks(placed, target, orient, data, set(retype), d, max_cut=max_cut,
                                 forbidden=forbidden, native_lock=native_lock, bus=bus,
                                 conj_names=conj_names, extra_forced_fn=extra_forced_fn,
                                 retire_lobes=retire_lobes)
    if rb["checks"] is None:
        return None
    log_pairs = [(P, sup) for _, P, sup in rb["logicals"]]
    # orientation-domain map for the hook-benign scheduler: patch cells carry
    # their declared orientation; bus/corridor cells follow the majority
    tally = {}
    for nm, _ in target:
        o = orient[nm]
        tally[o] = tally.get(o, 0) + 1
    bus_orient = max(sorted(tally), key=lambda o: tally[o])
    domains = {}
    for nm, _ in target:
        for q in placed[nm]:
            domains[q] = orient[nm]
    for q in rb["data"]:
        domains.setdefault(q, bus_orient)
    return MultiPatchLayout(distance=d, data=rb["data"], checks=rb["checks"],
                            logicals=rb["logicals"], x_observable=rb["x_observable"],
                            readout_chain=_readout_chain(rb["data"], rb["checks"], log_pairs),
                            target=list(target), domains=domains, bus=bus,
                            retyped=frozenset(retyped_bus))


# -----------------------------------------------------------------------------
# the router
# -----------------------------------------------------------------------------

@dataclass
class SubsetRoute:
    """Result of :func:`route_and_build`.  ``status == "ok"`` iff ``layout`` is a verified joint.

    On a **failure** an obstacle-free corridor may still exist but be un-hostable by the physics
    layer — ``attempted`` / ``attempted_arms`` carry the shortest such corridor (empty only when
    ``status == "no_path"``) so a caller can *draw the route that was found* and label it correctly
    (a rejected corridor is **not** "no route").
    """
    status: str    # "ok" | "no_path" (unreachable OR only disconnected corridors) | "no_verified_route"
    message: str = ""
    layout: object = None                             # MultiPatchLayout when status == "ok"
    root: str = None
    arms: dict = field(default_factory=dict)          # z-name -> corridor-cell path (VERIFIED bus)
    tree: set = field(default_factory=set)            # corridor cells of the VERIFIED bus
    attempted: set = field(default_factory=set)       # corridor cells of the shortest candidate (any status)
    attempted_arms: dict = field(default_factory=dict)  # z-name -> shortest candidate path
    data: set = field(default_factory=set)            # data qubits of the routed region
    placed: dict = field(default_factory=dict)        # name -> patch cells (ALL patches)
    target: list = field(default_factory=list)
    obstacles: list = field(default_factory=list)     # non-target patch names
    obstacle_fp: set = field(default_factory=set)     # non-target patch data qubits
    corridor: set = field(default_factory=set)        # all corridor-eligible cells
    tried: int = 0
    how: str = "standard"                             # "standard" | "corner-cut" (route_and_build)
    cut: tuple = ()                                   # convex-corner qubits removed (corner-cut only)
    n_walls: int = 0                                  # stretched (kf) seam walls in the layout - band
                                                      # apparatus is real cost, chargeable like cells

    @property
    def ok(self):
        return self.status == "ok"


def _obstacle_ancillas(patches, tnames):
    """The ancilla sites actually USED by the non-target (idle obstacle) patches.

    Each idle patch keeps its standalone construction; its selected syndrome sites are
    physical qubits the routed joint code may not re-use.  These are handed to the rule
    constructor as ``forbidden`` boundary positions, so a corridor sharing an ancilla
    line with an idle neighbour interleaves with it instead of colliding — edge-adjacent
    placement (keepout=0) is then physically sound whenever the parity allows.
    """
    out = set()
    for s in patches:
        if s.name in tnames:
            continue
        for c in place_patch(s)["checks"]:
            out.add(tuple(int(v) for v in c["syn"]))
    return frozenset(out)


def _seam_orientation_violation(patch_at, target, orient, tree,
                                skip_pairs=frozenset()):
    """Seam-design orientation rule: the measured logical must run PARALLEL to
    every seam its patch attaches through.  An E/W seam runs vertically
    (measuring Z needs 'X_horizontal', X needs 'X_vertical'); a N/S seam runs
    horizontally (the transpose).  A perpendicular logical silently yields a
    fallback layout with circuit-level distance 1.  Returns the violation
    message, or None.

    ``skip_pairs``: frozensets of UNSTITCHED cell pairs (no_stitch) — those
    adjacencies carry no seam, so the rule does not apply to them."""
    for nm, P in target:
        pa = tuple(patch_at[nm])
        for tc in tree:
            da, db = tc[0] - pa[0], tc[1] - pa[1]
            if abs(da) + abs(db) != 1:
                continue                        # not edge-adjacent: no seam
            if frozenset((pa, tuple(tc))) in skip_pairs:
                continue                        # unstitched: no seam there
            seam_ew = (da != 0)
            want = ('X_horizontal' if (P == 'Z') == seam_ew else 'X_vertical')
            if orient[nm] != want:
                return (f"{nm}: the measured logical must run PARALLEL to the "
                        f"seam — measuring {P} through a{'n E/W' if seam_ew else ' N/S'} "
                        f"seam needs orientation {want!r}, got {orient[nm]!r}")
    # direct patch-patch adjacency (zero-cell corridor, route=[]): the same
    # rule applies to the seam BETWEEN two target patches — unguarded, a
    # perpendicular logical silently builds the fallback layout at circuit
    # distance 1 (that is the rule table's row-2/3 wall territory, not a
    # plain merge)
    tgt = list(target)
    for i, (n1, P1) in enumerate(tgt):
        c1 = tuple(patch_at[n1])
        for n2, P2 in tgt[i + 1:]:
            c2 = tuple(patch_at[n2])
            da, db = c2[0] - c1[0], c2[1] - c1[1]
            if abs(da) + abs(db) != 1:
                continue
            if frozenset((c1, c2)) in skip_pairs:
                continue                        # unstitched: no seam there
            seam_ew = (da != 0)
            for nm, P in ((n1, P1), (n2, P2)):
                want = ('X_horizontal' if (P == 'Z') == seam_ew
                        else 'X_vertical')
                if orient[nm] != want:
                    return (f"{nm}: the measured logical must run PARALLEL to "
                            f"the patch-patch seam with its neighbour — "
                            f"measuring {P} through a"
                            f"{'n E/W' if seam_ew else ' N/S'} seam needs "
                            f"orientation {want!r}, got {orient[nm]!r} (a "
                            f"same-Pauli different-type pair needs the "
                            f"stretched-stabilizer wall, not a plain merge)")
    return None


def probe_wall_precheck(target, orient, conj_names, bus):
    """TREE-INDEPENDENT slice of the seam-table dispatch verdict.

    Returns False when the candidate is certainly infeasible for EVERY
    corridor: the wall classification depends only on (P, bus, conj,
    orientation), a wall-row target can never plain-stitch, and the
    guards mirrored here decline the due wall on any tree.  The step-1
    planner uses this to skip whole route probes (measured: on the
    30-patch 6-body Z step ~124/128 candidates are infeasible and each
    burned a ~4 s full tree scan — the wrong-bus half alone is 64
    candidates, all ≥2-direct6).  Every rule here MUST stay a verbatim
    mirror of _table_wall_dispatch — tree-dependent guards (terminal
    cell, exit handedness, same-cell, ns_pairs) stay out."""
    bus_eff = bus
    if bus_eff is None:
        n_x = sum(1 for _, P in target if P == "X")
        bus_eff = "X" if n_x >= len(target) - n_x else "Z"
    conj_eff = conj_names if conj_names is not None else \
        frozenset(nm for nm, P in target if P != bus_eff)
    snake = [nm for nm, P in target if P == bus_eff and nm in conj_eff]
    direct6 = [nm for nm, P in target
               if P != bus_eff and nm not in conj_eff] \
        if conj_names is not None else []
    if not snake and not direct6:
        return True                 # no wall due: no verdict from here
    # completeness mirror: a non-wall non-#1 target alongside due walls
    # makes dispatch bow out entirely, and the wall-row targets can never
    # plain-stitch -> infeasible on every tree
    for nm, P in target:
        if nm not in snake and nm not in direct6 \
                and not (P == bus_eff and nm not in conj_eff):
            return False
    if len(direct6) > 1:
        # verbatim mirror of _table_wall_dispatch's multi-wall gate
        # (lifted by default 2026-08-04; see the dispatch comment)
        import os as _os
        if _os.environ.get('LIGHTSTIM_FORBID_MULTI_WALL'):
            return False
    if snake and direct6:
        return False                # snake walls are vertical-only ->
                                    # the mix always hits the flip guard
    # (direct6 orientation is NOT checked here any more: both orientations
    # host — the 2026-08-02 chirality law is tree-dependent and lives in
    # _table_wall_dispatch._hosts_direct6)
    for nm, P in target:
        if nm in snake and orient.get(nm) != ('X_horizontal' if P == 'Z'
                                              else 'X_vertical'):
            return False            # legal faces on the horizontal axis:
                                    # the #4 horizontal band has no build
    return True


def _table_wall_dispatch(patch_at, target, orient, tree, conj_names, bus,
                         ns_pairs=frozenset()):
    """Post-routing seam-table dispatch — the user's TWO-BIT table (letter,
    colour), 2026-07-30.  TWO wall rows raise a stretched kf wall; TYPE
    never triggers one (same-letter same-colour type-split seams are plain
    row #1).  Step 2 never re-registers a patch: ``conj_eff`` is passed
    through, not grown.

      #4 native (letter same, colour diff): a conj-registered target
         measuring the BUS letter — the snake family.  On a vertical seam
         the corridor painting flips (g = 1: #4->#6 and #1->#7 wholesale).
      #6 native (letter diff, colour same): a STANDARD-colour target
         measuring the OTHER letter (design decision 2026-07-30: the mixed
         kf wall is the row's construct — the spatial-Hadamard interface
         turns the letter at the seam).  The wall hardware is the SAME
         local-rule family; the corridor never flips for it.  Fires only
         under an EXPLICIT ``conj_names`` — with ``conj_names=None`` the
         historical minority-conjugate inference keeps the legacy plain
         path (C4 mixed cells).

    Returns ``(arm_walls, conj_eff, flip_corridor)`` or ``None`` when no
    wall is due; ``(None, conj_eff, False)`` when a due wall cannot be
    hosted on this tree."""
    bus_eff = bus
    if bus_eff is None:
        n_x = sum(1 for _, P in target if P == "X")
        bus_eff = "X" if n_x >= len(target) - n_x else "Z"
    conj_eff = conj_names if conj_names is not None else \
        frozenset(nm for nm, P in target if P != bus_eff)
    # #4-native walls: colour-diff (conj) targets measuring the bus letter
    snake_nms = sorted(nm for nm, P in target
                       if P == bus_eff and nm in conj_eff)
    # #6-native walls: standard-colour targets measuring the other letter
    direct6_nms = sorted(nm for nm, P in target
                         if P != bus_eff and nm not in conj_eff) \
        if conj_names is not None else []
    wall_nms = snake_nms + direct6_nms
    if not wall_nms:
        return None
    # completeness: every NON-wall target must sit on row #1 (letter same,
    # colour same — plain stitch).  A leftover (T,T)/#7 seam in the same
    # joint has no verified construction alongside these walls (measured:
    # the layout passes the group oracle but the OBSERVABLE goes
    # non-deterministic) — decline to the legacy path instead.
    for nm, P in target:
        if nm not in wall_nms and not (P == bus_eff and nm not in conj_eff):
            return None
    tset = {tuple(c) for c in tree}
    # wall seams live on the LIVE-frame parallel-law-legal faces — the same
    # frame _auto_stitch judges plain seams in.  A wall reconciles the
    # CORRIDOR side of the seam (#4: the colour bit, #6: the bus letter),
    # never the target's own seam letter: a seam that is perpendicular in
    # the live frame is an ODD row — illegal, no wall due (measured: the
    # registered-frame faces admitted a S-seam #4 on a rotate_90'd conj
    # target and both construction phases refused it — 20-patch q17)
    groups_all, _miss = _legal_attach_groups(patch_at, target, orient, tset)
    legal_of = {nm: set(g) for (nm, _), g in zip(target, groups_all)}
    aw, snake_vertical = [], False

    def _hosts_direct6(nm, pc, wc):
        """Tree-dependent #6 hosting law (24/24 measured matrix, 2026-08-02:
        both orientations × all four sides × all exits, each combo built with
        a stubbed dispatch and checked verify + p0 + graphlike distance).
        The walled cell must be TERMINAL (degree 1 — the joint product only
        splices through an end wall; the interior middle-wall case measured
        (0,0,1) acceptance), and the bus may leave it straight ahead or with
        the scan-friendly turn only: the lexicographic NW-first scan lays the
        letter-turned continuation on ONE handedness.  With s = wc − pc and
        e = exit − wc, decline iff cross(s, e) == −1 for an X̄-vertical
        target and +1 for X̄-horizontal — the transpose is a REFLECTION, so
        the bad handedness flips with the family (ptype transpose symmetry
        makes everything else covariant, same argument as litinski ±x)."""
        bad = -1 if orient.get(nm) == 'X_vertical' else 1
        sx, sy = wc[0] - pc[0], wc[1] - pc[1]
        deg = 0
        for c in tset:
            ex_, ey_ = c[0] - wc[0], c[1] - wc[1]
            if abs(ex_) + abs(ey_) != 1:
                continue
            deg += 1
            if deg > 1:
                return False        # interior bus cell: splice unverified
            # 2026-09-11: the former handedness veto (cross(s, e) == bad)
            # is gone.  The geometry it declined is the corridor leaving the
            # walled cell past the band end on the near line; that case is
            # now constructed by the wall's rule-4 corner end record (see
            # wall_fn in route_and_build), so the veto only cost corridor
            # length.  ``bad``, ``sx``, ``sy`` are kept for the docstring's
            # bookkeeping of which family the transposed case belongs to.
        return True

    for nm in wall_nms:
        pc0 = tuple(patch_at[nm])
        # a user-unstitched seam never hosts a wall (no_stitch wins over
        # every table row — measured: the fig5-six PPM4 wall on q4's
        # unstitched (2,1) seam admits, then the walled build fails and
        # the plain fallback strands q1)
        cells = sorted(c for c in legal_of.get(nm, ())
                       if frozenset((pc0, tuple(c))) not in ns_pairs)
        if nm in direct6_nms:
            # first HOSTABLE legal cell wins (a target with two legal faces
            # may host the wall on either; the hosting law filters per cell)
            cells = [c for c in cells if _hosts_direct6(nm, pc0, tuple(c))]
        if not cells:
            return None, conj_eff, False   # this path cannot host the wall
        wc = cells[0]
        if nm in snake_nms and wc[0] == pc0[0]:
            # snake wall on a HORIZONTAL (N/S) seam — the #4-native
            # horizontal band has no verified construction (measured on
            # the minimal std|conj pair: phase 0 acceptance (0,0,1)
            # joint missing, phase 1 native-lock refusal).  The verified
            # dispatch-snake family is the VERTICAL seam with the
            # corridor flip, mirroring the gateway snake.  Decline so
            # probe = builder.
            return None, conj_eff, False
        aw.append((nm, wc))
        if nm in snake_nms and wc[0] != tuple(patch_at[nm])[0]:
            snake_vertical = True
    if len(direct6_nms) > 1:
        # Multiple native-#6 walls in one joint: historically declined (an
        # early measurement saw acceptance (0,0,1) joint missing on a
        # two-vertical-wall pair) with rotation as the planner's escape.
        # 2026-08-04 re-measurement (user: each target's bus attachment is
        # independent): two- and mixed-wall joints build, verify, run
        # silent and reach FULL graphlike AND hypergraph distance (ZXZ
        # 3-target 2-wall; XZXZ 4-target) — the decline was
        # over-conservative and is now lifted by default.  The env flag
        # restores the old behaviour for bisection.  NOTE: multi-wall
        # blocks carry intrinsic hyperedges — LER decoding needs mwpf, not
        # bare matching (paper-material §2.6).
        import os as _os
        if _os.environ.get('LIGHTSTIM_FORBID_MULTI_WALL'):
            return None, conj_eff, False
    if len({wc for _, wc in aw}) < len(aw):
        # two walls on the SAME corridor cell (both seams of one cell
        # walled) — no verified construction (the two-wall family puts
        # each wall on its own cell); measured: the build falls through
        # and dies on the plain path.  Decline so the probe agrees with
        # the builder and the planner takes another convention (the
        # all-swapped pair goes through the global-conjugate shortcut).
        return None, conj_eff, False
    if snake_vertical and direct6_nms:
        # the snake flip would repaint the corridor under the native-#6
        # walls, moving their seams to row #7 — no verified construction
        return None, conj_eff, False
    if snake_vertical:
        # g = 1 (flipped painting): a STANDARD-colour non-wall target's
        # plain seam becomes the recoloured column (#1 -> #7), verified
        # ONLY as the snake's horizontal-seam family — decline otherwise
        for nm, _P in target:
            if nm in wall_nms or nm in conj_eff:
                continue
            pc = tuple(patch_at[nm])
            adj = [c for c in tset
                   if abs(c[0] - pc[0]) + abs(c[1] - pc[1]) == 1]
            if not all(c[0] == pc[0] for c in adj):
                return None, conj_eff, False
    return aw, conj_eff, snake_vertical


def route_and_build(patches, target, pad=1, per_z=6, max_std=48, cut_budget=4, max_cut=4,
                    keepout=0, seed=0, max_trials=5000, route=None, seam=False, bus=None,
                    conj_names=None, flip_cells=None, no_stitch=None, wall_junction=None,
                    arm_walls=None, probe=False, geom_cache=None,
                    blocked_cells=frozenset()):
    """Fully-automatic route **and** build: no hand-written corridor needed.

    ``route`` (optional): an **explicit corridor** — a list of coarse cells ``[(a, b), …]``.
    When given, NO automatic routing happens: the joint code is built on exactly these
    cells by the deterministic rule constructor and gated by the full oracle.  The cells
    must not overlap any patch; the obstacle **keep-out margin is NOT enforced** for an
    explicit route (you are overriding the router), so check ``collision_report`` if
    obstacles sit next to your corridor.  Omit ``route`` (default) for auto-routing.

    ``conj_names`` (optional): the target patches whose LIVE construction is the
    colour-conjugate registration (see :func:`path_to_corridor`).  ``None`` keeps the
    historical inference (conjugate ⟺ minority basis).  Passing the true set lets a
    conjugate patch attach in the BUS basis — its recoloured seam wall is raised by
    the retype boundary, no rotation needed.

    ``flip_cells`` + ``wall_junction`` (optional, explicit route only): a SNAKE bus.
    ``flip_cells`` recolours those corridor cells into the conjugate gauge (they
    plain-stitch the conjugate targets); ``wall_junction = (cell_a, cell_b)`` names
    the adjacent corridor pair whose seam is REPLACED by a stretched-stabilizer
    colour wall (the K&F spatial-Hadamard interface: uniform pure-letter dominoes,
    flips the colours, keeps the letters).  No seam qubits are added there; the
    wall's kf records are injected as forced checks, so a same-letter joint between
    a colour-swapped and a standard patch closes with zero rotations and NO mixed
    checks anywhere.

    Propose-and-verify with retry: (1) candidate arms leave the X-anchor from **any** face (not just
    its X-faces), so clean below-/side-attach corridors are found; (2) arm-product unions are
    augmented with greedy **Steiner** candidates (:func:`_steiner_trees`) whose arms share a common
    trunk; (3) a candidate whose corridor is **disconnected** (its arms meet only through a target
    patch — two separate buses) is illegal and never tried; if *every* candidate is disconnected the
    result is an honest ``no_path``; (4) each surviving candidate is built by the **deterministic
    rule constructor** (:mod:`.deterministic_checks` — cut-free first, then rule-driven convex-corner
    cuts up to ``max_cut``) and gated by the full oracle.  Candidates are tried
    **fewest-corridor-cells first**, so the smallest *valid* bus wins — a shared straight trunk is
    preferred over fat multi-arm unions.  ``per_z`` / ``max_std`` / ``keepout`` shape the candidate
    pool; ``cut_budget`` / ``seed`` / ``max_trials`` are accepted for backward compatibility and
    ignored (there is no randomized search any more).  This is the routine the demo notebook calls
    instead of pasting cells.

    Returns a :class:`SubsetRoute` with ``status == "ok"`` and ``.layout`` / ``.tree`` (route cells) /
    ``.how`` (``"standard"`` | ``"corner-cut"``) / ``.cut`` set, or a failure ``SubsetRoute`` with
    status ``target_obstacle_conflict`` / ``no_path`` / ``no_verified_route``.
    """
    # corridor search lives in circls.compiler.routing (paper S3);
    # imported lazily so core never imports compiler at load time.
    # Resolved per call, so monkeypatching the routing module (the
    # optgap E3 audit does) is honoured here.
    from circls.compiler.routing import (_ARM_POOL_CAP,
        _EMV_GROUP_CAP, _bounded_product, _candidate_arms,
        _cells_connected, _corridor_graph, _legal_greedy_trees,
        _single_width, _steiner_trees, emv_corridor_candidates)
    patch_at, orient, d = _specs_to_cells(patches, target, seam=seam)
    tnames = [nm for nm, _ in target]
    # no_stitch: adjacencies that stay UNSTITCHED — the two boundaries sit next
    # to each other without merging (no seam qubits, no seam-orientation rule;
    # each side keeps its own free boundary).  Entries are (patch, cell) or
    # (patch, patch); normalize to cell pairs here.
    ns_pairs = frozenset(
        frozenset((tuple(patch_at[a]) if isinstance(a, str) else tuple(a),
                   tuple(patch_at[b]) if isinstance(b, str) else tuple(b)))
        for a, b in (no_stitch or ()))
    onames = [nm for nm in patch_at if nm not in tnames]

    def _auto_stitch(tree_cells, wall_pairs=frozenset()):
        """Parallel-law stitching policy: a target|cell (or target|target)
        adjacency is STITCHED iff the seam runs parallel to that target's
        measured logical; an illegal adjacency stays UNSTITCHED — the two
        boundaries simply sit next to each other, separated by the empty
        seam line (a corridor cell may pass BY one target to reach
        another).  User ``no_stitch`` pairs always stay unstitched; a
        walled junction counts as an attachment.  Returns
        ``(skip_pairs, violation)`` — ``violation`` is set when a target
        ends up with NO attachment at all."""
        skip = set(ns_pairs)
        attached = {nm: 0 for nm in tnames}
        # two-bit TABLE legality of a PLAIN (non-wall) stitch: row #1
        # (letter == bus, standard colours) or row #7 (letter != bus,
        # conj-registered — the retype-boundary M-column family).  The
        # wall rows #4/#6 cannot plain-stitch: without a dispatched wall
        # the seam stays open.  Mirrors the rule constructor exactly, so
        # the step-1 probe never green-lights a seam step 2 cannot build.
        n_x_s = sum(1 for _, P in target if P == "X")
        bus_s = bus if bus is not None else (
            "X" if n_x_s >= len(target) - n_x_s else "Z")
        conj_s = conj_names if conj_names is not None else \
            frozenset(nm for nm, P in target if P != bus_s)

        def _row_plain_ok(nm, P):
            return (P != bus_s) == (nm in conj_s)

        for nm, P in target:
            pa = tuple(patch_at[nm])
            legal_cells = []
            for tc in map(tuple, tree_cells):
                da, db = tc[0] - pa[0], tc[1] - pa[1]
                if abs(da) + abs(db) != 1:
                    continue
                pair = frozenset((pa, tc))
                if pair in wall_pairs:
                    attached[nm] += 1          # the wall IS the attachment
                    continue
                if pair in skip:
                    continue                   # user unstitch wins
                seam_ew = (da != 0)
                want = ('X_horizontal' if (P == 'Z') == seam_ew
                        else 'X_vertical')
                if orient[nm] != want or not _row_plain_ok(nm, P):
                    skip.add(pair)
                else:
                    legal_cells.append(tc)
            # SINGLE ATTACHMENT (user rule 2026-07-31): when the bus wraps a
            # target on several sides, the patch attaches through EXACTLY
            # ONE seam — any legal side works (measured: the flanked-patch
            # double stitch is what wrecked the corner geometry), the choice
            # is free; deterministic first-by-order here.  Extra legal
            # adjacencies stay unstitched (the corridor passes by).
            if attached[nm] == 0 and legal_cells:
                keep = sorted(legal_cells)[0]
                attached[nm] += 1
                for tc in legal_cells:
                    if tc != keep:
                        skip.add(frozenset((pa, tc)))
            else:
                for tc in legal_cells:
                    skip.add(frozenset((pa, tc)))
        tl = list(target)
        for i1, (n1, P1) in enumerate(tl):
            c1 = tuple(patch_at[n1])
            for n2, P2 in tl[i1 + 1:]:
                c2 = tuple(patch_at[n2])
                da, db = c2[0] - c1[0], c2[1] - c1[1]
                if abs(da) + abs(db) != 1:
                    continue
                pair = frozenset((c1, c2))
                if pair in wall_pairs:
                    attached[n1] += 1
                    attached[n2] += 1
                    continue
                if pair in skip:
                    continue
                seam_ew = (da != 0)
                legal = all(orient[nm] == ('X_horizontal'
                                           if (P == 'Z') == seam_ew
                                           else 'X_vertical')
                            and _row_plain_ok(nm, P)
                            for nm, P in ((n1, P1), (n2, P2)))
                if legal:
                    attached[n1] += 1
                    attached[n2] += 1
                else:
                    skip.add(pair)
        for nm, P in target:
            if attached[nm] == 0:
                return skip, (
                    f"{nm}: no parallel-law-legal attach seam to this "
                    f"corridor — measuring {P} needs a seam PARALLEL to the "
                    f"measured logical ({'vertical (E/W)' if (P == 'X') == (orient[nm] == 'X_vertical') else 'horizontal (N/S)'} "
                    f"for orientation {orient[nm]!r}); every adjacency is "
                    f"perpendicular, user-unstitched, or absent")
        # CONNECTIVITY: per-target degree >= 1 is not enough — two adjacent
        # targets can legally stitch to EACH OTHER while their corridor
        # seams are all skipped, forming an island the bus never reaches
        # (fig5 q6|q4 under all-vertical declarations).  The stitched merge
        # graph (corridor cells + target cells; edges = corridor internal
        # adjacency + every stitched/walled pair) must put ALL targets in
        # the component containing the corridor.
        cells_t = {tuple(patch_at[nm]): nm for nm in tnames}
        nodes = set(map(tuple, tree_cells)) | set(cells_t)
        adj = {n: set() for n in nodes}
        tset_c = set(map(tuple, tree_cells))
        for a in tset_c:
            for b in tset_c:
                if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1:
                    adj[a].add(b)
                    adj[b].add(a)
        def _link(a, b):
            pair = frozenset((a, b))
            if pair in wall_pairs or pair not in skip:
                adj[a].add(b)
                adj[b].add(a)
        for nm in tnames:
            pa = tuple(patch_at[nm])
            for tc in tset_c:
                if abs(tc[0] - pa[0]) + abs(tc[1] - pa[1]) == 1:
                    _link(pa, tc)
            for nm2 in tnames:
                pb = tuple(patch_at[nm2])
                if nm2 != nm and abs(pb[0] - pa[0]) + abs(pb[1] - pa[1]) == 1:
                    _link(pa, pb)
        seed = next(iter(tset_c), tuple(patch_at[tnames[0]]))
        seen_c, frontier = {seed}, [seed]
        while frontier:
            n = frontier.pop()
            for m in adj[n]:
                if m not in seen_c:
                    seen_c.add(m)
                    frontier.append(m)
        stranded = [nm for nm in tnames if tuple(patch_at[nm]) not in seen_c]
        if stranded:
            return skip, (
                f"{stranded}: stitched to a neighbouring target but the "
                f"island never reaches the corridor — every corridor seam "
                f"of the island is perpendicular, user-unstitched, or "
                f"absent (the joint product cannot flow onto the bus)")
        return skip, None

    placed_all = {nm: cell(*ab, d, seam) for nm, ab in patch_at.items()}
    obstacle_fp0 = set().union(*[placed_all[nm] for nm in onames]) if onames else set()
    base0 = dict(placed=placed_all, target=list(target), obstacles=onames, obstacle_fp=obstacle_fp0,
                 corridor=set())
    for tn in tnames:
        bad = [on for on in onames if _cheb(patch_at[tn], patch_at[on]) <= keepout]
        if bad:
            return SubsetRoute(status="target_obstacle_conflict", root=tnames[0],
                               message=(f"target {tn} is within keepout={keepout} of obstacle(s) "
                                        f"{bad}: their boundary ancillas would collide."), **base0)
    # route from a bus-basis anchor; ``bus`` forces a PLANNED basis, else the majority
    # target basis (tie keeps the historical X anchor)
    n_x = sum(1 for _, P in target if P == "X")
    n_z = sum(1 for _, P in target if P == "Z")
    bus_basis = bus if bus is not None else ("X" if n_x >= n_z else "Z")
    root0 = next((nm for nm, P in target if P == bus_basis), tnames[0])
    if wall_junction is not None and route is None:
        raise ValueError("wall_junction needs an explicit route")

    if route is not None:                  # explicit corridor: build on EXACTLY these cells
        tree = {tuple(c) for c in route}
        occupied_cells = set(patch_at.values())
        bad = sorted(tree & occupied_cells)
        if bad:
            raise ValueError(f"explicit route cells {bad} overlap patch cells; the corridor "
                             f"may only use empty coarse cells")
        # stitching decided by the parallel-law policy AFTER the wall
        # junctions are known (a walled seam counts as an attachment)
        # ---- corridor walls: mid-bus junctions (experimental) and ARM walls
        # (a stretched wall ON a target's attach seam — the production device
        # for rows #4/#6 of the seam table; the closed form is the VERIFIED
        # patch|patch spec, the corridor cell simply takes the far-patch role)
        # post-path seam-table dispatch (same rule as the auto branch): a
        # GIVEN route is still a routed path — when the caller assigned no
        # walls, classify the attach seams and assign the minority-type
        # targets' walls here
        if probe:
            # STEP-1 PROBE (stage B): geometric feasibility + table
            # classification only - route legality, parallel-law stitching
            # and the per-seam wall count, NO construction, NO oracle.
            wall_pairs2, aw_n = frozenset(), 0
            if seam and TABLE_WALL_DISPATCH and not arm_walls \
                    and wall_junction is None:
                dp = _table_wall_dispatch(patch_at, target, orient, tree,
                                          conj_names, bus,
                                          ns_pairs=ns_pairs)
                if dp is not None and dp[0] is not None:
                    aw_n = len(dp[0])
                    wall_pairs2 = frozenset(
                        frozenset((tuple(patch_at[nm]), tuple(wc)))
                        for nm, wc in dp[0])
            if arm_walls:
                aw_n += len(arm_walls)
                wall_pairs2 |= frozenset(
                    frozenset((tuple(patch_at[nm]), tuple(wc)))
                    for nm, wc in arm_walls)
            if seam:
                _, viol = _auto_stitch(tree, wall_pairs=wall_pairs2)
                if viol:
                    return SubsetRoute(status="no_verified_route", root=root0,
                                       tried=0, attempted=tree,
                                       message=f"probe: {viol}", **base0)
            return SubsetRoute(status="ok", message="probe", tree=tree,
                               attempted=tree, tried=0,
                               n_walls=aw_n, **base0)

        if seam and TABLE_WALL_DISPATCH and not arm_walls \
                and wall_junction is None:
            disp = _table_wall_dispatch(patch_at, target, orient, tree,
                                        conj_names, bus,
                                        ns_pairs=ns_pairs)
            if disp is not None and disp[0] is not None:
                # the dispatched wall IS this route's construction — its
                # outcome is FINAL (design decision 2026-07-31: a failed step-2
                # construction is a hard error, never a fallback; the old
                # swallow-and-fall-to-plain masked the real error as a
                # misleading 'no parallel-law-legal seam')
                aw2, conj_eff2, vertical2 = disp
                return route_and_build(
                    patches, target, pad=pad, per_z=per_z,
                    max_std=max_std, cut_budget=cut_budget,
                    max_cut=max_cut, keepout=keepout, seed=seed,
                    max_trials=max_trials, route=sorted(tree),
                    seam=True, bus=bus, conj_names=conj_eff2,
                    flip_cells=(sorted(tree) if vertical2
                                else flip_cells),
                    no_stitch=no_stitch, arm_walls=aw2)

        wall_descr = []          # (c1, c2, kind, patch_name)
        if wall_junction is not None:
            c1, c2 = tuple(wall_junction[0]), tuple(wall_junction[1])
            if c1 not in tree or c2 not in tree:
                raise ValueError(f"wall_junction {(c1, c2)} must be route cells")
            wall_descr.append((c1, c2, 'mid', None))
        for nm, wcell in (arm_walls or ()):
            pc, wc2 = tuple(patch_at[nm]), tuple(wcell)
            if wc2 not in tree:
                raise ValueError(f"arm wall cell {wc2} is not a route cell")
            if conj_names is None:
                raise ValueError("arm_walls needs an explicit conj_names")
            wall_descr.append((pc, wc2, 'arm', nm))
        walls = []
        retire_lobes = set()
        extra_connect = set()
        for c1, c2, kind, wnm in wall_descr:
            da, db = c2[0] - c1[0], c2[1] - c1[1]
            if abs(da) + abs(db) != 1:
                raise ValueError(f"wall cells {(c1, c2)} are not adjacent")
            if da == 0:
                near_c = c1 if db > 0 else c2
                axis = 'horizontal'
            else:
                near_c = c1 if da > 0 else c2
                axis = 'vertical'
            ox, oy = origin_of(*near_c, d, seam=seam)
            a0, n0 = (ox, oy) if axis == 'horizontal' else (oy, ox)
            near_ln = n0 + 2 * (d - 1)
            lo, seam_ln, hi, far_ln = (near_ln + 1, near_ln + 2, near_ln + 3,
                                       near_ln + 4)
            P = ((lambda a, m: (a, m)) if axis == 'horizontal'
                 else (lambda a, m: (m, a)))
            bus_eff2 = None
            if kind == 'arm':
                bus_eff2 = bus
                if bus_eff2 is None:
                    n_x2 = sum(1 for _, PP in target if PP == "X")
                    bus_eff2 = "X" if n_x2 >= len(target) - n_x2 else "Z"
                if (dict(target)[wnm] == bus_eff2
                        and wnm not in (conj_names or frozenset())):
                    raise BentLayoutError(
                        f"arm wall {wnm}|{c2}: the target is same-type with "
                        f"the corridor (rule-table row 1 = plain merge, no "
                        f"wall is due); register it in the recolour "
                        f"convention (conj_names) to raise a wall")
            end_row = None
            if kind == 'arm':
                # the stretched end lobe goes at the end where it is NOT next
                # to one of the attaching patch's own weight-2 lobes (the
                # boundary alternation rule); probe the patch's REGISTERED
                # construction for a lobe foot on the facing corner data
                s_w = next(s for s in patches if s.name == wnm)
                _fo = {"X_horizontal": "X_vertical",
                       "X_vertical": "X_horizontal"}
                reg_o = (_fo[s_w.orientation] if wnm in conj_names
                         else s_w.orientation)
                nat = place_patch(SimpleNamespace(
                    origin=s_w.origin, distance=d, orientation=reg_o))["checks"]
                pside = near_ln if tuple(patch_at[wnm]) == near_c else far_ln
                band_lines = {lo, hi}
                ax_i = 1 if axis == 'horizontal' else 0
                blocked = set()
                for c in nat:
                    if len(c["pauli"]) != 2:
                        continue
                    if tuple(c["syn"])[ax_i] in band_lines:
                        continue        # the patch's own FACING lobes — they
                                        # are retired by the wall, not blockers
                    for q in c["pauli"]:
                        if q == P(a0, pside):
                            blocked.add('low')
                        if q == P(a0 + 2 * (d - 1), pside):
                            blocked.add('high')
                if 'low' not in blocked:
                    end_row = a0
                elif 'high' not in blocked:
                    end_row = a0 + 2 * (d - 1)
                else:
                    raise BentLayoutError(
                        f"arm wall {wnm}|{c2}: both band ends sit next to the "
                        f"patch's own weight-2 lobes — no legal end for the "
                        f"stretched lobe")
            walls.append(dict(axis=axis, ox=ox, oy=oy, a0=a0,
                              near_ln=near_ln, lo=lo, seam_ln=seam_ln, hi=hi,
                              far_ln=far_ln, P=P, kind=kind,
                              end_row=end_row, bus_letter=bus_eff2))
            # the true facing-lobe sites (a tile there would couple two
            # band feet on the same facing line) are the wall's territory
            for a in range(a0 + 1, a0 + 2 * (d - 1), 2):
                retire_lobes.add(P(a, lo))
                retire_lobes.add(P(a, hi))
            extra_connect |= _seam_qubits({c1, c2}, d)
        wall_pairs = frozenset(frozenset((tuple(c1), tuple(c2)))
                               for c1, c2, _, _n in wall_descr)
        if seam:
            stitch_skip, viol = _auto_stitch(tree, wall_pairs=wall_pairs)
            if viol:
                raise BentLayoutError(viol)
        else:
            stitch_skip = ns_pairs
        data, retype = path_to_corridor(tree, placed_all, target, d, seam=seam,
                                        patch_at=patch_at, bus=bus,
                                        conj_names=conj_names,
                                        flip_cells=flip_cells,
                                        skip_seam_pairs=([(c1, c2)
                                                          for c1, c2, _, _n
                                                          in wall_descr]
                                                         + [tuple(p) for p
                                                            in stitch_skip])
                                        or None)
        wall_fn = None
        retyped_bus = frozenset()
        if walls:
            from .joint_merge import _wall_checks as _wc
            from lightstim.qec_code.surface_code.rotated.litinski_layouts import ptype as _pt
            _retype = frozenset(retype)
            _data_set = frozenset(tuple(q) for q in data)

            def wall_fn(phase, _wc=_wc, d=d, walls=walls, _retype=_retype,
                        _data_set=_data_set):
                out = []
                for w in walls:
                    P = w['P']
                    if w['kind'] == 'arm':
                        # the user's arrangement rules (K&F Fig 4 a/b):
                        # each stretched weight-4's side pair carries the
                        # OPPOSITE letter of the cell stabilizer right next
                        # to it on that side; the stretched weight-2 end
                        # lobe carries the flip of its neighbouring stretched
                        # record, at the end away from the patch's own lobes
                        nl, fl = w['near_ln'], w['far_ln']
                        nf = 1 if P(w['a0'], nl) in _retype else 0
                        ff = 1 if P(w['a0'], fl) in _retype else 0

                        def _cl(pos_a, line_m, flip):
                            c = P(pos_a, line_m)
                            L = _pt(c[0], c[1], phase)
                            return _FLIP[L] if flip else L

                        recs = []
                        for k in range(d - 1):
                            y = w['a0'] + 2 * k
                            Ll = _FLIP[_cl(y + 1, nl - 1, nf)]
                            Lr = _FLIP[_cl(y + 1, fl + 1, ff)]
                            feet = {P(y, nl): Ll, P(y + 2, nl): Ll,
                                    P(y, fl): Lr, P(y + 2, fl): Lr}
                            A, B = ((P(y + 1, w['hi']), P(y + 1, w['lo']))
                                    if Lr == 'Z'
                                    else (P(y + 1, w['lo']), P(y + 1, w['hi'])))
                            recs.append(
                                {'syn': B, 'type': 'M', 'pauli': feet,
                                 'corners': sorted(feet),
                                 'kf': {'flag': A,
                                        'shared': P(y + 1, w['seam_ln']),
                                        'orient': ('+' if A == P(y + 1, w['hi'])
                                                   else '-')}})
                        e = w['end_row']
                        adj = recs[0] if e == w['a0'] else recs[-1]
                        Ll_e = _FLIP[adj['pauli'][P(e, nl)]]
                        Lr_e = _FLIP[adj['pauli'][P(e, fl)]]
                        erow = e - 1 if e == w['a0'] else e + 1
                        feet = {P(e, nl): Ll_e, P(e, fl): Lr_e}
                        # paper rule 4 (concave inward corner) at the band
                        # end: when the corridor CONTINUES past this end on
                        # the near line (the next near-line data qubit beyond
                        # the band belongs to the merged region), the band end
                        # is a concave corner of the ancilla path, two gaps
                        # from its neighbouring boundary stabilizers, and the
                        # end record is the weight-3 corner stabilizer that
                        # takes that qubit in with the near-side letter.  The
                        # bare weight-2 lobe (the straight two-patch stretched
                        # seam of Fig. d) anticommutes with the corridor's own
                        # plaquette across the band end and leaves the joint
                        # product short by one Z (toffoli_n3 PPM 1, audited
                        # 2026-09-11: paper-derived layout differs from the
                        # code's by exactly this record; with it the west
                        # corridor verifies, p=0 silent, distance 3).
                        # The patch may sit on either line of the band, so
                        # look on both: the corridor side is whichever line
                        # still carries a data qubit past the band end (the
                        # patch side never does: the patch ends with the band
                        # and an unstitched neighbour has no seam qubits).
                        e_beyond = e - 2 if e == w['a0'] else e + 2
                        for line_m, letter in ((nl, Ll_e), (fl, Lr_e)):
                            beyond = P(e_beyond, line_m)
                            if beyond in _data_set:
                                feet[beyond] = letter
                        A, B = ((P(erow, w['hi']), P(erow, w['lo']))
                                if Lr_e == 'Z'
                                else (P(erow, w['lo']), P(erow, w['hi'])))
                        recs.append(
                            {'syn': B, 'type': 'M', 'pauli': feet,
                             'corners': sorted(feet),
                             'kf': {'flag': A, 'shared': P(erow, w['seam_ln']),
                                    'orient': ('+' if A == P(erow, w['hi'])
                                               else '-')}})
                        out.extend(recs)
                        continue
                    rep_near = P(w['a0'], w['near_ln'])
                    rep_far = P(w['a0'], w['far_ln'])
                    phi_n = phase ^ (1 if rep_near in _retype else 0)
                    phi_f = phase ^ (1 if rep_far in _retype else 0)
                    recs = _wc(d, phi_n, phi_f, w['ox'], w['oy'], w['axis'])
                    part = [{'syn': tuple(B), 'type': 'M', 'pauli': dict(p),
                             'corners': sorted(p), 'kf': dict(kf)}
                            for B, p, kf in recs]
                    if True:
                        # mid-bus junction (experimental): both band ends are
                        # free edges — mirror the closed form's end lobe
                        ends = {w['a0'] - 1, w['a0'] - 1 + 2 * d}
                        ax_i = 0 if w['axis'] == 'horizontal' else 1
                        used = next(a for a in ends
                                    if any(r['syn'][ax_i] == a for r in part))
                        m_e = next(a for a in ends if a != used)
                        sgn = 1 if m_e == w['a0'] - 1 else -1
                        adj = next(r for r in part
                                   if abs(r['syn'][ax_i] - (m_e + 2 * sgn)) <= 1)
                        letter = _FLIP[next(iter(set(adj['pauli'].values())))]
                        feet = {P(m_e + sgn, w['near_ln']): letter,
                                P(m_e + sgn, w['far_ln']): letter}
                        A, B = ((P(m_e, w['hi']), P(m_e, w['lo']))
                                if letter == 'Z'
                                else (P(m_e, w['lo']), P(m_e, w['hi'])))
                        part.append({'syn': B, 'type': 'M', 'pauli': feet,
                                     'corners': sorted(feet),
                                     'kf': {'flag': A,
                                            'shared': P(m_e - sgn, w['seam_ln']),
                                            'orient': '+' if A == P(m_e, w['hi'])
                                            else '-'}})
                    out.extend(part)
                return out
        if flip_cells:
            fq = set()
            for c in flip_cells:
                fq |= cell(*tuple(c), d, seam)
            if seam:
                fents = {tuple(c) for c in flip_cells} | \
                    {tuple(patch_at[nm]) for nm in tnames
                     if conj_names and nm in conj_names}
                fq |= _seam_qubits(fents, d)
            patchq = set().union(*[placed_all[nm] for nm in tnames])
            retyped_bus = frozenset((fq & set(data)) - patchq)
        # idle neighbours' USED ancilla sites — computed HERE, at the actual
        # construction, so the step-1 probe never touches place_patch (iron
        # rule: path finding is construction-free; this was also pure waste
        # on every probe — the probe branch cannot reach this point)
        forbidden = _obstacle_ancillas(patches, tnames)
        layout = _assemble_region(placed_all, target, orient, data, retype, d, seed,
                                  max_trials, max_cut=max_cut, forbidden=forbidden,
                                  native_lock=True, bus=bus, conj_names=conj_names,
                                  extra_forced_fn=wall_fn,
                                  extra_connect=frozenset(extra_connect),
                                  retyped_bus=retyped_bus,
                                  retire_lobes=frozenset(retire_lobes))
        if layout is not None and all(layout.verify().values()):
            cutq = tuple(sorted(set(data) - set(layout.data)))
            how = "corner-cut" if cutq else "standard"
            msg = f"verified subset joint ({how}, rule-based, EXPLICIT route)"
            return SubsetRoute(status="ok", message=msg,
                               layout=layout, root=root0, tree=tree, attempted=tree,
                               data=sorted(layout.data), tried=1, how=how, cut=cutq,
                               n_walls=len(wall_descr),
                               **base0)
        return SubsetRoute(status="no_verified_route", root=root0, tried=1, attempted=tree,
                           message=("the EXPLICIT route does not pass the rule-based "
                                    "construction; the physics layer cannot host this "
                                    "corridor"), **base0)

    # INTRA-STEP GEOMETRY CACHE (step-1 accelerator, geometry only — no
    # stabilizer content lives here, so the iron rule is untouched).  The
    # planner probes ~2^k x 2 candidates per step that differ ONLY in trial
    # orientations / bus / conj: patch positions, obstacles, the corridor
    # graph and the arm/Steiner pools are identical across them, so
    # ('pool', k1) memoizes graph + orientation-independent pools.  A pure
    # memo: a hit MUST be bit-identical to recomputation (the key covers
    # every input of the cached computation).  The EMV enumeration varies
    # per candidate (parallel-law faces follow the trial orientations) and
    # is consumed LAZILY below instead of being cached.
    k1 = None
    if geom_cache is not None:
        k1 = (frozenset((nm, tuple(c)) for nm, c in patch_at.items()),
              tuple((nm, P) for nm, P in target), root0, d, pad, keepout,
              seam, per_z)
    pool = geom_cache.get(('pool', k1)) if geom_cache is not None else None
    if pool is None:
        G, corridor, placed, occupied, obstacle_fp, onames = _corridor_graph(
            patch_at, target, d, pad, keepout=keepout, seam=seam)
        root = root0
        zs = [nm for nm in tnames if nm != root]
        cand = {}
        arm_fail = None
        for z in zs:
            arms = _candidate_arms(G, patch_at, corridor, root, NEIGH, z, per_z)   # any face, not just X
            if not arms:
                arm_fail = z
                break
            cand[z] = arms
        if arm_fail is None:
            total = 1
            for z in zs:
                total *= len(cand[z])
                if total > _ARM_POOL_CAP:
                    break
            if not zs:
                combos = [()]
            elif total <= _ARM_POOL_CAP:
                combos = list(itertools.product(*[cand[z] for z in zs]))
            else:
                combos = _bounded_product([cand[z] for z in zs],
                                          _ARM_POOL_CAP)
            raw_base = [set().union(*[set(p) for p in c]) if c else set() for c in combos]
            raw_base += _steiner_trees(G, patch_at, corridor, root, zs)   # shared-trunk candidates
        else:
            raw_base = []
        pool = (G, corridor, placed, occupied, obstacle_fp, onames,
                raw_base, arm_fail)
        if geom_cache is not None:
            geom_cache[('pool', k1)] = pool
    G, corridor, placed, occupied, obstacle_fp, onames, raw_base, arm_fail = pool
    if blocked_cells:
        # parallel batching (2026-08-05): batch-mates' corridors are
        # off-limits to this route.  Filtering AFTER the pool keeps the
        # orientation-agnostic geometry cache blocked-agnostic.
        _bc = {tuple(c) for c in blocked_cells}
        corridor = {c for c in corridor if c not in _bc}
        G = G.subgraph([v for v in G.nodes if v not in _bc]).copy()
        raw_base = [t for t in raw_base
                    if not ({tuple(c) for c in t} & _bc)]
    base = dict(placed=placed, target=list(target), obstacles=onames, obstacle_fp=obstacle_fp,
                corridor=corridor)
    root = root0
    zs = [nm for nm in tnames if nm != root]

    if not seam:
        # LEGACY abutting pipeline (seam=False): no EMV (it is seam-only)
        # and no table admission — materialize the arm/Steiner pool and
        # enumerate by construction as before.  The two-step no-retry
        # ruling (2026-07-31) governs the seam/corridor pipeline.
        if zs and not raw_base:
            return SubsetRoute(status="no_path", root=root,
                               message=(f"path search found no obstacle-free "
                                        f"corridor {root} -> {arm_fail}"),
                               **base)
        uniq, trees, dropped = set(), [], 0
        for t in raw_base:
            key = frozenset(t)
            if key in uniq:
                continue
            uniq.add(key)
            if zs and not _cells_connected(t):     # a corridor split by a target patch is illegal
                dropped += 1
                continue
            trees.append(t)
        if zs and not trees:
            return SubsetRoute(status="no_path", root=root,
                               message=(f"every candidate corridor is disconnected (the {dropped} "
                                        f"arm-unions meet only through a target patch); no single "
                                        f"connected ancilla bus exists for this placement"), **base)
        trees.sort(key=len)
        trees = trees[:max_std]
        if probe:
            return SubsetRoute(status="ok", message="probe", root=root,
                               tree=trees[0], attempted=trees[0], tried=0,
                               n_walls=0, **base)
        viol = None
        for tree in trees:
            try:
                sub = route_and_build(patches, target, pad=pad, per_z=per_z,
                                      max_std=max_std, cut_budget=cut_budget,
                                      max_cut=max_cut, keepout=keepout,
                                      seed=seed, max_trials=max_trials,
                                      route=sorted(tree), seam=False, bus=bus,
                                      conj_names=conj_names,
                                      no_stitch=no_stitch)
            except (BentLayoutError, ValueError) as e:
                viol = viol or str(e)
                continue
            if sub is not None and sub.ok:
                return sub
            if sub is not None and sub.message:
                viol = viol or sub.message
        if viol:
            raise BentLayoutError(viol)
        return SubsetRoute(status="no_verified_route", root=root,
                           tried=len(trees), attempted=trees[0] if trees else set(),
                           message=("no candidate corridor passes the "
                                    "rule-based construction; the physics "
                                    "layer cannot host this geometry"), **base)

    # SEAM pipeline — LAZY exact enumeration.  PRIMARY source: exact EMV
    # group-Steiner over the PARALLEL-LAW legal attach cells
    # (corridor-routing-model.md); the arm/Steiner pool is extra diversity.
    # No extra_cost is passed, so the EMV cost IS the cell count and the
    # Lawler heap yields in nondecreasing length — the merged stream below
    # reproduces the old materialized pipeline's order EXACTLY (stable
    # len-sort of [EMV yields, pool] = ordered merge with EMV first on
    # ties), while consuming the generator only as far as the selection
    # actually looks: the old list() forced the full 24-candidate Lawler
    # branching (~hundreds of DP solves per probe) even when the FIRST
    # candidate was admitted (profiled: 99.5% of the 30-patch build).
    groups_, missing_ = _legal_attach_groups(patch_at, target, orient,
                                             corridor)
    # above the EMV cap the exact enumerator is silent — supply the
    # legal-faces greedy trees instead (orientation-dependent, so computed
    # here per call, never in the orientation-agnostic geom pool cache)
    legal_greedy = ([] if missing_ or len(groups_) <= _EMV_GROUP_CAP
                    else _legal_greedy_trees(G, groups_))
    stats = {'raw': 0, 'dropped': 0, 'passed': 0}

    def _candidates():
        emv_it = (emv_corridor_candidates(G, groups_) if not missing_
                  else iter(()))
        pool_sorted = sorted(raw_base + legal_greedy, key=len)  # stable: original order within a length
        pi, pn = 0, len(pool_sorted)
        uniq = set()
        nxt = next(emv_it, None)
        while True:
            if nxt is not None and (pi >= pn
                                    or len(nxt) <= len(pool_sorted[pi])):
                t = nxt
                nxt = next(emv_it, None)
            elif pi < pn:
                t = pool_sorted[pi]
                pi += 1
            else:
                return
            stats['raw'] += 1
            key = frozenset(t)
            if key in uniq:
                continue
            uniq.add(key)
            if zs and not _cells_connected(t):     # a corridor split by a target patch is illegal
                stats['dropped'] += 1
                continue
            if t and not _single_width(G, t):
                # single-width rule: the corridor's induced cell subgraph must be
                # a tree — 2x2 blobs / loops are unconstructible
                stats['dropped'] += 1
                continue
            stats['passed'] += 1
            yield t
            if stats['passed'] >= max_std:
                return

    # STEP-1 SELECTION for BOTH modes: the first candidate that passes the
    # geometric legality checks (table dispatch verdict + parallel-law
    # stitch + connectivity) is THE corridor — probe reports it, and the
    # REAL path builds EXACTLY it, once (design decision 2026-07-31: path
    # finding never constructs; a failed construction is a hard error and
    # NEVER retries another corridor).  Sharing this literal loop is what
    # keeps step-1 verdicts and step-2 buildability aligned.
    last_viol = None
    selected = None
    sel_walls = 0
    first_passed = None
    for tree in _candidates():
        if first_passed is None:
            first_passed = tree
        wall_pairs2, aw_n = frozenset(), 0
        if TABLE_WALL_DISPATCH:
            dp = _table_wall_dispatch(patch_at, target, orient, tree,
                                      conj_names, bus,
                                      ns_pairs=ns_pairs)
            if dp is not None and dp[0] is not None:
                aw_n = len(dp[0])
                wall_pairs2 = frozenset(
                    frozenset((tuple(patch_at[nm]), tuple(wc)))
                    for nm, wc in dp[0])
            # dp[0] is None: type split but no wall on this path - the
            # builder falls through to the PLAIN attempt, so the tree is
            # judged by the plain parallel-law check below
        _, viol = _auto_stitch(tree, wall_pairs=wall_pairs2)
        import os as _os
        if _os.environ.get('LIGHTSTIM_DEBUG_PROBE'):
            import sys as _sys
            print(f"[probe] tree={sorted(tree)} aw_n={aw_n} "
                  f"wall_pairs={sorted(map(sorted, wall_pairs2))} "
                  f"viol={viol}", file=_sys.stderr)
        if viol:
            last_viol = viol
            continue
        selected, sel_walls = tree, aw_n
        break
    if selected is None:
        # stream exhausted — classify the failure exactly as the old
        # materialized pipeline did
        if zs and not stats['raw']:
            return SubsetRoute(status="no_path", root=root,
                               message=(f"path search found no obstacle-free "
                                        f"corridor {root} -> {arm_fail}"),
                               **base)
        if zs and not stats['passed']:
            return SubsetRoute(status="no_path", root=root,
                               message=(f"every candidate corridor is disconnected (the "
                                        f"{stats['dropped']} arm-unions meet only through a "
                                        f"target patch); no single connected ancilla bus "
                                        f"exists for this placement"), **base)
        if probe:
            return SubsetRoute(status="no_verified_route", root=root, tried=0,
                               attempted=(first_passed if first_passed
                                          is not None else set()),
                               message=f"probe: {last_viol or 'no legal candidate'}",
                               **base)
        # REAL mode: no geometrically legal corridor is a HARD error (the
        # planner filters probe verdicts; reaching here unplanned means the
        # declared inputs cannot serve the step — fail loudly, same
        # exception taxonomy as the explicit branch's stitch violation)
        raise BentLayoutError(last_viol or "no legal candidate corridor")
    if probe:
        return SubsetRoute(status="ok", message="probe", root=root,
                           tree=selected, attempted=selected, tried=0,
                           n_walls=sel_walls, **base)

    # STEP-2: ONE delegation of the selected corridor to the explicit-route
    # pipeline (one shared implementation: seam-table wall dispatch, stitch
    # legality + connectivity, rule construction, corner cuts).  Exceptions
    # propagate and a non-ok result is final — no next-tree retry, no
    # wall-to-plain fallback (design decision 2026-07-31).
    return route_and_build(patches, target, pad=pad, per_z=per_z,
                           max_std=max_std, cut_budget=cut_budget,
                           max_cut=max_cut, keepout=keepout, seed=seed,
                           max_trials=max_trials, route=sorted(selected),
                           seam=seam, bus=bus, conj_names=conj_names,
                           flip_cells=flip_cells, no_stitch=no_stitch)


# -----------------------------------------------------------------------------
# failure taxonomy
# -----------------------------------------------------------------------------

def complete_code(patches, target, tree_cells, phase=None):
    """Build a **complete, maximal commuting** stabilizer code on the routed region: force the
    weight-4 plaquettes, then greedily add **every** commuting-and-independent boundary plaquette
    (weight-2 and the weight-3 corner plaquettes).  The result is a *valid* code — ``k = n - rank``
    logical qubits, all stabilizers commuting — not just the forced bulk.  Use it to judge measurability
    honestly: the joint is measurable **iff it lies in the plaquette span** (``joint_in_span``), which
    is a property of *all* plaquettes, independent of which complete code you pick.

    Returns a dict: ``stabilizers`` (the complete code, with ``corners``), ``k``, ``weights``
    (count by weight), ``commute``, ``no_weight1``, ``no_twist``, ``joint_in_span``,
    ``singles_in_span``, ``subproducts_in_span``, ``readout_chain`` (the stabilizers whose product is
    the joint, when measurable), ``data``, ``reps``, ``phase``.

    ``patches`` is a list of :class:`PatchSpec` (see :func:`route_and_build`); ``tree_cells`` is the
    explicit corridor to complete, a list of coarse cells ``[(a, b), ...]``.  ``phase`` (default
    ``None`` = auto) **forces** a specific parity phase (``0``/``1``) instead of auto-picking the
    joint-in-span one — use it to probe whether a given local checkerboard coloring changes
    measurability (it does not: joint-in-span is a property of the full plaquette span, checked here
    for whichever phase(s) are considered).
    """
    patch_at, orient, d = _specs_to_cells(patches, target)
    placed = {nm: cell(*ab, d) for nm, ab in patch_at.items()}
    data, retype = path_to_corridor(set(tree_cells), placed, target, d)
    data = sorted(data)
    sv, n = _symplectic(data)
    isv, _ = _int_symplectic(data)
    N = len(target)
    # pick the parity that hosts both reps AND (preferably) puts the joint in span -- exactly what
    # _assemble_region does; picking merely the first rep-valid phase can miss a measurable joint that
    # only closes in the OTHER parity (e.g. a single L-bend).  A forced ``phase`` restricts the search.
    cands = []
    for ph in ((0, 1) if phase is None else (phase,)):
        pl = _bent_plaquettes(data, retype, ph)
        F = [sv(p["pauli"]) for p in pl if len(p["pauli"]) >= 4]
        rp = {nm: (P, _patch_rep(placed[nm], P, _logical_direction(P, orient[nm]), F, sv, n))
              for nm, P in target}
        reps_ok = all(rp[nm][1] is not None for nm, _ in target)
        jspan = False
        if reps_ok:
            Bph = _IntBasis()
            for p in pl:
                Bph.add(isv(p["pauli"]))
            jspan = Bph.contains(_xor_ints([isv({q: P for q in rp[nm][1]}) for nm, P in target]))
        cands.append((jspan, reps_ok, ph, pl, rp))
    cands.sort(key=lambda c: (c[0], c[1]), reverse=True)   # prefer joint-in-span, then rep-valid
    _, _, phase, plaqs, reps = cands[0]

    # measurability is a property of the WHOLE plaquette span (any complete code is a subset of it)
    Bpool = _IntBasis()
    for p in plaqs:
        Bpool.add(isv(p["pauli"]))
    have_reps = all(reps[nm][1] is not None for nm, _ in target)
    singles, joint_in_span, subp = {}, None, None
    if have_reps:
        svecs = [isv({c: P for c in reps[nm][1]}) for nm, P in target]
        singles = {nm: Bpool.contains(isv({c: P for c in reps[nm][1]})) for nm, P in target}
        joint_in_span = Bpool.contains(_xor_ints(svecs))
        subp = sum(Bpool.contains(_xor_ints([svecs[i] for i in comb]))
                   for r in range(1, N) for comb in itertools.combinations(range(N), r))

    # the code to PRESENT: the joint-MEASURING code (with a readout chain) when measurable, else a
    # maximal complete commuting memory code (so a failing region still shows a full, valid patch).
    # The measuring code comes from the deterministic rule constructor (cut-free, on the region
    # exactly as given) — no randomized search anywhere.
    present, chain = None, set()
    if joint_in_span:
        forbidden = _obstacle_ancillas(patches, {nm for nm, _ in target})
        rb = rule_based_joint_checks(placed, target, orient, data, set(retype), d, max_cut=0,
                                     forbidden=forbidden)
        if rb["checks"] is not None:
            present = rb["checks"]
            chain = _readout_chain(data, present, [reps[nm] for nm, _ in target])
    if present is None:                                    # greedy maximal commuting complete code
        # CRUCIAL: only keep stabilizers that COMMUTE with the target logicals X̄ᵢ, so the reps stay
        # *valid* logicals of the drawn code (a stabilizer that anti-commutes with X̄₁ would make X̄₁
        # not a logical -- which is exactly the odd-overlap inconsistency to avoid).  X̄ᵢ commute with
        # the weight-4 bulk by construction (_patch_rep), so this only constrains the boundary.
        repvecs = [isv({q: P for q in reps[nm][1]}) for nm, P in target if reps[nm][1] is not None]
        present, vecs, B = [], [], _IntBasis()
        for p in sorted(plaqs, key=lambda q: -len(q["pauli"])):    # weight-4 first, then boundary/corner
            v = isv(p["pauli"])
            if (all(_icommute(v, w, n) for w in vecs)
                    and all(_icommute(v, rv, n) for rv in repvecs)   # keep every target logical valid
                    and B.reduce(v) != 0):
                p2 = dict(p); p2["corners"] = sorted(p["pauli"])
                present.append(p2); vecs.append(v); B.add(v)
        # EXPLICITLY search for a readout chain on the COMPLETE code too (a linear solve over its
        # stabilizers) -- it returns empty iff the joint is not a product of them, confirming the
        # "no chain" verdict directly rather than short-circuiting on joint_in_span.
        if have_reps:
            chain = _readout_chain(data, present, [reps[nm] for nm, _ in target])

    # health of the PRESENTED code (k, commutation, no weight-1 leftover)
    Bp, pvecs = _IntBasis(), []
    for p in present:
        v = isv(p["pauli"]); pvecs.append(v); Bp.add(v)
    k = n - Bp.rank
    weights = {}
    for p in present:
        weights[len(p["pauli"])] = weights.get(len(p["pauli"]), 0) + 1
    commute = all(_icommute(pvecs[i], pvecs[j], n)
                  for i in range(len(pvecs)) for j in range(i + 1, len(pvecs)))
    twist = any(P == "Y" for p in present for P in p["pauli"].values())
    w1 = any(Bp.reduce(isv({q: P})) != 0 and all(_icommute(isv({q: P}), w, n) for w in pvecs)
             for q in data for P in "XZ")
    return dict(stabilizers=present, k=k, weights=weights, commute=commute, no_weight1=not w1,
                no_twist=not twist, joint_in_span=joint_in_span, singles_in_span=singles,
                subproducts_in_span=subp, readout_chain=chain, data=data, reps=reps, phase=phase)


ACCEPTANCE_ITEMS = (
    "target logicals commute with all stabilizers",
    "remaining logical dof == N-1",
    "full joint in span",
    "no single logical measured",
    "no proper sub-product measured",
    "no weight-1 leftover logical",
    "all stabilizers commute",
    "no Y / no twist",
    "no MPP",
    "DEM valid",
    "no tick collision",
    "readout chain exists and product == joint",
)


def acceptance(patches, target, tree_cells, seed=0, max_trials=5000):
    """The **strict** subset-joint acceptance gate: ``accept=True`` iff **all twelve**
    :data:`ACCEPTANCE_ITEMS` hold on the actually-**selected measuring code** (not the complete
    memory code).  A geometry is only feasible for a joint measurement when it *measures the joint*,
    so a region with ``remaining logical dof != N-1`` (the joint left as a logical) is a **FAIL** — it
    is never reported as passing.

    Returns a dict: ``accept`` (bool), ``has_measuring_code`` (did the physics layer build a code that
    measures the joint at all), ``items`` (an ordered ``{name: bool|None}`` over the twelve
    conditions; ``None`` = undefined because no measuring code exists), ``k`` (remaining logical dof),
    ``n_stab``, ``data``, ``readout_chain_len``.  ``patches``/``tree_cells`` are as in
    :func:`route_and_build` / :func:`complete_code`.
    """
    patch_at, orient, d = _specs_to_cells(patches, target)
    placed = {nm: cell(*ab, d) for nm, ab in patch_at.items()}
    data, retype = path_to_corridor(set(tree_cells), placed, target, d)
    data = sorted(data)
    N = len(target)
    lay = _assemble_region(placed, target, orient, data, retype, d, seed, max_trials)
    if lay is None:                                        # no code measures the joint -> reject
        cc = complete_code(patches, target, tree_cells)
        items = {k: None for k in ACCEPTANCE_ITEMS}
        items["remaining logical dof == N-1"] = (cc["k"] == N - 1)
        items["full joint in span"] = cc["joint_in_span"]
        items["readout chain exists and product == joint"] = False
        return dict(accept=False, has_measuring_code=False, k=cc["k"],
                    n_stab=len(cc["stabilizers"]), data=len(data),
                    readout_chain_len=len(cc["readout_chain"]), items=items)

    a = acceptance_of_layout(lay)
    return dict(accept=a["accept"], has_measuring_code=True, k=a["k"], n_stab=a["n_stab"],
                data=a["data"], readout_chain_len=a["readout_chain_len"], items=a["items"])


def acceptance_of_layout(lay):
    """The twelve :data:`ACCEPTANCE_ITEMS` computed directly on an already-built
    :class:`MultiPatchLayout` — the layout-level core of :func:`acceptance`, usable on **any** built
    layout, including the convex-corner-cut ones :func:`route_and_build` returns (which
    :func:`acceptance` cannot reach — it rebuilds the standard construction).

    Returns a dict: ``accept`` (bool), ``items`` (an ordered ``{name: bool}`` over the twelve
    conditions), ``N`` (target-logical count), ``data``, ``n_stab``, ``k`` (remaining logical dof),
    ``readout_chain_len``.
    """
    v = lay.verify()
    isv, n = _int_symplectic(lay.data)
    S = [isv(c["pauli"]) for c in lay.checks]
    B = _IntBasis()
    for s in S:
        B.add(s)
    logvecs = [isv({c: P for c in sup}) for _, P, sup in lay.logicals]
    log_ok = all(_icommute(Lv, s, n) for Lv in logvecs for s in S)   # every target logical valid
    joint = _xor_ints(logvecs)
    chain = lay.readout_chain
    prod = _xor_ints([isv(c["pauli"]) for c in lay.checks if c["syn"] in chain])
    chain_ok = bool(chain) and prod == joint                        # chain product IS the joint
    items = {
        "target logicals commute with all stabilizers": log_ok,
        "remaining logical dof == N-1": v["logical_count"],
        "full joint in span": v["joint"],
        "no single logical measured": v["no_single"],
        "no proper sub-product measured": v["no_subjoint"],
        "no weight-1 leftover logical": v["no_weight1_logical"],
        "all stabilizers commute": v["commute"],
        "no Y / no twist": v["no_twist"],
        "no MPP": v["no_mpp"],
        "DEM valid": v["dem_valid"],
        "no tick collision": v["no_tick_collision"],
        "readout chain exists and product == joint": chain_ok,
    }
    return dict(accept=all(bool(x) for x in items.values()), items=items, N=len(lay.logicals),
                data=len(lay.data), n_stab=len(lay.checks), k=len(lay.data) - B.rank,
                readout_chain_len=len(chain))


def collision_report(patches, target, tree_cells, layout=None, seam=False):
    """**Physical placement check**: does the routed code collide with any idle obstacle patch?

    ``patches`` is a list of :class:`PatchSpec` (see :func:`route_and_build`); ``tree_cells`` is the
    routed corridor, a list of coarse cells ``[(a, b), ...]``.

    The routed joint code and every non-target (obstacle) patch are *separate* patches on one chip, so
    they may share no physical qubit.  Returns a dict with three counts (all should be **0**):

    * ``data``               — a routed **data** qubit sits on an obstacle data qubit;
    * ``corner_uses_obstacle`` — a routed **stabilizer** uses an obstacle data qubit as a corner;
    * ``ancilla``            — a routed **ancilla** (plaquette centre) coincides with an obstacle
      patch's own boundary ancilla (the ``keepout=0`` failure: edge-adjacent cells share that line).

    Plus ``clean`` (all zero) and per-obstacle ``details``.  Run it on any routed corridor to prove the
    layout is physically placeable.
    """
    patch_at, _orient, d = _specs_to_cells(patches, target, seam=seam)
    if layout is not None:                 # judge the ACTUAL built code, not a re-derivation
        stabs, rdata = layout.checks, set(layout.data)
    else:
        cc = complete_code(patches, target, tree_cells)
        stabs, rdata = cc["stabilizers"], set(cc["data"])
    tnames = {nm for nm, _ in target}
    onames = [nm for nm in patch_at if nm not in tnames]
    routed_syn = {tuple(int(v) for v in p["syn"]) for p in stabs}
    routed_data = rdata
    routed_corners = (set().union(*[set(p["pauli"]) for p in stabs]) if stabs else set())
    nd, nc, na, details = 0, 0, 0, {}
    spec_of = {p.name: p for p in patches}
    for on in onames:
        Odata = cell(*patch_at[on], d, seam)
        Osyn = {tuple(int(v) for v in c["syn"]) for c in place_patch(spec_of[on])["checks"]}
        a, b, c = routed_data & Odata, routed_corners & Odata, routed_syn & Osyn
        nd += len(a); nc += len(b); na += len(c)
        if a or b or c:
            details[on] = dict(data=sorted(a), corner=sorted(b), ancilla=sorted(c))
    return dict(data=nd, corner_uses_obstacle=nc, ancilla=na,
                clean=(nd == 0 and nc == 0 and na == 0), details=details)




# =============================================================================
# merged public surface
# =============================================================================
__all__ = sorted(set(_ALL_DETERMINISTIC) | set(_ALL_ROUTING) | {
    "PatchSpec", "place_patch", "BentLayoutError",
    "MultiPatchLayout",
})


# =============================================================================
# SECTION: RotatedRoutedMultiPatchCoupler — IR coupler over the routed layout
# =============================================================================
# The atomic-operation face of the seam-column design: wraps route_and_build's
# verified merged layout as a LogicalCouplerProtocol so the experiment layer
# (protocols/routed_multi_patch_ls.py) can drive the standard IR sequence
# (initialize -> SE -> activate_coupler -> SE -> readout).  Ownership rule:
# a merged check is COUPLER-owned unless it is byte-identical to a patch's
# registered check (rule-0 native lock) — majority patches register their
# textbook construction, minority patches the CONJUGATE-CONVENTION one
# (conjugate_patch_records), so both keep their interiors native; every
# registered check not carried into the merge is listed in
# conflicting_stabilizer_coords so QECSystem.activate_coupler pauses it.

from lightstim.ir.coupler import LogicalCouplerProtocol as _LCP


class RotatedRoutedMultiPatchCoupler(_LCP):
    """Routed multi-patch coupler for ROTATED patches (seam-column design)."""

    EXPECTED_PATCH_COUNT = None
    _FLIP_O = {'X_horizontal': 'X_vertical', 'X_vertical': 'X_horizontal'}

    @staticmethod
    def _key(ch):
        return (tuple(ch['syn']),
                frozenset((tuple(q), P) for q, P in ch['pauli'].items()))

    def _build_coupler_geometry(self, coupler_patch, patches, *, specs, target,
                                subset_route=None, seam=True, route=None,
                                minority_names=frozenset()):
        from types import SimpleNamespace
        r = subset_route
        if r is None:
            r = route_and_build(specs, target, seam=seam, route=route)
        if r.status != 'ok':
            raise ValueError(f'route_and_build failed: {r.status} — {r.message}')
        lay = r.layout
        tnames = {nm for nm, _ in target}
        coupler_patch.conflicting_stabilizer_coords = set()

        merged_keys = {self._key(ch) for ch in lay.checks}
        kept, native_syns, patch_cells = set(), set(), set()
        for s in specs:
            if s.name not in tnames:
                continue
            patch_cells |= {(s.origin[0] + 2 * i, s.origin[1] + 2 * j)
                            for i in range(s.distance) for j in range(s.distance)}
            # the construction the patch REGISTERED in the system; minority
            # patches register the conjugate-convention construction
            # (transposed geometry, records typeswapped)
            reg_orient = self._FLIP_O[s.orientation] if s.name in minority_names \
                else s.orientation
            reg = place_patch(SimpleNamespace(origin=s.origin, distance=s.distance,
                                              orientation=reg_orient))['checks']
            if s.name in minority_names:
                reg = [dict(ch, type=_FLIP[ch['type']],
                            pauli={q: _FLIP[P] for q, P in ch['pauli'].items()})
                       for ch in reg]
            for ch in reg:
                native_syns.add(tuple(ch['syn']))
                key = self._key(ch)
                if key not in merged_keys:
                    coupler_patch.conflicting_stabilizer_coords.add(tuple(ch['syn']))
                else:
                    kept.add(key)

        # new data qubits: corridor + seam columns
        for q in sorted(set(map(tuple, lay.data)) - patch_cells):
            coupler_patch.add_qubit(q[0], q[1], role='data')
        # coupler-owned checks + any genuinely new ancilla positions
        added = set()
        for ch in lay.checks:
            if self._key(ch) in kept:
                continue
            syn = tuple(ch['syn'])
            typ = ch.get('type') or next(iter(set(ch['pauli'].values())))
            kf = ch.get('kf')
            if kf is not None:
                # stretched (kf) record: B reads out in X, flag A and the
                # shared relay S in Z (same apparatus convention as
                # RotatedSeamWallCoupler)
                if syn not in native_syns and syn not in added:
                    coupler_patch.add_qubit(syn[0], syn[1], role='syndrome_x')
                    added.add(syn)
                for coord, role in ((tuple(kf['flag']), 'syndrome_z'),
                                    (tuple(kf['shared']), 'syndrome_z')):
                    if coord not in native_syns and coord not in added:
                        coupler_patch.add_qubit(coord[0], coord[1], role=role)
                        added.add(coord)
                coupler_patch.stabilizers.append({
                    'pauli': {tuple(q): P for q, P in ch['pauli'].items()},
                    'type': 'MIXED',
                    'syn_coord': syn,
                    'kf': {'flag': tuple(kf['flag']),
                           'shared': tuple(kf['shared']),
                           'orient': kf['orient']},
                })
                continue
            if syn not in native_syns and syn not in added:
                coupler_patch.add_qubit(syn[0], syn[1],
                                        role=('syndrome_z' if typ == 'Z'
                                              else 'syndrome_x'))
                added.add(syn)
            coupler_patch.stabilizers.append({
                'pauli': {tuple(q): P for q, P in ch['pauli'].items()},
                'type': typ if typ in ('X', 'Z') else 'MIXED',
                'syn_coord': syn,
            })
        coupler_patch.routed_layout = lay          # introspection for the protocol layer
        coupler_patch.subset_route = r
