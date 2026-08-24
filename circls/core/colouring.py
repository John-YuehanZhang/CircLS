"""Birth-colour planning: global 2-colouring of the corridor parity law.

STATUS (2026-08-04): EXPERIMENTAL, default OFF.  First-contact measurement:
global letter-uniformity is not a free lunch — a colour swap also flips the
patch's live orientation at allocation, fighting the first-letter
orientation heuristic across every step the patch touches (ghz_8: 0 -> 7
rotations), and cascading into downstream corners (compact-7 slot clash on
bbpssw, a stage-generator StopIteration on dj_8/steane).  The principled
version needs COST-AWARE colouring (walls saved vs orientation churn), not
blind constraint propagation.  Kept for that follow-up.

The corridor parity law says patches joined by one corridor measure the
SAME effective letter iff they share the SAME colour convention.  Rather
than reconciling per-step mismatches with walls (>=2 walls per step gets
geometrically tight) or rotations (swap rotation is banned as non-FT,
litinski keeps colour), plan the colours ONCE at birth: within every PPM
step, same-letter targets must share a colour and different-letter targets
must differ — a graph 2-colouring over the whole program.  A conjugate-born
(colour-swapped) patch measuring Z behaves as X on the corridor, so a
satisfying colouring makes every step's effective letters uniform: plain
corridors, zero walls, zero rotations.

Completeness (the user's prior result, 2026-07): a program whose colour
graph has an odd cycle is exactly one whose PPM set anticommutes — not a
legal computation.  So for valid Clifford programs the colouring always
exists; an odd cycle here raises loudly instead of guessing.

Polarity: both global flips satisfy the constraints; we pick the one that
swaps fewer patches (ties: swap the lexicographically later group) so the
default convention stays dominant.  |Y> gadget ancillas participate like
any other patch.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, Iterable, List, Tuple


class OddColourCycleError(ValueError):
    """The program's colour-constraint graph is not 2-colourable — by the
    parity-law completeness result this means the declared PPM set does not
    commute (not a legal computation)."""


def plan_birth_colours(steps: Iterable[Iterable[Tuple[str, str]]],
                       ) -> FrozenSet[str]:
    """Return the set of patch names to birth colour-swapped.

    ``steps``: iterable of PPM interaction lists ``[(name, letter), ...]``
    (a ``PPMStep.interaction_type`` works as-is).
    """
    # collect parity constraints: (a, b, want_diff)
    edges: List[Tuple[str, str, bool]] = []
    names: List[str] = []
    seen = set()
    for st in steps:
        members = list(st)
        for nm, _ in members:
            if nm not in seen:
                seen.add(nm)
                names.append(nm)
        anchor, a_letter = members[0]
        for nm, letter in members[1:]:
            edges.append((anchor, nm, letter != a_letter))

    # BFS 2-colouring with parity, per connected component
    colour: Dict[str, int] = {}
    adj: Dict[str, List[Tuple[str, int]]] = {nm: [] for nm in names}
    for a, b, diff in edges:
        adj[a].append((b, 1 if diff else 0))
        adj[b].append((a, 1 if diff else 0))
    swapped: List[str] = []
    for root in names:
        if root in colour:
            continue
        colour[root] = 0
        comp = [root]
        queue = [root]
        while queue:
            v = queue.pop(0)
            for u, parity in adj[v]:
                want = colour[v] ^ parity
                if u not in colour:
                    colour[u] = want
                    comp.append(u)
                    queue.append(u)
                elif colour[u] != want:
                    raise OddColourCycleError(
                        f"colour constraints between {v!r} and {u!r} form an "
                        f"odd cycle — the declared PPM set does not commute; "
                        f"this input is not a legal Clifford computation "
                        f"(parity-law completeness, 2026-07)")
        # per-component polarity: |Y> ancillas MUST stay standard (the
        # Gidney birth v1 rejects colour-swapped); otherwise minimize swaps
        y_bits = {colour[nm] for nm in comp if nm.startswith("y")}
        if len(y_bits) == 2:
            raise OddColourCycleError(
                f"component of {root!r} forces two |Y> ancillas onto "
                f"opposite colours — colour-swapped Gidney birth is not "
                f"implemented (v1); this program needs the wall fallback")
        if y_bits:
            flip = y_bits.pop()               # make every y standard (0)
        else:
            ones = sum(colour[nm] for nm in comp)
            flip = 1 if (ones > len(comp) - ones
                         or (2 * ones == len(comp)
                             and sorted(nm for nm in comp if colour[nm])
                             < sorted(nm for nm in comp if not colour[nm]))
                         ) else 0
        swapped += [nm for nm in comp if colour[nm] ^ flip]
    return frozenset(swapped)
