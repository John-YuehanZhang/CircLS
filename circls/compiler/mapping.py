"""Static placement (mapping) for the QASM->stim pipeline.

Transcribed from the literature — not invented here:

* DASCOT (Molavi, Xu, Tannu, Albarghouthi; OOPSLA 2025), Sec. 6
  "Architectures":
      "The Square Sparse architecture ... is the smallest square grid which
       includes enough space to surround each circuit qubit with routing
       ancillae on all sides.  For a circuit with n qubits, this results in
       a side length of 2*ceil(sqrt(n)) + 1."
* Watkins et al. (Quantum 8, 1354 (2024)) — the liblsqecc "sparse layout":
  data patches interleaved with routing space.  Same family; TopoLS Sec. 6.1
  records that TopoLS, DASCOT and LaSsynth all adopt the sparse layout, so
  using it keeps every baseline on the same floorplan family.

Transcription onto our coarse seam grid: patch cells sit at odd-odd
coordinates ``(2i+1, 2j+1)``, ``i, j in [0, ceil(sqrt(n)))``; every even row
and column is routing space, and the outer ring stays clear (DASCOT reserves
it for magic states — none in the Clifford scope, but the clear ring
preserves boundary routing).

Assignment is row-major in natural name order — a disclosed heuristic, not a
contribution.  Orientation: |Y> gadget ancillas must be ``X_vertical`` (the
Gidney birth v1 constraint); data patches default from their first-use
letter (``Z -> X_horizontal``, ``X -> X_vertical``) so the first joint tends
to satisfy the seam-parallel orientation rule without a rotation; the
planner's auto-rotate covers the rest.

Dynamic slot reuse (retired patches' cells hosting later-born patches) is a
planned planner pass on top of this static floor.
"""
from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List, Optional, Tuple

from circls.core.routed_multi_patch_ls import PatchSpec, origin_of


def grid_side(n: int) -> int:
    """DASCOT Square Sparse side length: ``2*ceil(sqrt(n)) + 1``."""
    if n <= 0:
        return 1
    return 2 * math.ceil(math.sqrt(n)) + 1


def square_sparse_cells(n: int) -> List[Tuple[int, int]]:
    """The first ``n`` patch cells of the Square Sparse grid, row-major:
    odd-odd coordinates ``(2i+1, 2j+1)`` with ``i, j in [0, ceil(sqrt(n)))``."""
    if n <= 0:
        return []
    m = math.ceil(math.sqrt(n))
    return [(2 * (k % m) + 1, 2 * (k // m) + 1) for k in range(n)]


def _natural_key(name: str):
    return [int(tok) if tok.isdigit() else tok
            for tok in re.split(r"(\d+)", name)]


def patch_orientation(name: str, first_letter: Optional[str]) -> str:
    """Birth-orientation rule: |Y> ancillas are pinned ``X_vertical``
    (Gidney birth v1); data patches follow their first-use letter."""
    if name.startswith("y"):
        return "X_vertical"
    return "X_horizontal" if first_letter == "Z" else "X_vertical"


def magic_orientations(cells: Dict[str, Tuple[int, int]], steps,
                       magic_names: Iterable[str],
                       default: str = "X_vertical",
                       orients: Optional[Dict[str, str]] = None,
                       ) -> Dict[str, str]:
    """Birth orientation of each |+>-proxy gadget ancilla from the geometry
    of the step(s) that consume it.

    The seam rule (``multi_patch_coupler._seam_orientation_violation``,
    ``assignment.legal_axis``) says a Z measurement through an E/W seam
    needs ``X_horizontal`` and through a N/S seam ``X_vertical`` (X: the
    transpose).  So the ancilla should face its partners: every partner
    patch in a consuming step votes for the axis of its displacement on the
    coarse grid (|da| > |db| -> E/W, |db| > |da| -> N/S), weighted by
    1 / Manhattan distance, and the ancilla's letter in that step maps the
    winning axis to an orientation.  A tied vote (diagonal partners, or
    partners on both sides) is settled by the assignment's exact corridor
    judge (``assignment.ExactScorer``) on the consuming step, both
    orientations scored with the partners' orientations ``orients``; a tie
    there keeps ``default``.  Only |+> (or any non-|Y>) ancillas may use
    this: the Gidney |Y> birth layout is fixed ``X_vertical``."""
    orients = orients or {}
    out: Dict[str, str] = {}
    for nm in magic_names:
        ew = ns = 0.0
        a0, b0 = cells[nm]
        letter = "Z"
        consuming = []
        for st in steps:
            members = list(st)
            if nm not in [m for m, _ in members]:
                continue
            consuming.append(members)
            letter = next(l for m, l in members if m == nm)
            for m, _ in members:
                if m == nm or m not in cells:
                    continue
                da, db = abs(cells[m][0] - a0), abs(cells[m][1] - b0)
                w = 1.0 / max(da + db, 1)
                if da > db:
                    ew += w
                elif db > da:
                    ns += w
        ew_orient = "X_horizontal" if letter == "Z" else "X_vertical"
        ns_orient = "X_vertical" if letter == "Z" else "X_horizontal"
        if ew > ns:
            out[nm] = ew_orient
        elif ns > ew:
            out[nm] = ns_orient
        else:
            out[nm] = _judge_tie(cells, consuming, nm, orients, default)
    return out


def _judge_tie(cells, consuming, nm, orients, default):
    """Both orientations of ancilla ``nm`` scored by the assignment's exact
    corridor judge over its consuming steps (all slots blocked, each member
    attaching only through its parallel-law-legal faces); the strictly
    cheaper one wins, a tie keeps ``default``."""
    if not consuming:
        return default
    import math
    from circls.compiler.assignment import ExactScorer, legal_axis
    n = len(cells)
    m = max(math.ceil(math.sqrt(n)),
            max(max(c) for c in cells.values()) // 2 + 1)
    judge = ExactScorer([(2 * (k % m) + 1, 2 * (k // m) + 1)
                         for k in range(m * m)])
    cost = {}
    for orient in ("X_vertical", "X_horizontal"):
        total = 0
        for members in consuming:
            axes = [legal_axis(l, orient if p == nm else
                               (orients.get(p) or patch_orientation(p, None)))
                    for p, l in members if p in cells]
            total += judge.step_cost([cells[p] for p, _ in members
                                      if p in cells], axes)
        cost[orient] = total
    if cost["X_horizontal"] < cost["X_vertical"]:
        return "X_horizontal"
    if cost["X_vertical"] < cost["X_horizontal"]:
        return "X_vertical"
    return default


def place_patches(names: Iterable[str], distance: int,
                  first_letters: Optional[Dict[str, str]] = None,
                  cells: Optional[Dict[str, Tuple[int, int]]] = None,
                  orients: Optional[Dict[str, str]] = None,
                  ) -> List[PatchSpec]:
    """Place ``names`` on the Square Sparse grid.  Default assignment is
    row-major in natural name order; pass ``cells`` (name -> coarse cell)
    to use an externally optimized assignment (circls.compiler.assignment), and
    ``orients`` (name -> orientation) to override the birth-orientation
    rule for those patches (docs/API_HOOKS.md: orientation=)."""
    ordered = sorted(names, key=_natural_key)
    first_letters = first_letters or {}
    orients = orients or {}
    if cells is None:
        cells = dict(zip(ordered, square_sparse_cells(len(ordered))))
    specs = []
    for nm in ordered:
        a, b = cells[nm]
        orient = orients.get(nm) or patch_orientation(
            nm, first_letters.get(nm))
        specs.append(PatchSpec(nm, origin_of(a, b, distance, seam=True),
                               distance, orient))
    return specs


def place_patches_optimized(names: Iterable[str], steps, distance: int,
                            first_letters: Optional[Dict[str, str]] = None,
                            orients: Optional[Dict[str, str]] = None,
                            free_magic: Optional[Iterable[str]] = None,
                            ) -> List[PatchSpec]:
    """Optimized assignment: run the portfolio +
    refinement + exact-judge pipeline of ``circls.compiler.assignment`` against the
    program's PPM steps, then place with the winning cells.  ``free_magic``
    names the gadget ancillas whose orientation is chosen by
    ``magic_orientations`` (|+> proxy); the assignment is re-optimized
    against the chosen orientations until both agree (at most 3 rounds)."""
    from circls.compiler.assignment import optimize_assignment
    names = list(names)
    first_letters = first_letters or {}
    forced = orients or {}
    free_magic = [nm for nm in (free_magic or []) if nm not in forced]
    all_orients = {nm: forced.get(nm) or patch_orientation(
                       nm, first_letters.get(nm))
                   for nm in names}
    magic = {}
    for _ in range(3):        # assignment and ancilla orientation depend on
        cells, _report = optimize_assignment(names, steps,      # each other
                                             orients=all_orients)
        if not free_magic:
            break
        new = magic_orientations(cells, steps, free_magic,
                                 orients=all_orients)
        if new == magic:
            break
        magic = new
        all_orients.update(magic)
    return place_patches(names, distance, first_letters=first_letters,
                         cells=cells, orients={**magic, **forced})
