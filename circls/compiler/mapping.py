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
                            ) -> List[PatchSpec]:
    """Optimized assignment: run the portfolio +
    refinement + exact-judge pipeline of ``circls.compiler.assignment`` against the
    program's PPM steps, then place with the winning cells."""
    from circls.compiler.assignment import optimize_assignment
    names = list(names)
    first_letters = first_letters or {}
    forced = orients or {}
    all_orients = {nm: forced.get(nm) or patch_orientation(
                       nm, first_letters.get(nm))
                   for nm in names}
    cells, _report = optimize_assignment(names, steps, orients=all_orients)
    return place_patches(names, distance, first_letters=first_letters,
                         cells=cells, orients=forced)
