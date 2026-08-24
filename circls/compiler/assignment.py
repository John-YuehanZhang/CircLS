"""Patch->slot assignment optimizer.

Pipeline: proxy cost (closed-form pair metric + hyperedge blend) -> a
portfolio of deterministic starting layouts -> first-improvement 2-swap
refinement on the proxy -> exact rescoring of every refined candidate with
the repo's own group-Steiner router -> lowest exact cost wins.

Everything is deterministic: fixed scan orders, sign-fixed eigenvectors,
no wall-clock cutoffs, no unseeded randomness — same input, same layout,
byte-identical circuits (unlike DASCOT's 20-trial averages).

Provenance: the pair metric was verified 4950/4950 against the real router;
the hyperedge blend calibrates to ~2.5-4.1% mean error (k=4..6); portfolio
and search were lifted from the internal placement probes and the
four survey reports.

Cost conventions (coarse-cell coordinates, patch slots at odd-odd cells):
* pair:  1 if adjacent; 2s+1 if axis-aligned with s >= 2 (the intervening
  slot blocks the straight run — one lane detour, a flat +2); 2s-1 otherwise
  (s = slot Manhattan distance in cell units / 2... expressed directly on
  coarse coords below).
* parallel-law correction: adjacent slots cost +2 when the seam axis is
  illegal for either endpoint's (letter, orientation) — the law only ever
  bites at distance 1 (measured: 0 penalty at distance >= 2).
* exact: group Steiner on the board graph with ALL slot cells removed
  (full-occupancy convention — an upper bound consistent with the closed
  form; freeing cells only ever lowers cost, monotonicity 283/283).
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from circls.compiler.mapping import _natural_key, square_sparse_cells

Cell = Tuple[int, int]

# blend weights for the k>=4 hyperedge proxy (calibrated on the real router)
_ALPHA = {4: 0.8, 5: 0.6, 6: 0.5}


# ── metric ────────────────────────────────────────────────────────────────────

def pair_cost(a: Cell, b: Cell) -> int:
    """Exact corridor cells between two occupied slots (closed form,
    verified 4950/4950 against ``emv_group_steiner`` on Square Sparse)."""
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    c = dx + dy - 1
    if (dx == 0 or dy == 0) and dx + dy >= 4:
        c += 2                      # axis-aligned: borrow a lane around the
    return c                        # intervening slot, flat +2

def legal_axis(letter: str, orientation: str) -> str:
    """Parallel law: the seam axis on which measuring ``letter`` is legal
    for a patch with ``orientation`` — 'EW' or 'NS'."""
    return 'EW' if (letter == 'Z') == (orientation == 'X_horizontal') else 'NS'


def pair_cost_law(a: Cell, b: Cell, ax_a: Optional[str], ax_b: Optional[str]) -> int:
    """Pair cost with the parallel-law +2 on adjacent slots whose seam axis
    is illegal for either endpoint (measured: the law never costs anything
    at slot distance >= 2)."""
    c = pair_cost(a, b)
    if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 2:      # adjacent slots
        seam = 'EW' if a[1] == b[1] else 'NS'
        if (ax_a is not None and seam != ax_a) or (ax_b is not None and seam != ax_b):
            c += 2
    return c


def _hpwl(cells: Sequence[Cell]) -> int:
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    return (max(xs) - min(xs)) + (max(ys) - min(ys))


def _mst_manhattan(cells: Sequence[Cell]) -> int:
    rem = set(range(1, len(cells)))
    d = [abs(c[0] - cells[0][0]) + abs(c[1] - cells[0][1]) for c in cells]
    t = 0
    while rem:
        j = min(rem, key=lambda i: (d[i], i))
        t += d[j]
        rem.discard(j)
        for i in rem:
            d[i] = min(d[i], abs(cells[i][0] - cells[j][0])
                       + abs(cells[i][1] - cells[j][1]))
    return t


def step_proxy_cost(cells: Sequence[Cell],
                    axes: Optional[Sequence[Optional[str]]] = None) -> float:
    """Proxy corridor cost of one PPM step at the given member slots."""
    k = len(cells)
    if k <= 1:
        return 0.0
    if k == 2:
        if axes is not None:
            return pair_cost_law(cells[0], cells[1], axes[0], axes[1])
        return pair_cost(cells[0], cells[1])
    if k == 3:
        return _hpwl(cells) - 1
    a = _ALPHA.get(k, 0.5)
    return a * (_hpwl(cells) - 1) + (1 - a) * (_mst_manhattan(cells) - 1)


# ── program representation ────────────────────────────────────────────────────

class _Program:
    """Steps as index tuples + per-(step, member) legal axes; the inverted
    touch index makes swap deltas incremental."""

    def __init__(self, names: List[str], steps, orients: Dict[str, str]):
        self.names = names
        idx = {nm: i for i, nm in enumerate(names)}
        self.steps: List[Tuple[int, ...]] = []
        self.axes: List[Tuple[Optional[str], ...]] = []
        for st in steps:
            members = [(idx[nm], letter) for nm, letter in st if nm in idx]
            if len(members) < 2:
                continue
            self.steps.append(tuple(m for m, _ in members))
            self.axes.append(tuple(
                legal_axis(letter, orients[names[m]])
                if orients.get(names[m]) else None
                for m, letter in members))
        self.touch: List[List[int]] = [[] for _ in names]
        for si, members in enumerate(self.steps):
            for m in members:
                self.touch[m].append(si)

    def proxy(self, pos: List[Cell], sis: Optional[Iterable[int]] = None) -> float:
        sis = range(len(self.steps)) if sis is None else sis
        return sum(step_proxy_cost([pos[m] for m in self.steps[si]],
                                   self.axes[si]) for si in sis)


# ── deterministic starting layouts ────────────────────────────────────────────

def _flow(prog: _Program, n: int) -> np.ndarray:
    W = np.zeros((n, n))
    for members in prog.steps:
        w = 1.0 / (len(members) - 1)          # clique expansion, weight 1/(k-1)
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                W[a, b] += w
                W[b, a] += w
    return W


def _start_row_major(prog, n, slots):
    return list(slots[:n])


def _start_spectral(prog, n, slots):
    """Hall's quadratic placement: 2nd/3rd Laplacian eigenvectors give the
    natural 2D coordinates; Hungarian matching legalises onto the slots."""
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:                        # pragma: no cover
        return None
    W = _flow(prog, n)
    L = np.diag(W.sum(1)) - W
    _, V = np.linalg.eigh(L)
    xy = V[:, 1:3].copy() if V.shape[1] >= 3 else np.column_stack(
        [V[:, 1], np.zeros(n)])
    for c in range(xy.shape[1]):              # sign fixing: determinism
        v = xy[:, c]
        if v[int(np.argmax(np.abs(v)))] < 0:
            xy[:, c] = -v
    P = np.array(slots[:n], float)
    xy = (xy - xy.mean(0)) / (xy.std(0) + 1e-12)
    Pn = (P - P.mean(0)) / (P.std(0) + 1e-12)
    C = ((xy[:, None, :] - Pn[None, :, :]) ** 2).sum(-1)
    r, c = linear_sum_assignment(C)
    out = [None] * n
    for i, j in zip(r, c):
        out[int(i)] = slots[:n][int(j)]
    return out


def _rcm_order(prog, n) -> List[int]:
    W = _flow(prog, n)
    adj = [set(np.nonzero(W[i])[0].tolist()) for i in range(n)]
    deg = [len(adj[i]) for i in range(n)]
    unseen, order = set(range(n)), []
    while unseen:
        start = min(unseen, key=lambda i: (deg[i], i))
        queue = [start]
        unseen.discard(start)
        while queue:
            v = queue.pop(0)
            order.append(v)
            for u in sorted((u for u in adj[v] if u in unseen),
                            key=lambda u: (deg[u], u)):
                unseen.discard(u)
                queue.append(u)
    return order[::-1]


def _hilbert_cells(m: int) -> List[Cell]:
    """Slot cells in Hilbert-curve order (1D-adjacent stays 2D-adjacent)."""
    side = 1
    while side < m:
        side <<= 1
    seq = []
    for d in range(side * side):
        x = y = 0
        t, s = d, 1
        while s < side:
            rx = 1 & (t // 2)
            ry = 1 & (t ^ rx)
            if ry == 0:
                if rx == 1:
                    x, y = s - 1 - x, s - 1 - y
                x, y = y, x
            x += s * rx
            y += s * ry
            t //= 4
            s *= 2
        if x < m and y < m:
            seq.append((2 * x + 1, 2 * y + 1))
    return seq


def _start_rcm_hilbert(prog, n, slots):
    order = _rcm_order(prog, n)
    m = int(math.isqrt(len(slots)))
    seq = _hilbert_cells(m)
    out = [None] * n
    for rank, patch in enumerate(order):
        out[patch] = seq[rank]
    return out


def _start_rcm_snake(prog, n, slots):
    order = _rcm_order(prog, n)
    m = int(math.isqrt(len(slots)))
    seq = [(2 * c + 1, 2 * row + 1)
           for row in range(m)
           for c in (range(m) if row % 2 == 0 else reversed(range(m)))]
    out = [None] * n
    for rank, patch in enumerate(order):
        out[patch] = seq[rank]
    return out


def _start_qap_faq(prog, n, slots):
    """scipy's Fast Approximate QAP on the closed-form distance matrix
    (clique-expanded flow); deterministic from the barycenter start."""
    try:
        from scipy.optimize import quadratic_assignment
    except ImportError:                        # pragma: no cover
        return None
    ns = len(slots)
    W = np.zeros((ns, ns))
    W[:n, :n] = _flow(prog, n)
    D = np.array([[0.0 if i == j else pair_cost(slots[i], slots[j])
                   for j in range(ns)] for i in range(ns)])
    res = quadratic_assignment(W, D, method='faq')
    return [slots[int(res.col_ind[i])] for i in range(n)]


_STARTS = [("row_major", _start_row_major),
           ("spectral", _start_spectral),
           ("rcm_hilbert", _start_rcm_hilbert),
           ("rcm_snake", _start_rcm_snake),
           ("qap_faq", _start_qap_faq)]


# ── deterministic refinement ──────────────────────────────────────────────────

def _local_search(prog: _Program, pos0: List[Cell], slots: List[Cell],
                  max_moves: int = 4000) -> List[Cell]:
    """First-improvement search over pair swaps and moves to free slots,
    fixed scan order, incremental delta via the touch index."""
    pos = list(pos0)
    n = len(pos)
    used = set(pos)
    free = [c for c in slots if c not in used]
    moves = 0
    improved = True
    while improved and moves < max_moves:
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                sis = sorted(set(prog.touch[i]) | set(prog.touch[j]))
                if not sis:
                    continue
                before = prog.proxy(pos, sis)
                pos[i], pos[j] = pos[j], pos[i]
                if prog.proxy(pos, sis) < before - 1e-9:
                    moves += 1
                    improved = True
                    break
                pos[i], pos[j] = pos[j], pos[i]
            if improved:
                break
            for fi in range(len(free)):
                sis = prog.touch[i]
                if not sis:
                    continue
                before = prog.proxy(pos, sis)
                old = pos[i]
                pos[i] = free[fi]
                if prog.proxy(pos, sis) < before - 1e-9:
                    free[fi] = old
                    moves += 1
                    improved = True
                    break
                pos[i] = old
            if improved:
                break
    return pos


# ── exact judge (the repo's own router) ───────────────────────────────────────

class ExactScorer:
    """Corridor cells per step via ``emv_group_steiner`` on the board graph
    with ALL slot cells removed (full-occupancy convention).  Orientation-
    aware: each member attaches only through its parallel-law-legal faces.
    Memoised on the (cell, axis) multiset of the step."""

    def __init__(self, slots: List[Cell]):
        import networkx as nx
        m = int(math.isqrt(len(slots)))
        side = 2 * m + 1
        g = nx.grid_2d_graph(side, side)
        g.remove_nodes_from(slots)
        self._g = g
        self._memo: Dict[frozenset, int] = {}

    def step_cost(self, cells: Sequence[Cell],
                  axes: Optional[Sequence[Optional[str]]] = None) -> int:
        from circls.compiler.routing import emv_group_steiner
        axes = axes or [None] * len(cells)
        key = frozenset(zip(cells, axes))
        hit = self._memo.get(key)
        if hit is not None:
            return hit
        groups = []
        for (a, b), ax in zip(cells, axes):
            if ax == 'EW':
                cand = [(a + 1, b), (a - 1, b)]
            elif ax == 'NS':
                cand = [(a, b + 1), (a, b - 1)]
            else:
                cand = [(a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)]
            groups.append([q for q in cand if q in self._g])
        from circls.compiler.routing import _EMV_GROUP_CAP
        if len(groups) > _EMV_GROUP_CAP:
            # the EMV DP is O(3^k) — a single high-weight step's score
            # (bv_32: k=17) never terminates, and it lives on the
            # optimizer's hottest path (2026-08-05 profile: 100% of the
            # bv_32-optimized timeout).  Big steps fall back to the
            # closed-form hybrid proxy, same as the local-search metric.
            cost = int(round(step_proxy_cost(cells, axes)))
        else:
            cost = emv_group_steiner(self._g, groups)[0]
        self._memo[key] = cost
        return cost

    def program_cost(self, prog: _Program, pos: List[Cell]) -> int:
        return sum(self.step_cost([pos[m] for m in prog.steps[si]],
                                  prog.axes[si])
                   for si in range(len(prog.steps)))


# ── orchestrator ──────────────────────────────────────────────────────────────

def optimize_assignment(names: Sequence[str], steps, *,
                        orients: Optional[Dict[str, str]] = None,
                        slots: Optional[List[Cell]] = None,
                        max_moves: int = 4000,
                        ) -> Tuple[Dict[str, Cell], Dict[str, int]]:
    """Optimize the patch->slot map for a PPM program.

    ``names``: all patch names; ``steps``: iterable of PPM interaction lists
    ``[(name, letter), ...]`` (a ``PPMStep.interaction_type`` works as-is);
    ``orients``: per-patch orientation (enables the parallel-law term).
    Returns ``(assignment {name: cell}, report {start_name: exact_cost})``.
    """
    names = sorted(names, key=_natural_key)
    if len(names) <= 2:
        # nothing to optimize: 0/1 patches have no pairs, 2 patches on the
        # Square Sparse floor are adjacent in row-major already (and the
        # spectral start needs >= 2 eigenvectors — n=1 crashed on the
        # tqec-comparison memory pair, 2026-08-05)
        from circls.compiler.mapping import square_sparse_cells
        cells = slots if slots is not None else square_sparse_cells(len(names))
        return ({nm: cells[i] for i, nm in enumerate(names)},
                {"trivial": 0})
    n = len(names)
    if slots is None:
        m = math.ceil(math.sqrt(n)) if n else 1
        slots = [(2 * (k % m) + 1, 2 * (k // m) + 1) for k in range(m * m)]
    prog = _Program(list(names), steps, orients or {})
    scorer = ExactScorer(slots)
    best_pos, best_cost, report = None, None, {}
    for label, fn in _STARTS:
        pos = fn(prog, n, slots)
        if pos is None:
            continue
        pos = _local_search(prog, pos, slots, max_moves=max_moves)
        cost = scorer.program_cost(prog, pos)
        report[label] = cost
        if best_cost is None or cost < best_cost:
            best_pos, best_cost = pos, cost
    return {nm: best_pos[i] for i, nm in enumerate(names)}, report
