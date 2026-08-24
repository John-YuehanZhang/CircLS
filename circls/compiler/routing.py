"""Corridor search (paper Section 3): propose ancilla-path trees.

The search half of routing: the corridor graph over free tiles, the
exact EMV group-Steiner enumeration (used whenever the step has at
most _EMV_GROUP_CAP groups), and the greedy candidate pool (nearest-
first plus the fixed attach orders).  Everything here only PROPOSES
cell sets; the construction and its oracle live in
circls.core.multi_patch_coupler, whose route_and_build consumes these
proposals (it imports this module lazily, so the two never form an
import cycle at load time).  Replaceable from outside via the router=
hook (docs/API_HOOKS.md).
"""
import itertools

import networkx as nx

from circls.core.multi_patch_coupler import NEIGH, _cheb, cell

def _corridor_graph(patch_at, target, d, pad, keepout=1, seam=False):
    """Build the coarse-cell corridor graph.

    A coarse cell is **corridor-eligible** iff it holds no patch and it stays clear of every obstacle
    patch by a **keep-out margin** of ``keepout`` cells (king-move / Chebyshev distance ``> keepout``).
    ``keepout=1`` (the default) forbids any cell **adjacent** to an obstacle, so the routed code never
    shares a boundary ancilla line with an idle obstacle patch (``keepout=0`` did — two edge-adjacent
    cells share the ancilla row/column between them, a physical collision).

    Returns ``(G, corridor, placed, occupied, obstacle_fp, onames)``.
    """
    tnames = [nm for nm, _ in target]
    onames = [nm for nm in patch_at if nm not in tnames]
    occupied = {ab: nm for nm, ab in patch_at.items()}
    placed = {nm: cell(*ab, d, seam) for nm, ab in patch_at.items()}
    obstacle_fp = set().union(*[placed[nm] for nm in onames]) if onames else set()
    obstacle_cells = [patch_at[nm] for nm in onames]

    blocked = set(occupied)
    forbid_fp = obstacle_fp
    A = [a for a, b in patch_at.values()]
    B = [b for a, b in patch_at.values()]
    corridor = set()
    for a in range(min(A) - pad, max(A) + pad + 1):
        for b in range(min(B) - pad, max(B) + pad + 1):
            if (a, b) in blocked:
                continue                                  # a patch sits here (targets, or all patches)
            if cell(a, b, d, seam) & forbid_fp:
                continue                                  # d-wide footprint would hit an obstacle
            if any(_cheb((a, b), oc) <= keepout for oc in obstacle_cells):
                continue                                  # keep-out margin: no shared boundary ancilla
            corridor.add((a, b))
    G = nx.Graph()
    G.add_nodes_from(corridor)
    for (a, b) in corridor:
        for (da, db) in NEIGH:
            if (a + da, b + db) in corridor:
                G.add_edge((a, b), (a + da, b + db))
    return G, corridor, placed, occupied, obstacle_fp, onames


def _faces_in(patch_at, corridor, nm, dirs):
    """The corridor cells adjacent to patch ``nm`` on the given face directions."""
    a, b = patch_at[nm]
    return [(a + da, b + db) for (da, db) in dirs if (a + da, b + db) in corridor]


def _candidate_arms(G, patch_at, corridor, root, root_faces, z, per_z):
    """Up to ``per_z`` shortest obstacle-aware paths from an X-face of ``root`` to a face of ``z``."""
    paths = []
    for s in _faces_in(patch_at, corridor, root, root_faces):
        for e in _faces_in(patch_at, corridor, z, NEIGH):
            if s in G and e in G and nx.has_path(G, s, e):
                paths.append(nx.shortest_path(G, s, e))
    paths.sort(key=len)
    seen, keep = set(), []
    for p in paths:
        if tuple(p) not in seen:
            seen.add(tuple(p))
            keep.append(p)
        if len(keep) >= per_z:
            break
    return keep


def _cells_connected(cells):
    """True iff the coarse cells form ONE 4-neighbour connected component (empty = connected).

    A union of arms that meets only *through* a target patch is **not** connected: the ancilla
    bus must be a single contiguous region on its own, so such a corridor is an illegal route
    (it would be two separate buses), not a candidate for the physics layer.
    """
    cells = set(map(tuple, cells))
    if not cells:
        return True
    start = next(iter(cells))
    seen, stack = {start}, [start]
    while stack:
        a, b = stack.pop()
        for n in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)):
            if n in cells and n not in seen:
                seen.add(n)
                stack.append(n)
    return len(seen) == len(cells)


def emv_group_steiner(G, groups, banned=frozenset(), extra_cost=None):
    """EXACT minimum node-weighted group Steiner corridor — the
    Erickson–Monma–Veinott send-and-split DP on cells (the user's
    corridor-routing-model.md, w ≡ 1):

        base   f[{t}][v] = multi-source Dijkstra from the WHOLE legal
                           attach set g_t                      (seed)
        merge  f[S][v] ← f[S'][v] + f[S\\S'][v] − w(v)          (merge)
        grow   f[S][y] ← f[S][x] + w(y)                        (walk)

    Node-weight corrections are load-bearing: merge subtracts the shared
    cell once; grow charges the ENTERED cell (textbook edge-weight DW is
    wrong here).  Groups need NO auxiliary-vertex reduction (an aux vertex
    acts as a wormhole through obstacles) — they only widen the base's
    fire sources.  Cost = number of corridor cells; ``extra_cost`` (cell →
    surcharge) raises single cells' node weight — the hook-safety penalty
    rides on this.  Returns ``(cost, cell_set)`` or ``None``."""
    import heapq
    extra = extra_cost or {}
    nodes = [v for v in G.nodes if v not in banned]
    nodeset = set(nodes)
    groups = [[c for c in g if c in nodeset] for g in groups]
    if not nodes or not groups or any(not g for g in groups):
        return None
    if len(groups) == 2 and not extra:
        # two groups = plain multi-source multi-target shortest path (unit
        # node weights): BFS, exact, milliseconds.  The full send-and-split
        # DP is pure Python and pays seconds per call on big boards — pair
        # steps are the most common step shape, and bv_n140's planner
        # probes spent their whole 20-min timeout here (2026-08-05).
        ga, gb = set(groups[0]), set(groups[1])
        inter = ga & gb
        if inter:
            c = min(inter)
            return (1, {c})
        prev = {c: None for c in ga}
        frontier = sorted(ga)
        hit = None
        while frontier and hit is None:
            nxt = []
            for v in frontier:
                for u in sorted(G.neighbors(v)):
                    if u in prev or u not in nodeset:
                        continue
                    prev[u] = v
                    if u in gb:
                        hit = u
                        break
                    nxt.append(u)
                if hit is not None:
                    break
            frontier = nxt
        if hit is None:
            return None
        cells = set()
        while hit is not None:
            cells.add(hit)
            hit = prev[hit]
        return (len(cells), cells)
    # hot loop: precomputed node weights + adjacency lists (this DP runs
    # hundreds of times per probe via the Lawler branching — the profiled
    # cost was interpreter dispatch: a per-touch weight lambda and
    # G.neighbors per relaxation, 1.1e9 / 1.4e8 calls on the 30-patch run)
    wv = {v: 1 + extra.get(v, 0) for v in nodes}
    adj = {v: tuple(u for u in G.neighbors(v) if u in nodeset)
           for v in nodes}
    k = len(groups)
    FULL = (1 << k) - 1
    INF = float('inf')
    f = {v: [INF] * (FULL + 1) for v in nodes}
    par = {}
    for S in range(1, FULL + 1):
        if S & (S - 1) == 0:                       # base: seed
            gi = S.bit_length() - 1
            for c in groups[gi]:
                wc = wv[c]
                if f[c][S] > wc:
                    f[c][S] = wc                   # the attach cell itself
                    par[(c, S)] = ('fire', gi)
        else:                                       # merge: merge (minus w(v))
            for v in nodes:
                fv = f[v]
                wvv = wv[v]
                sub = (S - 1) & S
                while sub:
                    rest = S ^ sub
                    if sub <= rest:
                        cand = fv[sub] + fv[rest] - wvv
                        if cand < fv[S]:
                            fv[S] = cand
                            par[(v, S)] = ('split', sub)
                    sub = (sub - 1) & S
        heap = [(fv[S], v) for v, fv in f.items() if fv[S] < INF]
        heapq.heapify(heap)                         # grow: walk (plus w(y))
        push = heapq.heappush
        pop = heapq.heappop
        while heap:
            dv, v = pop(heap)
            if dv > f[v][S]:
                continue
            for u in adj[v]:
                fu = f[u]
                nd = dv + wv[u]
                if nd < fu[S]:
                    fu[S] = nd
                    par[(u, S)] = ('walk', v)
                    push(heap, (nd, u))
    best_v = min(nodes, key=lambda v: f[v][FULL])
    if f[best_v][FULL] == INF:
        return None
    tree, stack = set(), [(best_v, FULL)]
    while stack:
        v, S = stack.pop()
        tree.add(v)
        kind = par.get((v, S))
        if kind is None or kind[0] == 'fire':
            continue
        if kind[0] == 'walk':
            stack.append((kind[1], S))
        else:
            stack.append((v, kind[1]))
            stack.append((v, S ^ kind[1]))
    return f[best_v][FULL], tree


def _single_width(G, cells):
    """The corridor must be SINGLE-WIDTH: its induced subgraph in the cell
    grid is a tree (no 2x2 blobs, no loops) — |induced edges| = |cells| − 1.
    A vertex-set property, not a tree property, so it is checked on the
    solution (per the design doc: enumerate by increasing cost, take the
    first that passes)."""
    cells = set(cells)
    edges = sum(1 for v in cells for u in G.neighbors(v) if u in cells) // 2
    return edges == len(cells) - 1


def emv_corridor_candidates(G, groups, limit=24, extra_cost=None):
    """Solutions of :func:`emv_group_steiner` in increasing cost, Lawler
    style: pop the cheapest, then branch by banning each of its cells.
    Feeds the propose-and-verify loop (single-width and the physics oracle
    both reject by taking the next candidate)."""
    import heapq
    if len(groups) > _EMV_GROUP_CAP:
        # the EMV DP is O(3^k) in the group count — measured 2026-08-05:
        # k=9 ~13 s, k=10 ~60 s, k=17 hours (bv_32's whole timeout lived
        # here).  Above the cap this generator yields nothing and the
        # merged candidate stream runs on the bounded arm/Steiner pool
        # alone: per-tree construction verification is unchanged, only the
        # exact-optimality claim narrows to small steps.
        return
    first = emv_group_steiner(G, groups, extra_cost=extra_cost)
    if first is None:
        return
    seen = {frozenset(first[1])}
    heap = [(first[0], 0, first[1], frozenset())]
    count = 0
    serial = 1
    while heap and count < limit:
        cost, _, tree, banned = heapq.heappop(heap)
        yield tree
        count += 1
        for c in tree:
            nb = banned | {c}
            sol = emv_group_steiner(G, groups, banned=nb, extra_cost=extra_cost)
            if sol is None:
                continue
            key = frozenset(sol[1])
            if key in seen:
                continue
            seen.add(key)
            heapq.heappush(heap, (sol[0], serial, sol[1], nb))
            serial += 1


def _greedy_connect(G, groups, seq):
    """Grow a tree over ``groups`` in the FIXED order ``seq``: each group
    attaches via the shortest path from the whole current tree
    (multi-source BFS).  Deterministic; returns None when some group is
    unreachable."""
    tree = {min(groups[seq[0]])}
    for gi in seq[1:]:
        targets = set(groups[gi])
        if tree & targets:
            continue
        prev = {c: None for c in tree}
        frontier = sorted(tree)
        hit = None
        while frontier and hit is None:
            nxt = []
            for v in frontier:
                for u in sorted(G.neighbors(v)):
                    if u in prev:
                        continue
                    prev[u] = v
                    if u in targets:
                        hit = u
                        break
                    nxt.append(u)
                if hit is not None:
                    break
            frontier = nxt
        if hit is None:
            return None
        while hit is not None:
            tree.add(hit)
            hit = prev[hit]
    return tree


def _nearest_first_tree(G, groups):
    """Prim-style candidate: seed at the most-constrained group, then
    repeat a single multi-source BFS from the whole tree; the first
    not-yet-connected group it reaches is the nearest, and the BFS path
    attaches it.  Deterministic ties: sorted frontier/neighbours, then
    group index.  Classic 2-approximate Steiner heuristic; measured
    2026-08-08 on 8 real k>9 instances: loses to the fixed orders for
    k<=33 (+10..24% cells) and wins at k=64 (-5.6%) — hence the pool
    takes the min over all five (design decision: best-of-5)."""
    seed = min(range(len(groups)), key=lambda i: (len(groups[i]),
                                                  min(groups[i])))
    tree = {min(groups[seed])}
    left = set(range(len(groups))) - {seed}
    cell_of = {}
    for i in sorted(left):
        for c in groups[i]:
            cell_of.setdefault(c, i)
    while left:
        hit_g = None
        for i in sorted(left):
            if tree & set(groups[i]):
                hit_g, hit = i, None
                break
        if hit_g is None:
            prev = {c: None for c in tree}
            frontier = sorted(tree)
            hit = None
            while frontier and hit is None:
                nxt = []
                for v in frontier:
                    for u in sorted(G.neighbors(v)):
                        if u in prev:
                            continue
                        prev[u] = v
                        gi = cell_of.get(u)
                        if gi is not None and gi in left:
                            hit, hit_g = u, gi
                            break
                        nxt.append(u)
                    if hit is not None:
                        break
                frontier = nxt
            if hit is None:
                return None
            while hit is not None:
                tree.add(hit)
                hit = prev[hit]
        left.discard(hit_g)
    return tree


def _legal_greedy_trees(G, groups):
    """Polynomial legal-faces-aware Steiner candidates for steps ABOVE the
    EMV group cap: grow a tree that touches one PARALLEL-LAW legal cell of
    every group (greedy nearest-target connection via multi-source BFS),
    in a few deterministic orders.  Complements the orientation-agnostic
    arm/Steiner pool, whose candidates attach on arbitrary faces and are
    rejected wholesale by the stitch policy on high-weight steps (measured
    2026-08-05: bv_32/dj_16 'no parallel-law-legal attach seam')."""
    def connect(seq):
        return _greedy_connect(G, groups, seq)

    idx = list(range(len(groups)))
    orders = [idx, idx[::-1],
              sorted(idx, key=lambda i: (len(groups[i]), min(groups[i]))),
              sorted(idx, key=lambda i: min(groups[i]))]
    out, seen = [], set()
    for seq in orders:
        t = connect(seq)
        if t is not None and frozenset(t) not in seen:
            seen.add(frozenset(t))
            out.append(t)
    # fifth candidate (design decision 2026-08-08: best-of-5): the Prim-style
    # nearest-first tree — neither side dominates (fixed orders win k<=33,
    # nearest-first wins k=64), the pool takes the min
    nf = _nearest_first_tree(G, groups)
    if nf is not None and frozenset(nf) not in seen:
        seen.add(frozenset(nf))
        out.append(nf)
    return out


_ARM_POOL_CAP = 4096
_EMV_GROUP_CAP = 9


def _bounded_product(per, cap):
    """First ``cap`` combos of the per-target arm cross-product in
    nondecreasing sum-of-arm-lengths order (best-first over index vectors).

    The full product is exponential in the target count — measured
    2026-08-04 (bv_32, w=17): ~150e9 combos materialized at
    ``list(itertools.product(...))``, 87%% of wall time inside gc_collect
    before dying.  The pool is only supplementary diversity for the seam
    pipeline (EMV is the primary, exact, lazy source), so a bounded
    cost-ordered prefix keeps the useful low-cost candidates and drops
    the combinatorial tail."""
    import heapq
    per = [sorted(arms, key=lambda a: (len(a), tuple(sorted(a))))
           for arms in per]
    start = (0,) * len(per)
    heap = [(sum(len(p[0]) for p in per), start)]
    seen = {start}
    out = []
    while heap and len(out) < cap:
        cost, idx = heapq.heappop(heap)
        out.append(tuple(p[i] for p, i in zip(per, idx)))
        for pos in range(len(per)):
            if idx[pos] + 1 < len(per[pos]):
                nxt = idx[:pos] + (idx[pos] + 1,) + idx[pos + 1:]
                if nxt not in seen:
                    seen.add(nxt)
                    heapq.heappush(
                        heap,
                        (cost - len(per[pos][idx[pos]])
                         + len(per[pos][idx[pos] + 1]), nxt))
    return out


def _steiner_trees(G, patch_at, corridor, root, zs):
    """Approximately-minimal **connected** corridors touching a face of the root and of every
    ``z`` — the "fewest corridor cells" candidates the per-arm product enumeration misses.

    Greedy Steiner approximation: seed with the shortest root-face → first-target path, then
    attach each next target via its shortest path from the *current tree* (multi-source
    Dijkstra), so arms share cells instead of duplicating them.  Several target orders are
    tried (near-first, far-first, given); connected-by-construction, smallest first.
    """
    if not zs:
        return []
    root_faces = [c for c in _faces_in(patch_at, corridor, root, NEIGH) if c in G]
    zfaces = {z: [c for c in _faces_in(patch_at, corridor, z, NEIGH) if c in G] for z in zs}
    if not root_faces or any(not v for v in zfaces.values()):
        return []
    rc = patch_at[root]
    orders = [sorted(zs, key=lambda z: _cheb(patch_at[z], rc)),
              sorted(zs, key=lambda z: -_cheb(patch_at[z], rc)),
              list(zs)]
    trees, seen = [], set()
    for order in orders:
        tree = set()
        ok = True
        for z in order:
            sources = tree if tree else set(root_faces)
            try:
                lengths, paths = nx.multi_source_dijkstra(G, sources)
            except ValueError:
                ok = False
                break
            ends = [f for f in zfaces[z] if f in lengths]
            if not ends:
                ok = False
                break
            end = min(ends, key=lambda f: lengths[f])
            tree |= set(paths[end])
        if ok and tree:
            key = frozenset(tree)
            if key not in seen:
                seen.add(key)
                trees.append(tree)
    trees.sort(key=len)
    return trees


