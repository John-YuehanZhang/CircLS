"""Lifetime-aware step scheduling.

Reorders the swept op stream (PauliOp 's'/'m', pre-expand_gadgets) so that
each patch's first and last JOINT-step uses sit close together: patch
lifetime — birth at first use (first_use_init) to retirement at last use
(liveness) — is what the spacetime volume bills, and the executed order is
the knob that sets it.  The classical-compiler analogue is instruction
scheduling for register-pressure reduction.

Correctness never rests on assumptions: an order constraint is added for
every pair of ops whose Pauli products ANTICOMMUTE (measured with the
symplectic form, never inferred), and only step-inducing ops are permuted —
within the slot positions they already occupy, so weight-1 folded readouts
and every other fixed op keep their absolute positions (the M1 tail
invariant survives untouched).  Worst case (a fully ordered chain) returns
the input order verbatim: the pass can only tie or win.

Objective: sum over data patches of (last step position - first step
position), evaluated in closed form.  The general problem is NP-hard
(minimum storage-time sequencing family), so: deterministic portfolio
(original / closes-first greedy / hub-median anchor) + first-improvement
local search (adjacent swaps and single reinsertions that respect the
constraint DAG and fixed-op sides), deterministic tie-breaks throughout.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from circls.interop.ir.pauli_ir import PauliCircuit, PauliOp


def _anticommutes(a: PauliOp, b: PauliOp) -> bool:
    odd = 0
    for q, pa in a.paulis.items():
        pb = b.paulis.get(q)
        if pb is not None and pa != pb:
            odd ^= 1
    return bool(odd)


def _is_step(op: PauliOp) -> bool:
    """Ops that become JOINT steps downstream: every 's' (gadget = joint
    PPM with its |Y> ancilla) and any 'm' touching >= 2 qubits."""
    return op.kind == "s" or len(op.paulis) >= 2


def _lifetime_sum(order: Sequence[int], ops: List[PauliOp]) -> int:
    """Sum of per-patch spans over JOINT-step positions only: a weight-1
    folded readout never pins a lifetime (the patch retires at its last
    joint use and the folded bit is captured by the retirement readout),
    so counting fixed ops would just add a constant floor that drowns the
    real signal."""
    first: Dict[int, int] = {}
    last: Dict[int, int] = {}
    for pos, idx in enumerate(order):
        if idx is None or not _is_step(ops[idx]):
            continue
        for q in ops[idx].paulis:
            first.setdefault(q, pos)
            last[q] = pos
    return sum(last[q] - first[q] for q in first)


def _valid(order: Sequence[int], pred: Dict[int, set]) -> bool:
    pos = {idx: p for p, idx in enumerate(order)}
    return all(pos[a] < pos[i] for i, ps in pred.items() for a in ps
               if a in pos and i in pos)


def contiguous_batches(qsets: List[frozenset]) -> List[List[int]]:
    """Greedy contiguous batching of an EXECUTION order: position p joins
    the current batch iff its qubit set is disjoint from every member's.
    Disjoint supports always commute, so a batch is a legal simultaneous
    merge window; the rule is deliberately contiguous — no step ever jumps
    over a conflicting one, so index-based lifetime bookkeeping stays
    valid.  Worst case every batch has one member: today's serial
    behaviour."""
    batches: List[List[int]] = []
    cur: List[int] = []
    cur_qs: set = set()
    for p, qs in enumerate(qsets):
        if cur and (cur_qs & qs):
            batches.append(cur)
            cur, cur_qs = [], set()
        cur.append(p)
        cur_qs |= qs
    if cur:
        batches.append(cur)
    return batches


def batch_span_cost(order: Sequence[int], ops: List[PauliOp]) -> Tuple[int, int, int]:
    """Batch-aware objective, lexicographic: (NUMBER of batches, sum of
    per-patch batch spans, sum of per-patch step spans).  Wall-clock is
    batches x d rounds, so batch COUNT comes first — the original
    span-only version happily built many narrow batches, minimizing
    lifetimes while wrecking the schedule length (measured on the
    ghz-mixed chain: 6 batches chosen over the 3-batch interleave)."""
    qsets = [frozenset(ops[i].paulis) for i in order]
    batches = contiguous_batches(qsets)
    batch_of = {}
    for b, members in enumerate(batches):
        for p in members:
            batch_of[p] = b
    first_b: dict = {}
    last_b: dict = {}
    first_s: dict = {}
    last_s: dict = {}
    for p, qs in enumerate(qsets):
        for q in qs:
            first_b.setdefault(q, batch_of[p])
            last_b[q] = batch_of[p]
            first_s.setdefault(q, p)
            last_s[q] = p
    return (len(batches),
            sum(last_b[q] - first_b[q] for q in first_b),
            sum(last_s[q] - first_s[q] for q in first_s))


def verify_schedule(pre: PauliCircuit, post: PauliCircuit,
                    perm) -> None:
    """Contract check for an externally supplied schedule
    (docs/API_HOOKS.md): perm[new_pos] = old_pos must be a permutation
    reproducing ``post`` from ``pre``; fixed (non-step) ops keep their
    absolute positions; anticommuting step-step pairs keep their
    original relative order; every step op stays on its original side
    of any anticommuting fixed op."""
    ops = pre.ops
    n = len(ops)
    if sorted(perm) != list(range(n)):
        raise ValueError("schedule perm is not a permutation")
    if len(post.ops) != n or any(post.ops[i] != ops[perm[i]]
                                 for i in range(n)):
        raise ValueError(
            "scheduled circuit does not match perm applied to input")
    pos_of = [0] * n
    for new, old in enumerate(perm):
        pos_of[old] = new
    for i in range(n):
        if not _is_step(ops[i]) and pos_of[i] != i:
            raise ValueError(f"fixed op at {i} moved to {pos_of[i]}")
    step_idx = [i for i in range(n) if _is_step(ops[i])]
    for x in range(len(step_idx)):
        for y in range(x + 1, len(step_idx)):
            a, b = step_idx[x], step_idx[y]
            if _anticommutes(ops[a], ops[b]) and pos_of[a] > pos_of[b]:
                raise ValueError(
                    f"anticommuting step pair ({a},{b}) swapped")
    for f in range(n):
        if _is_step(ops[f]):
            continue
        for i in step_idx:
            if _anticommutes(ops[f], ops[i]) and                     (i < f) != (pos_of[i] < f):
                raise ValueError(
                    f"step op {i} crossed anticommuting fixed op {f}")


def schedule_ops(circuit: PauliCircuit,
                 parallel: bool = False) -> Tuple[PauliCircuit, List[int]]:
    """Return (reordered circuit, permutation) with perm[new_pos] = old_pos.

    Fixed (non-step) ops keep their absolute positions; step ops are
    permuted within the step slots, subject to (a) anticommuting step-step
    pairs keeping their original relative order and (b) every step op
    staying on its original side of any anticommuting FIXED op.
    """
    ops = circuit.ops
    n = len(ops)
    step_idx = [i for i in range(n) if _is_step(ops[i])]
    if len(step_idx) <= 2:
        return circuit, list(range(n))
    fixed_idx = [i for i in range(n) if not _is_step(ops[i])]

    # (b) side constraints against fixed ops -> an allowed SLOT RANGE per
    # step op (slots are the original positions of step ops, ascending)
    slots = step_idx
    lo = {i: 0 for i in step_idx}
    hi = {i: len(slots) - 1 for i in step_idx}
    for f in fixed_idx:
        for i in step_idx:
            if _anticommutes(ops[f], ops[i]):
                if i < f:      # must stay in a slot before position f
                    k = max(k for k, s in enumerate(slots) if s < f)
                    hi[i] = min(hi[i], k)
                else:          # must stay in a slot after position f
                    k = min(k for k, s in enumerate(slots) if s > f)
                    lo[i] = max(lo[i], k)

    # (a) precedence DAG among step ops (original order for anticommuters)
    pred: Dict[int, set] = {i: set() for i in step_idx}
    for a_pos in range(len(step_idx)):
        for b_pos in range(a_pos + 1, len(step_idx)):
            a, b = step_idx[a_pos], step_idx[b_pos]
            if _anticommutes(ops[a], ops[b]):
                pred[b].add(a)
    # DECODER-FAITHFULNESS caveat (measured 2026-08-05): on overlapping
    # relay-chain structures (teleport family — the ONLY offender in a
    # 17-case audit) the rescheduled merge order creates high-symptom
    # hyperedge error mechanisms whose forced decomposition miscorrects
    # under matching (88/39191 singles; 0 without scheduling), an LER
    # floor of ~70x the physical rate with graphlike AND hypergraph
    # distance still reading full.  The distinction is geometric, not
    # algebraic (teleport's terminal set is fully commuting like everyone
    # else's), so there is no local rule to apply here — enforcement is
    # empirical: experiments/ler.py runs the single-error faithfulness
    # check on every point and falls back to the original order when it
    # fails; benchmark configs pin the audited offenders off.

    def slot_ok(order: Sequence[int]) -> bool:
        return all(lo[i] <= k <= hi[i] for k, i in enumerate(order))

    def cost(order: Sequence[int]):
        if parallel:
            # batch-aware: patches are billed per batch window (see
            # batch_span_cost); fixed weight-1 ops never form batches so
            # the step-op order alone decides
            return batch_span_cost(order, ops)
        # positions: step ops at their assigned slots, fixed ops in place
        full = [None] * n
        for k, i in enumerate(order):
            full[slots[k]] = i
        for f in fixed_idx:
            full[f] = f
        return _lifetime_sum(full, ops)

    candidates: List[List[int]] = [list(step_idx)]          # original

    # closes-first greedy: prefer the ready op that closes the most open
    # lifetimes and opens the fewest new ones
    remaining_uses: Dict[int, int] = {}
    for i in step_idx:
        for q in ops[i].paulis:
            remaining_uses[q] = remaining_uses.get(q, 0) + 1
    uses = dict(remaining_uses)
    indeg = {i: len(pred[i]) for i in step_idx}
    ready = sorted(i for i in step_idx if indeg[i] == 0)
    opened: set = set()
    greedy: List[int] = []
    succ: Dict[int, list] = {i: [] for i in step_idx}
    for i, ps in pred.items():
        for a in ps:
            succ[a].append(i)
    while ready:
        k = len(greedy)
        best = None
        for i in ready:
            if not (lo[i] <= k <= hi[i]):
                continue
            closes = sum(1 for q in ops[i].paulis
                         if q in opened and uses[q] == 1)
            opens = sum(1 for q in ops[i].paulis if q not in opened)
            key = (-closes, opens, i)
            if best is None or key < best[0]:
                best = (key, i)
        if best is None:       # slot-range deadlock: abandon this heuristic
            greedy = None
            break
        i = best[1]
        ready.remove(i)
        greedy.append(i)
        for q in ops[i].paulis:
            uses[q] -= 1
            opened.add(q)
        for s in succ[i]:
            indeg[s] -= 1
            if indeg[s] == 0:
                ready.append(s)
        ready.sort()
    if greedy is not None and len(greedy) == len(step_idx) \
            and slot_ok(greedy) and _valid(greedy, pred):
        candidates.append(greedy)

    # hub-median anchor: move each high-degree op to the median slot of the
    # OTHER steps touching its patches (star-shape optimum; general nudge)
    base = list(step_idx)
    by_deg = sorted(base, key=lambda i: -len(ops[i].paulis))
    anchored = list(base)
    for hub in by_deg[:3]:
        others = [k for k, i in enumerate(anchored)
                  if i != hub and any(q in ops[hub].paulis
                                     for q in ops[i].paulis)]
        if not others:
            continue
        med = others[len(others) // 2]
        cur = anchored.index(hub)
        trial = list(anchored)
        trial.pop(cur)
        trial.insert(med, hub)
        if slot_ok(trial) and _valid(trial, pred):
            anchored = trial
    candidates.append(anchored)

    if parallel:
        # graph-colouring order: first-fit classes by qubit-disjointness,
        # classes concatenated — the chain even/odd interleave falls out
        classes: List[List[int]] = []
        class_qs: List[set] = []
        for i in step_idx:
            qs = set(ops[i].paulis)
            for c, cqs in enumerate(class_qs):
                if not (cqs & qs):
                    classes[c].append(i)
                    cqs |= qs
                    break
            else:
                classes.append([i])
                class_qs.append(set(qs))
        coloured = [i for cls in classes for i in cls]
        if slot_ok(coloured) and _valid(coloured, pred):
            candidates.append(coloured)
        # precedence-aware first-fit layering (ASAP list schedule): a step
        # joins the earliest class AFTER all of its predecessors' classes
        # whose qubit set it is disjoint from.  Where the plain colouring
        # is invalid (an op first-fits into a class before its
        # predecessor's), this still yields the register-wise interleave of
        # independent sub-programs written one after another -- measured
        # 2026-09-06 on three 4-gate routines listed register by register:
        # 24 singleton batches in program order, 8 batches of width 3 here;
        # the local search below never reaches it from the original order
        # (single moves cannot lower the batch count past the first swap).
        layers: List[List[int]] = []
        layer_qs: List[set] = []
        layer_of: Dict[int, int] = {}
        for i in step_idx:
            qs = set(ops[i].paulis)
            start = max((layer_of[a] + 1 for a in pred[i]), default=0)
            for c in range(start, len(layers)):
                if not (layer_qs[c] & qs):
                    layers[c].append(i)
                    layer_qs[c] |= qs
                    layer_of[i] = c
                    break
            else:
                layers.append([i])
                layer_qs.append(set(qs))
                layer_of[i] = len(layers) - 1
        layered = [i for cls in layers for i in cls]
        if slot_ok(layered) and _valid(layered, pred):
            candidates.append(layered)

    best = min(candidates, key=lambda o: (cost(o), o != candidates[0]))

    # local search: adjacent swaps + single reinsertion, first-improvement
    improved = True
    sweeps = 0
    while improved and sweeps < 40:
        improved = False
        sweeps += 1
        c0 = cost(best)
        for k in range(len(best) - 1):
            trial = list(best)
            trial[k], trial[k + 1] = trial[k + 1], trial[k]
            if slot_ok(trial) and _valid(trial, pred) and cost(trial) < c0:
                best = trial
                improved = True
                break
        if improved:
            continue
        for k in range(len(best)):
            for k2 in range(len(best)):
                if k == k2:
                    continue
                trial = list(best)
                i = trial.pop(k)
                trial.insert(k2, i)
                if slot_ok(trial) and _valid(trial, pred) \
                        and cost(trial) < c0:
                    best = trial
                    improved = True
                    break
            if improved:
                break

    perm = [None] * n
    for k, i in enumerate(best):
        perm[slots[k]] = i
    for f in fixed_idx:
        perm[f] = f
    new_ops = [ops[i] for i in perm]
    return PauliCircuit(circuit.num_qubits, new_ops), perm
