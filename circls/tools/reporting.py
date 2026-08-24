"""Per-shot assembly of the compiled program's ORIGINAL output bits.

The user-facing missing piece (2026-08-05): run the compiled circuit,
recover every original terminal-measurement bit — deterministic AND
random — from the raw measurement records, through every bookkeeping
layer: swept-op sign flips, |Y>-gadget corrections (c = b_joint XOR b_x
XOR kappa), weight-1 terminal folding, and the measurement-reduction
reconstruction (executed bits -> original bits by XOR + const).

Bits are returned in LOAD ORDER (original m op order = qubit order), the
canonical space shared by reduction-on and reduction-off compilations —
which is exactly what makes the equivalence verifier possible.
"""
from __future__ import annotations

from functools import reduce
from typing import Dict, List, Optional, Tuple

import numpy as np
import stim

from circls.interop.ir.gosc_gadgets import folded_out_bits, step_roles
from circls.interop.ir.ppm_import import patch_name

_LETTER = {"X": 1, "Y": 2, "Z": 3}


def _support_recs(exp, nm: str, letter: str) -> List[int]:
    support = [dq for rec in exp.system.logical_ops
               if rec.get('patch_name') == nm and rec.get('type') == letter
               for dq in rec['pauli']]
    return [exp.final_readout_recs[dq] for dq in support]


def _bit_targets(prog) -> List[Dict[int, str]]:
    """out-bit index -> the executed mpp's {program_qubit: letter} (the
    same walk that numbers bits in folded_out_bits/step_roles)."""
    anc = {g.ancilla for g in prog.gadgets}
    out = []
    for op in prog.ops:
        if op.kind != "mpp":
            continue
        if len(op.targets) == 1:
            (q, p), = op.targets.items()
            if p == "X" and q in anc:
                continue
        if any(q in anc for q in op.targets):
            continue
        out.append(dict(op.targets))
    return out


def _pstring(n: int, targets: Dict[int, str]) -> stim.PauliString:
    ps = stim.PauliString(n)
    for q, letter in targets.items():
        ps[q] = _LETTER[letter]
    return ps


def _mask(ps: stim.PauliString) -> int:
    n = len(ps)
    m = 0
    for q in range(n):
        v = ps[q]
        if v in (1, 2):
            m |= 1 << q
        if v in (2, 3):
            m |= 1 << (n + q)
    return m


def _solve_ideal(tgt: stim.PauliString, gens) -> Optional[Tuple[set, int]]:
    """Express tgt as a product of generators; return (records, sign_flip)
    or None.  gens: list of (mask, PauliString, records:frozenset, is_seed).
    Value identity: seeds pairwise commute (one letter per patch), anchors
    pairwise commute (terminal m-set), and the seed REMAINDER of any
    solution commutes with every anchor as a whole — so the product taken
    anchors-then-seeds has a well-defined real sign, and
    v(tgt) = XOR of generator values XOR [sign == -1]."""
    if not gens:
        return None
    basis = []                                  # (mask, combo bitset)
    for j, (m0, *_rest) in enumerate(gens):
        m, c = m0, 1 << j
        for bm, bc in basis:
            if (m >> (bm.bit_length() - 1)) & 1:
                m ^= bm
                c ^= bc
        if m:
            basis.append((m, c))
            basis.sort(key=lambda t: -t[0].bit_length())
    t, combo = _mask(tgt), 0
    for bm, bc in basis:
        if t and (t >> (bm.bit_length() - 1)) & 1:
            t ^= bm
            combo ^= bc
    if t:
        return None
    chosen = [gens[j] for j in range(len(gens)) if (combo >> j) & 1]
    n = len(tgt)
    prod = stim.PauliString(n)
    for grp in (False, True):                   # anchors first, then seeds
        for _, ps, _, is_seed in chosen:
            if is_seed == grp:
                prod = prod * ps
    check = prod * tgt                          # equal unsigned parts square
    assert check.weight == 0 and check.sign in (1, -1), \
        f"ideal-solve sign not real: {check.sign}"
    recs = reduce(lambda a, b: a ^ b, (g[2] for g in chosen), frozenset())
    return set(recs), int(check.sign == -1)


def executed_bit_recs(cp) -> List[Tuple[List[int], int, List[int]]]:
    """Per program out-bit (in program order): (record indices, flip,
    gadget indices whose correction XORs in).

    Joint-step bits are reconstructed at their IDEAL value where possible:
    a bit whose operator factors over seed-pinned logicals (plus earlier
    raw-anchored bits) gets the frame-free seed-closure recipe — a window
    executed earlier (scheduling/parallelism) can no longer contaminate it
    with its split byproduct.  Bits outside that span keep the raw
    post-closure and become anchors themselves; gadget-corrected and
    folded bits keep their legacy (actual-branch) paths."""
    prog, exp = cp.program, cp.experiment
    roles = step_roles(prog)
    folded = folded_out_bits(prog)
    m_step_of = {bit: j for j, (kind, bit) in enumerate(roles)
                 if kind == "m"}
    targets = _bit_targets(prog)
    n = prog.num_data
    qubit_of = {patch_name(q): q for q in range(n)}
    gens = []                                   # seeds, then anchors as met
    for (nm, letter), recs in sorted(getattr(exp, "seed_closure", {}).items()):
        q = qubit_of.get(nm)
        if q is None:
            continue                            # y-ancilla patch: gadget flow
        ps = _pstring(n, {q: letter})
        gens.append((_mask(ps), ps, frozenset(recs), True))
    out = []
    for i, ob in enumerate(prog.out_bits):
        if i in folded:
            q, letter = folded[i]
            out.append((_support_recs(exp, patch_name(q), letter),
                        ob.flip, list(ob.gadgets)))
            continue
        post = exp.step_joint_records_post.get(m_step_of[i])
        if not ob.gadgets:
            tgt = _pstring(n, targets[i])
            sol = _solve_ideal(tgt, gens)
            if sol is not None:
                recs, sflip = sol
                out.append((sorted(recs), ob.flip ^ sflip, []))
                continue
            if post is not None:
                gens.append((_mask(tgt), tgt, frozenset(post), False))
        if post is None:
            raise RuntimeError(f"out bit {i}: no post records captured")
        out.append((sorted(post), ob.flip, list(ob.gadgets)))
    return out


def _gadget_cond_recs(cp):
    """Per gadget: (joint-outcome records, ancilla-X records, kappa,
    earlier-gadget chain ``conds``)."""
    prog, exp = cp.program, cp.experiment
    roles = step_roles(prog)
    g_step_of = {k: j for j, (kind, k) in enumerate(roles)
                 if kind == "gadget"}
    out = []
    for k, g in enumerate(prog.gadgets):
        post = exp.step_joint_records_post.get(g_step_of[k])
        if post is None:
            raise RuntimeError(f"gadget {k}: no post records captured")
        anc_nm = f"y{g.ancilla - prog.num_data}"
        out.append((sorted(post), _support_recs(exp, anc_nm, "X"), g.kappa,
                    list(getattr(g, "conds", []))))
    return out


class _RecTracker:
    """Record-set symbolic stabilizer tracker (t_as_s reconstruction).

    Generators carry (unsigned PauliString, record set, const): the
    operator's per-shot value is XOR(records) ^ const.  Measuring an op
    that commutes with the group DERIVES its value from the generator
    product (sign checked real); an anticommuting op is ANCHORED to its
    actual record closure and the group updates in the standard way.
    Because every derived value routes through the shared anchors, split
    byproducts cancel — the failure mode of the raw-post path once
    gadget masks stop being pure-Z (mixed merges put random X-frame
    byproducts into the closures; measured: t;t on one qubit reads a
    random bit where the dense oracle says deterministic)."""

    def __init__(self):
        self.gens = []            # [ps(+), recs frozenset, const int]

    @staticmethod
    def _mask_ps(ps):
        n = len(ps)
        m = 0
        for q in range(n):
            v = ps[q]
            if v in (1, 2):
                m |= 1 << q
            if v in (2, 3):
                m |= 1 << (n + q)
        return m

    def add_seed(self, ps, recs, const=0):
        self.gens.append([ps, frozenset(recs), const])

    def measure(self, ps, actual_recs):
        anti = [g for g in self.gens if not g[0].commutes(ps)]
        if not anti:
            basis = []                       # (mask, combo)
            for j, g in enumerate(self.gens):
                m, c = self._mask_ps(g[0]), 1 << j
                for bm, bc in basis:
                    if (m >> (bm.bit_length() - 1)) & 1:
                        m ^= bm
                        c ^= bc
                if m:
                    basis.append((m, c))
                    basis.sort(key=lambda t: -t[0].bit_length())
            t, combo = self._mask_ps(ps), 0
            for bm, bc in basis:
                if t and (t >> (bm.bit_length() - 1)) & 1:
                    t ^= bm
                    combo ^= bc
            if t:
                raise RuntimeError(
                    "commuting measurement not in the tracker group — a "
                    "patch without a seed closure reached a derived bit")
            prod = stim.PauliString(len(ps))
            recs, const = frozenset(), 0
            for j in range(len(self.gens)):
                if (combo >> j) & 1:
                    prod = prod * self.gens[j][0]
                    recs ^= self.gens[j][1]
                    const ^= self.gens[j][2]
            check = prod * ps
            assert check.weight == 0 and check.sign in (1, -1), \
                f"tracker derivation sign not real: {check.sign}"
            return recs, const ^ int(check.sign == -1)
        if actual_recs is None:
            raise RuntimeError("random outcome with no captured records")
        pivot = anti[0]
        for g in anti[1:]:
            newps = g[0] * pivot[0]           # both in the abelian group
            assert newps.sign in (1, -1), "imaginary tracker product"
            g[2] ^= pivot[2] ^ int(newps.sign == -1)
            newps.sign = 1
            g[0] = newps
            g[1] = g[1] ^ pivot[1]
        pivot[0] = ps
        pivot[1] = frozenset(actual_recs)
        pivot[2] = 0
        return frozenset(actual_recs), 0


def _tracker_bit_specs(cp) -> List[Tuple[List[int], int]]:
    """Per program out-bit (program order): (record indices, const) via the
    symbolic tracker, gadget corrections folded in.  Events run in PHYSICAL
    order: joint/m steps by step index, then the terminal readouts (ancilla
    X and folded bits — all single-patch, pairwise commuting)."""
    prog, exp = cp.program, cp.experiment
    N = prog.num_qubits
    roles = step_roles(prog)
    folded = folded_out_bits(prog)
    targets = _bit_targets(prog)
    events = []
    for j, (kind, idx) in enumerate(roles):
        post = exp.step_joint_records_post.get(j)
        if kind == "gadget":
            g = prog.gadgets[idx]
            events.append((_pstring(N, {**g.mask, g.ancilla: "Z"}),
                           post, ("joint", idx)))
        else:
            events.append((_pstring(N, targets[idx]), post, ("mbit", idx)))
    for k, g in enumerate(prog.gadgets):
        nm = f"y{g.ancilla - prog.num_data}"
        events.append((_pstring(N, {g.ancilla: "X"}),
                       _support_recs(exp, nm, "X"), ("ancx", k)))
    for i, (q, letter) in folded.items():
        events.append((_pstring(N, {q: letter}),
                       _support_recs(exp, patch_name(q), letter),
                       ("mbit", i)))
    tr = _RecTracker()
    qubit_of = {patch_name(q): q for q in range(prog.num_data)}
    for (nm, letter), recs in sorted(getattr(exp, "seed_closure", {}).items()):
        q = qubit_of.get(nm)
        if q is None:
            continue
        tr.add_seed(_pstring(N, {q: letter}), recs)
    for g in prog.gadgets:
        # Gidney in-place birth pins the logical Y to +1: no records
        tr.add_seed(_pstring(N, {g.ancilla: "Y"}), frozenset())
    val = {}
    for ps, recs, tag in events:
        val[tag] = tr.measure(ps, recs)
    conds = []
    for k, g in enumerate(prog.gadgets):
        recs = val[("joint", k)][0] ^ val[("ancx", k)][0]
        const = val[("joint", k)][1] ^ val[("ancx", k)][1] ^ g.kappa
        for k2 in getattr(g, "conds", []):
            recs ^= conds[k2][0]
            const ^= conds[k2][1]
        conds.append((recs, const))
    out = []
    for i, ob in enumerate(prog.out_bits):
        recs, const = val[("mbit", i)]
        const ^= ob.flip
        for k in ob.gadgets:
            recs ^= conds[k][0]
            const ^= conds[k][1]
        out.append((sorted(recs), const))
    return out


def sample_program_bits(cp, shots: int, seed: int) -> np.ndarray:
    """(shots x n_original_bits) matrix of the ORIGINAL terminal bits, in
    load order."""
    samp = cp.circuit.compile_sampler(seed=seed)
    m = samp.sample(shots)                        # shots x num_measurements
    if getattr(cp, "t_as_s", False) or any(
            l == "X" for g in cp.program.gadgets for l in g.mask.values()):
        # every t_as_s compile takes the symbolic tracker: the raw-post
        # path's contamination comes from mixed-axis MERGES anywhere in
        # the program, not only from X letters in gadget masks (review
        # 2026-08-20: an all-Z-mask proxy with an X-bearing m frame reads
        # deterministic parities as random on the legacy path)
        specs = _tracker_bit_specs(cp)
        prog_bits = np.zeros((shots, len(specs)), dtype=bool)
        for i, (recs, const) in enumerate(specs):
            b = np.zeros(shots, dtype=bool)
            for r in recs:
                b ^= m[:, r]
            prog_bits[:, i] = b ^ bool(const)
    else:
        recs_spec = executed_bit_recs(cp)
        gadgets = _gadget_cond_recs(cp)
        conds = []
        for joint, ancx, kappa, chain in gadgets:
            b = np.zeros(shots, dtype=bool)
            for r in joint:
                b ^= m[:, r]
            for r in ancx:
                b ^= m[:, r]
            b ^= bool(kappa)
            # an earlier gadget's pending frame anticommutes with this
            # joint: observed b1 is the true value XOR that condition
            for k in chain:
                b ^= conds[k]
            conds.append(b)
        prog_bits = np.zeros((shots, len(recs_spec)), dtype=bool)
        for i, (recs, flip, gs) in enumerate(recs_spec):
            b = np.zeros(shots, dtype=bool)
            for r in recs:
                b ^= m[:, r]
            b ^= bool(flip)
            for k in gs:
                b ^= conds[k]
            prog_bits[:, i] = b
    # program order -> executed LOAD order via the full permutation chain
    # (weight-1 reorder AND step scheduling; cp.m_load_trace)
    trace = cp.m_load_trace
    n = prog_bits.shape[1]
    executed = np.zeros((shots, n), dtype=bool)
    for pos, load_idx in enumerate(trace):
        executed[:, load_idx] = prog_bits[:, pos]
    if cp.reconstruction is None:
        return executed
    # reduced run: original bit k = const[k] XOR executed bits over inverse[k]
    rec = cp.reconstruction
    orig = np.zeros((shots, len(rec.inverse)), dtype=bool)
    for k, inv in enumerate(rec.inverse):
        b = np.zeros(shots, dtype=bool)
        for j in inv:
            b ^= executed[:, j]
        orig[:, k] = b ^ bool(rec.const[k])
    return orig


def affine_structure(bits: np.ndarray) -> Tuple[List[int], List[int]]:
    """The deterministic-parity affine structure of a stabilizer output
    distribution: (basis of parity masks constant across shots, their
    values).  Masks are ints over bit positions; basis is in reduced
    row-echelon form so two structures compare by equality."""
    shots, n = bits.shape
    # difference rows: a parity mask v is deterministic iff it is GF(2)-
    # orthogonal to every (shot_s XOR shot_0)
    rows = []
    for s in range(1, shots):
        diff = 0
        for j in range(n):
            if bits[s, j] ^ bits[0, j]:
                diff |= 1 << j
        if diff:
            rows.append(diff)

    def rref(vectors):
        basis = []
        for v in vectors:
            for b in basis:
                if (v >> (b.bit_length() - 1)) & 1:
                    v ^= b
            if v:
                basis.append(v)
                basis.sort(key=lambda x: -x.bit_length())
        for i in range(len(basis)):          # back-substitution
            p = basis[i].bit_length() - 1
            for j in range(len(basis)):
                if j != i and (basis[j] >> p) & 1:
                    basis[j] ^= basis[i]
        basis.sort(reverse=True)
        return basis

    diff_basis = rref(rows)
    pivots = {b.bit_length() - 1 for b in diff_basis}
    dets = []
    for f in (j for j in range(n) if j not in pivots):
        v = 1 << f
        for b in diff_basis:
            if (b >> f) & 1:
                v |= 1 << (b.bit_length() - 1)
        dets.append(v)
    red = rref(dets)
    vals = []
    for v in red:
        x = 0
        for j in range(n):
            if (v >> j) & 1:
                x ^= int(bits[0, j])
        vals.append(x)
    return red, vals
