"""Program-level and circuit-level readout of a compiled program.

Three views of the ORIGINAL output bits of a CompiledProgram, all in LOAD
ORDER (original m op order = qubit order), the canonical space shared by
reduction-on and reduction-off compilations:

* ``sample_program_bits``: per-shot bits of the real stim circuit at
  p = 0, assembled from the measurement records by the record-set
  symbolic tracker (``_RecTracker``): random outcomes are anchored to
  the records that produced them, deterministic ones are derived from
  the seed closures (patch initialisations, |Y> births).  Every
  bookkeeping layer is applied: |Y>-gadget corrections
  (c = b_joint XOR b_x XOR kappa, chained through ``Gadget.conds``),
  swept-op sign flips, weight-1 terminal folding, the weight-1 /
  scheduling permutation chain (``m_load_trace``) and the
  measurement-reduction reconstruction.
* ``logical_program_structure``: the same classical bookkeeping applied
  to a LOGICAL-level simulation of the expanded PPM program (one qubit
  per patch), independent of the physical circuit.  Compared with
  ``source_truth_structure`` it checks the whole front end: loader,
  re-selection, reordering, Y-free sweep, scheduling, gadget expansion
  and the reconstruction map.
* ``logical_state_check``: the physical circuit run noise-free up to its
  terminal readout, with every logical stabilizer of the expected
  logical state peeked through the experiment's logical operators;
  |<S>| = 1 for all of them means the circuit realises the program's
  logical state up to a Pauli frame (a byproduct can only flip a sign).

The last two read no measurement record at all.  They replaced the
record-level affine comparison in ``verify`` (CircLS issue #1,
2026-10-14): the previous raw-record reconstruction mis-accounted the
lattice-surgery byproducts of programs that contain both X-type merges
and the pi/4 gadgets inserted by the Y-free rewrite, and reported a
deterministic program bit as random on circuits that were correct.

Known blind spot: no check in this module or in ``verify`` compares the
VALUE of a physical outcome with the source truth.  A logical Pauli
inserted into the circuit (a Pauli-frame error) commutes with every
check, changes no detector, keeps the program observables constant
(stim's detector sampler reports flips relative to the noiseless
reference) and is invisible to ``logical_state_check`` (which takes
|<S>|).  Deterministic bits of ``sample_program_bits`` are derived from
the seed closures and do not see it either.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import stim

from circls.interop.ir.gosc_gadgets import folded_out_bits, step_roles
from circls.interop.ir.ppm_import import patch_name

_LETTER = {"X": 1, "Y": 2, "Z": 3}
_ANNOTATIONS = {"DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS",
                "SHIFT_COORDS", "TICK"}


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


def _assemble_original_bits(cp, prog_bits: np.ndarray) -> np.ndarray:
    """Program-order bits -> executed LOAD order (the weight-1 reorder AND
    step-scheduling permutation chain, ``cp.m_load_trace``) -> ORIGINAL
    bits through the measurement-reduction reconstruction (original bit k
    = const[k] XOR executed bits over inverse[k])."""
    shots, n = prog_bits.shape
    trace = cp.m_load_trace if cp.m_load_trace is not None else list(range(n))
    executed = np.zeros((shots, n), dtype=bool)
    for pos, load_idx in enumerate(trace):
        executed[:, load_idx] = prog_bits[:, pos]
    if cp.reconstruction is None:
        return executed
    rec = cp.reconstruction
    orig = np.zeros((shots, len(rec.inverse)), dtype=bool)
    for k, inv in enumerate(rec.inverse):
        b = np.zeros(shots, dtype=bool)
        for j in inv:
            b ^= executed[:, j]
        orig[:, k] = b ^ bool(rec.const[k])
    return orig


def sample_program_bits(cp, shots: int, seed: int) -> np.ndarray:
    """(shots x n_original_bits) matrix of the ORIGINAL terminal bits, in
    load order, read off the noise-free circuit through the record-set
    symbolic tracker (see the module docstring).  A p = 0 oracle: the
    deterministic bits are reconstructed from the seed closures, so this
    is not a decoder for noisy records."""
    samp = cp.circuit.compile_sampler(seed=seed)
    m = samp.sample(shots)                        # shots x num_measurements
    specs = _tracker_bit_specs(cp)
    prog_bits = np.zeros((shots, len(specs)), dtype=bool)
    for i, (recs, const) in enumerate(specs):
        b = np.zeros(shots, dtype=bool)
        for r in recs:
            b ^= m[:, r]
        prog_bits[:, i] = b ^ bool(const)
    return _assemble_original_bits(cp, prog_bits)


# ── record-free checks (CircLS issue #1) ─────────────────────────────────────

def _program_patch_name(prog, q: int) -> str:
    return patch_name(q) if q < prog.num_data else f"y{q - prog.num_data}"


def _terminal_mpp_indices(prog) -> List[int]:
    """Indices (into ``prog.ops``) of the single-patch mpp ops that the
    experiment realises as a patch's terminal readout: gadget-ancilla X
    readouts and the folded weight-1 data readouts."""
    # folded_out_bits raises on a weight-1 op that is reused later, so every
    # single-patch op of an expanded program is terminal for its qubit
    return [i for i, op in enumerate(prog.ops)
            if op.kind == "mpp" and len(op.targets) == 1]


_INIT = {"Y": "R {q}\nH {q}\nS {q}",     # |+i>, the +1 eigenstate of Y
         "X": "R {q}\nH {q}",              # |+>, the |+> proxy ancilla
         "Z": "R {q}"}


def simulate_program(prog, seed: int = 0, skip_ops=(), init_letters=None):
    """Logical-level run of an expanded PPM program, one qubit per patch.

    ``init_y`` prepares |+i> (the +1 eigenstate of Y; a global flip of
    every ancilla's sign complex-conjugates a Y-free program and changes
    no outcome distribution, so the convention is immaterial to the
    checks built on this) unless ``init_letters`` maps the ancilla to
    another letter (``'X'``: the |+> proxy ancilla of ``magic_proxy="X"``
    compiles); every ``mpp`` measures its (positive) Pauli product and
    consumes the next record index, exactly as ``expand_gadgets`` numbers
    ``Gadget.rec_joint`` / ``rec_x`` / ``OutBit.rec``.  Ops whose index
    is in ``skip_ops`` are not executed (their record is None).  Returns
    ``(records, simulator)``."""
    n = prog.num_qubits
    sim = stim.TableauSimulator(seed=seed)
    sim.set_num_qubits(n)
    recs: List[Optional[int]] = []
    skip = set(skip_ops)
    init_letters = init_letters or {}
    for i, op in enumerate(prog.ops):
        if op.kind == "init_y":
            q, = op.targets
            sim.do(stim.Circuit(_INIT[init_letters.get(q, "Y")].format(q=q)))
        elif op.kind == "mpp":
            if i in skip:
                recs.append(None)
                continue
            recs.append(int(sim.measure_observable(_pstring(n, op.targets))))
        else:
            raise ValueError(f"unknown program op kind {op.kind!r}")
    return recs, sim


def program_bits_from_records(cp, recs: List[int]) -> List[int]:
    """The compiler's own classical rules applied to one shot of program-
    order outcomes: gadget conditions c = b_joint ^ b_x ^ kappa (chained
    through ``conds``), out-bit flips and gadget corrections."""
    prog = cp.program
    conds: List[int] = []
    for g in prog.gadgets:
        c = recs[g.rec_joint] ^ recs[g.rec_x] ^ g.kappa
        for k in getattr(g, "conds", []):
            c ^= conds[k]
        conds.append(c)
    out = []
    for ob in prog.out_bits:
        b = recs[ob.rec] ^ ob.flip
        for k in ob.gadgets:
            b ^= conds[k]
        out.append(b)
    return out


def logical_program_structure(cp, shots: int = 256, seed: int = 0):
    """Affine structure (``affine_structure``) of the ORIGINAL output bits
    of a logical-level simulation of ``cp.program`` with the compiler's
    own correction and reconstruction rules.  Reads no circuit record."""
    rows = []
    for shot in range(shots):
        recs, _ = simulate_program(cp.program, seed=seed * 7919 + shot)
        rows.append(program_bits_from_records(cp, recs))
    bits = np.array(rows, dtype=bool).reshape(shots, -1)
    return affine_structure(_assemble_original_bits(cp, bits))


def source_truth_structure(qasm: str, shots: int = 256, seed: int = 0):
    """Affine structure of the source program's terminal measurements,
    from the Clifford front end's operators measured on |0..0> by a stim
    tableau simulation (compilation-independent)."""
    from circls.interop.nwqec.frontend import load_clifford_mpauli
    raw = load_clifford_mpauli(qasm)
    n = raw.num_qubits
    mops = []
    for op in raw.ops:
        ps = stim.PauliString(n)
        for q, l in op.paulis.items():
            ps[q] = _LETTER[l]
        if op.sign == -1:
            ps.sign = -1
        mops.append(ps)
    rows = []
    for shot in range(shots):
        ts = stim.TableauSimulator(seed=1000 + seed + shot)
        rows.append([ts.measure_observable(ps) for ps in mops])
    return affine_structure(np.array(rows, dtype=bool))


@dataclass
class LogicalStateReport:
    ok: Optional[bool]                 # None: the readout cut was not clean
    n_stabilizers: int                 # expected logical stabilizers checked
    min_abs_expectation: Optional[int] # 1 = all realised; 0 = some not
    survivors: List[str]               # patches alive at the terminal readout
    cut_record: Optional[int]          # first record of the terminal readout
    detail: str = ""


def _restrict_to_support(stabs, n: int, keep) -> List[stim.PauliString]:
    """Generators of the subgroup of <stabs> supported on the qubits in
    ``keep`` (the stabilizer group of the marginal state): GF(2)
    elimination on the columns OUTSIDE ``keep``; every generator whose
    outside part reduces to zero is a product of the input generators
    with no outside support."""
    outside = 0
    for q in range(n):
        if q not in keep:
            outside |= (1 << q) | (1 << (n + q))

    def vec(ps):
        v = 0
        for q in range(n):
            c = ps[q]
            if c in (1, 2):
                v |= 1 << q
            if c in (2, 3):
                v |= 1 << (n + q)
        return v

    basis: Dict[int, int] = {}                 # pivot bit (outside) -> full vector
    kept = []
    for ps in stabs:
        v = vec(ps)
        while True:
            o = v & outside
            if not o:
                kept.append(v)
                break
            piv = o.bit_length() - 1
            if piv in basis:
                v ^= basis[piv]
            else:
                basis[piv] = v
                break
    out = []
    for v in kept:
        if not v:
            continue
        ps = stim.PauliString(n)
        for q in range(n):
            x, z = (v >> q) & 1, (v >> (n + q)) & 1
            ps[q] = 1 if (x and not z) else 2 if (x and z) else 3 if z else 0
        out.append(ps)
    return out


def logical_state_check(cp) -> LogicalStateReport:
    """Does the noise-free circuit realise the program's logical state?

    The expected state is the logical-level simulation of ``cp.program``
    with the terminal readouts of the patches that are still alive at the
    circuit's final readout left unexecuted (patches retired earlier by
    the liveness pass are measured in the simulation as well).  Its
    stabilizer group, restricted to the surviving patches, is mapped to
    physical Paulis through ``exp.system.logical_ops`` and peeked on the
    stim tableau simulator run up to the first terminal-readout record.
    Every |<S>| must be 1: Pauli byproducts of the surgery only change the
    sign, while an operation that is missing, wrong or misplaced gives 0
    whenever it changes the state (a gadget whose mask stabilises the
    state at that point acts trivially and is invisible here, as it is in
    the output distribution).  Reads no record and compares no outcome
    value (see the module docstring)."""
    prog, exp, circ = cp.program, cp.experiment, cp.circuit
    owner = exp.system.index_to_owner_map
    first_rec: Dict[str, int] = {}
    for q, r in exp.final_readout_recs.items():
        nm = owner.get(q)
        if nm is not None:
            first_rec[nm] = min(r, first_rec.get(nm, r))
    if not first_rec:
        return LogicalStateReport(None, 0, None, [], None,
                                  "no terminal readout records captured")
    # the terminal readout = the last run of measuring instructions
    flat = circ.flattened()
    rec, run_start = 0, 0
    for inst in flat:
        k = inst.num_measurements
        if k:
            rec += k
        elif inst.name not in _ANNOTATIONS:
            run_start = rec
    survivors = sorted(nm for nm, r in first_rec.items() if r >= run_start)
    if not survivors:
        return LogicalStateReport(None, 0, None, [], None,
                                  "no patch readout in the circuit's last "
                                  "run of measurements (instructions after "
                                  "the terminal readout?)")
    cut = min(first_rec[nm] for nm in survivors)
    # expected logical state: everything but the survivors' terminal readouts
    name_of = {q: _program_patch_name(prog, q) for q in range(prog.num_qubits)}
    skip = [i for i in _terminal_mpp_indices(prog)
            if name_of[next(iter(prog.ops[i].targets))] in survivors]
    # ancillas in the state the experiment actually prepares (|Y> birth, or
    # the |+> proxy): the check is "the circuit realises the program as
    # compiled", the ideal semantics are logical_program_structure's job
    init_letters = {q: exp.initial_states[nm]
                    for q, nm in name_of.items()
                    if q >= prog.num_data and nm in exp.initial_states}
    _, lsim = simulate_program(prog, seed=7, skip_ops=skip,
                               init_letters=init_letters)
    keep = {q for q, nm in name_of.items() if nm in survivors}
    stabs = _restrict_to_support(lsim.canonical_stabilizers(),
                                 prog.num_qubits, keep)
    reps = {(r["patch_name"], r["type"]): r["pauli"]
            for r in exp.system.logical_ops}
    nphys = circ.num_qubits

    def physical(lp: stim.PauliString) -> stim.PauliString:
        tot = stim.PauliString(nphys)
        for q in range(prog.num_qubits):
            c = lp[q]
            if not c:
                continue
            nm = name_of[q]
            if c in (1, 2):
                tot = tot * _pstring(nphys, reps[(nm, "X")])
            if c in (2, 3):
                tot = tot * _pstring(nphys, reps[(nm, "Z")])
        tot.sign = +1
        return tot

    sim = stim.TableauSimulator(seed=3)
    sim.set_num_qubits(nphys)
    rec = 0
    clean = False
    for inst in flat:
        if inst.name in _ANNOTATIONS:
            continue
        k = inst.num_measurements
        if k and rec == cut:
            clean = True                       # the terminal readout starts here
            break
        if k and rec + k > cut:
            break                              # a measuring instruction straddles the cut
        sim.do(inst)                           # gates before the readout are applied
        rec += k
    else:
        clean = rec == cut
    if not clean:
        return LogicalStateReport(None, len(stabs), None, survivors, cut,
                                  f"terminal readout does not start at an "
                                  f"instruction boundary (record {cut})")
    exps = [abs(sim.peek_observable_expectation(physical(st))) for st in stabs]
    mn = min(exps) if exps else 1
    notes = []
    if mn != 1:
        notes.append(f"{sum(1 for e in exps if e != 1)} of {len(exps)} "
                     f"expected logical stabilizers not realised")
    if len(stabs) < len(survivors):
        notes.append(f"expected state fixes only {len(stabs)} of "
                     f"{len(survivors)} surviving logical qubits")
    return LogicalStateReport(mn == 1, len(stabs), mn, survivors, cut,
                              "; ".join(notes))


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
