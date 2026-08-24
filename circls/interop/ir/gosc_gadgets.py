"""GoSC pass 2a: expand the y_free_sweep output into a PPM program.

Every explicit pi/4 rotation ``PauliOp('s', A, -1)`` (= A_{-pi/4}, the leading
Fig. 10 masks) becomes the Fig. 11b gadget in its |Y>-resource form:

    GoSC Sec. 2.1: "Similarly to a pi/8 rotation, a P_pi/4 rotation can be
    executed using a resource state |Y> = |0> + i|1>, as shown in Fig. 11b."

(GoSC then substitutes a |0> ancilla + P (x) Y measurement because "|Y> ...
cannot be readily prepared in our framework"; OUR framework prepares logical
|Y> fault-tolerantly in-place — Gidney arXiv:2302.07395, implemented in
LightStim — so we use the resource-state form directly, as decided 2026-08:
all PPM letters stay X/Z and the correction is Pauli-only, never Clifford.)

DECLARED EQUIVALENT FORM + cross-validation (project rule): with the ancilla
``a`` prepared in |+i> the gadget is

    1. jointly measure  A (x) Z_a          -> outcome m1 (+-1, bit b1)
    2. measure          X_a                -> outcome m2 (+-1, bit b2)
    3. conditional Pauli correction A on the data qubits.

Hand derivation (verified exactly by the dense oracle in
``tests/test_gosc_gadgets.py``): splitting |psi> = |psi+> + |psi-> into A
eigenspaces, the residual state is |psi+> +- i|psi->, i.e. A_{+-pi/4}|psi>,
and the correction A is needed iff  m1*m2 = -sigma  for a target rotation
A_{sigma*pi/4}.  The derivation never uses the letters of A, so it holds for
any X/Z mask and both signs: sigma = op.sign, giving the condition bit
c = b1 XOR b2 XOR kappa  with kappa = 1 iff sigma = -1.  (The Clifford
sweep's own heads are always (pure-Z, -1); the t_as_s proxy front end
injects arbitrary-axis heads of either sign.)

The correction is never executed: it is compiled into static XOR rules.  A
Pauli frame F passes through later measurements unchanged, flipping the
outcome of exactly those whose operator anticommutes with F; the swept stream
contains no unitaries after any gadget, so the frame only flips later
OUTCOMES.  A sweep-emitted pure-Z mask commutes with every later gadget joint
(all-Z joints, disjoint MX) and can only flip later 'm' bits; a general X/Z
mask can also anticommute with a LATER gadget's joint, in which case that
gadget's observed b1 is the true value XOR c, so its condition bit chains:
c_j = b1_j XOR b2_j XOR kappa_j XOR (XOR of c_k over earlier gadgets k whose
mask anticommutes with gadget j's mask) — ``Gadget.conds`` records those k.
Anticommutation of two Y-free masks is symplectic: odd number of positions
where both act with DIFFERENT letters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from circls.interop.ir.pauli_ir import PauliCircuit, PauliOp


@dataclass
class ProgramOp:
    """One primitive of the expanded program.

    kind 'init_y': prepare qubit (the single key of ``targets``) in |+i>.
    kind 'mpp':    jointly measure the (positive) Pauli product ``targets``;
                   consumes the next record index.
    Ancillas are indexed ``num_data + k`` for gadget k.
    """
    kind: str
    targets: Dict[int, str]


@dataclass
class Gadget:
    """Bookkeeping for one Fig. 11b gadget (mask A_{sigma*pi/4})."""
    mask: Dict[int, str]        # X/Z data support of A
    ancilla: int                # global qubit index of its |Y> ancilla
    rec_joint: int              # record index of the A (x) Z_a joint outcome
    rec_x: int                  # record index of the ancilla X readout
    kappa: int                  # c = b1 ^ b2 ^ kappa ^ (c_k for k in conds)
    conds: List[int] = field(default_factory=list)  # earlier gadgets whose
                                # frame anticommutes with this joint


@dataclass
class OutBit:
    """Classical definition of one original m_pauli bit:
    bit = record[rec] XOR flip XOR (XOR of conditions of ``gadgets``)."""
    rec: int
    flip: int                   # 1 iff the swept 'm' op carried sign -1
    gadgets: List[int] = field(default_factory=list)


@dataclass
class PPMProgram:
    num_data: int
    ops: List[ProgramOp]
    gadgets: List[Gadget]
    out_bits: List[OutBit]      # positional: i-th entry = i-th input m bit

    @property
    def num_ancilla(self) -> int:
        return len(self.gadgets)

    @property
    def num_qubits(self) -> int:
        return self.num_data + len(self.gadgets)


def _mask_anticommutes(mask: Dict[int, str], paulis: Dict[int, str]) -> bool:
    """Symplectic test for two Y-free operators: anticommute iff an odd
    number of positions carry both, with different letters (X vs Z)."""
    return sum(1 for q, l in mask.items()
               if q in paulis and paulis[q] != l) % 2 == 1


def expand_gadgets(swept: PauliCircuit) -> PPMProgram:
    """y_free_sweep output -> PPMProgram (gadget-expanded, corrections as XOR).

    Accepts the sweep's output alphabet: ``('s', X/Z mask, sign +-1)`` (the
    Clifford sweep itself only emits pure-Z sign -1 heads; t_as_s proxy heads
    are arbitrary-axis, either sign) and X/Z-only ``'m'`` ops — anything else
    is a loud error (t/z would be a pi/8 / pi rotation, out of scope for the
    Clifford-only version).
    """
    ops: List[ProgramOp] = []
    gadgets: List[Gadget] = []
    out_bits: List[OutBit] = []
    recs = 0
    for op in swept.ops:
        if any(p == "Y" for p in op.paulis.values()):
            raise ValueError(f"{op} contains Y — run y_free_sweep first")
        if op.kind == "s":
            if op.sign not in (1, -1) or not op.paulis:
                raise ValueError(f"unexpected pi/4 op {op}")
            a = swept.num_qubits + len(gadgets)
            ops.append(ProgramOp("init_y", {a: "Y"}))
            ops.append(ProgramOp("mpp", {**op.paulis, a: "Z"}))
            rec_joint = recs
            recs += 1
            ops.append(ProgramOp("mpp", {a: "X"}))
            rec_x = recs
            recs += 1
            gadgets.append(Gadget(
                dict(op.paulis), a, rec_joint, rec_x,
                # correction iff m1*m2 = -sigma  ->  c = b1^b2^kappa
                kappa=1 if op.sign == -1 else 0,
                conds=[k for k, g in enumerate(gadgets)
                       if _mask_anticommutes(g.mask, op.paulis)]))
        elif op.kind == "m":
            conds = [k for k, g in enumerate(gadgets)
                     if _mask_anticommutes(g.mask, op.paulis)]
            ops.append(ProgramOp("mpp", dict(op.paulis)))
            out_bits.append(OutBit(recs, int(op.sign < 0), conds))
            recs += 1
        else:
            raise ValueError(
                f"op kind {op.kind!r} unsupported in the Clifford-only "
                f"pipeline (t = pi/8 needs magic states, z = pi is a frame "
                f"update upstream)")
    return PPMProgram(swept.num_qubits, ops, gadgets, out_bits)


def _patch_name_of(program: PPMProgram, q: int) -> str:
    from circls.interop.ir.ppm_import import patch_name
    if q < program.num_data:
        return patch_name(q)
    return f"y{q - program.num_data}"


def program_step_interactions(program: PPMProgram):
    """The interaction_type list of every PPMStep the program maps to, in
    order — the SINGLE source of truth shared by ``to_experiment_inputs``
    (which builds the steps) and ``append_program_observables`` (which
    validates the experiment against them, the M2 handshake)."""
    anc = {g.ancilla for g in program.gadgets}
    folded_qubits = {q for q, _ in folded_out_bits(program).values()}
    out = []
    for op in program.ops:
        if op.kind != "mpp":
            continue
        if len(op.targets) == 1:
            (q, p), = op.targets.items()
            if p == "X" and q in anc:
                continue            # ancilla X readout -> final_measure_states
            if q in folded_qubits and q not in anc:
                continue            # folded weight-1 bit -> terminal readout
        out.append([(_patch_name_of(program, q), op.targets[q])
                    for q in sorted(op.targets)])
    return out


def folded_out_bits(program: PPMProgram):
    """{out_bit_index: (data_qubit, letter)} for the weight-1 DATA
    measurements that can be DEFERRED to their patch's terminal readout: the
    qubit must not appear in ANY later program op (joint, mask or
    measurement) — deferring past an anticommuting later operation would
    change the outcome.  After ``reorder_weight1_last`` this covers every
    weight-1 op whose operator survived the sweep thin; a weight-1 op whose
    qubit IS reused later cannot be realised (no single-patch measurement
    primitive; the measurement-ancilla expansion is future work) and raises.
    """
    anc = {g.ancilla for g in program.gadgets}
    mpps = [op for op in program.ops if op.kind == "mpp"]
    folded = {}
    bit = 0
    for j, op in enumerate(mpps):
        if len(op.targets) == 1:
            (q, p), = op.targets.items()
            if p == "X" and q in anc:
                continue                     # ancilla readout, not a bit
        if any(q in anc for q in op.targets):
            continue                         # gadget joint, not a bit
        if len(op.targets) == 1:
            (q, p), = op.targets.items()
            reused = any(q in later.targets for later in mpps[j + 1:])
            if reused:
                raise ValueError(
                    f"weight-1 measurement of qubit {q} (bit {bit}) is "
                    f"followed by later operations touching qubit {q} — it "
                    f"cannot be deferred to the terminal readout, and there "
                    f"is no single-patch measurement primitive (v1). Run "
                    f"reorder_weight1_last on the m stream before the sweep.")
            folded[bit] = (q, p)
        bit += 1
    return folded


def step_roles(program: PPMProgram):
    """Role of each experiment PPMStep, in to_experiment_inputs order:
    ('gadget', k) or ('m', out_bit_index).  Ancilla X readouts and FOLDED
    weight-1 bits (terminal-readout deferrals) never become steps."""
    roles = []
    n_g = n_m = 0
    anc = {g.ancilla for g in program.gadgets}
    folded = folded_out_bits(program)
    for op in program.ops:
        if op.kind != "mpp":
            continue
        if len(op.targets) == 1:
            (q, p), = op.targets.items()
            if p == "X" and q in anc:
                continue
        if any(q in anc for q in op.targets):
            roles.append(("gadget", n_g))
            n_g += 1
        else:
            if n_m not in folded:
                roles.append(("m", n_m))
            n_m += 1
    return roles


def append_program_observables(circuit, exp, program: PPMProgram):
    """Append one OBSERVABLE_INCLUDE per DETERMINISTIC program bit to the
    built circuit; return {out_bit_index: observable_index or None}.

    The observable for bit i is the deterministic closure
    ``pre XOR post`` of its m-step's joint-parity record sets (extracted by
    the experiment's read-only hooks): ``post`` reconstructs this step's
    outcome, ``pre`` reconstructs the same operator from everything measured
    BEFORE the step — the solver pulls in whatever earlier records (gadget
    joints, prior m outcomes) make it deterministic, so the GoSC correction
    bookkeeping is baked in automatically.  ``pre is None`` means the bit is
    a genuinely random free output (measuring Y-ish operators on |0..0> IS
    random): no error-sensitive content of the program is lost by skipping
    it — noise can only manifest on deterministic parities, all of which are
    emitted.  Constants (op signs, gadget kappas) are classical
    postprocessing and never enter a stim observable.
    """
    import stim
    from circls.interop.ir.ppm_import import patch_name as _pn
    roles = step_roles(program)
    folded = folded_out_bits(program)
    # M2 handshake: the positional binding bit -> step index is only sound if
    # the experiment's ppm_sequence IS this program's step list.  Validate
    # count and per-step content loudly — catches a mispaired (exp, program)
    # and any drift between the step builder and the roles walk.
    expected = program_step_interactions(program)
    seq = getattr(exp, 'ppm_sequence', None)
    if seq is None or len(seq) != len(expected):
        raise ValueError(
            f"experiment/program mismatch: experiment has "
            f"{None if seq is None else len(seq)} PPM steps, the program "
            f"maps to {len(expected)} — wrong (exp, program) pairing?")
    if len(roles) != len(expected):
        raise AssertionError(
            f"internal drift: step_roles yields {len(roles)} steps but "
            f"program_step_interactions yields {len(expected)}")
    for j, (want, got) in enumerate(zip(expected, seq)):
        if list(got.interaction_type) != list(want):
            raise ValueError(
                f"experiment/program mismatch at step {j}: experiment has "
                f"{got.interaction_type}, the program expects {want} — "
                f"wrong pairing or reordered/edited steps")
    m_step_of = {bit: j for j, (kind, bit) in enumerate(roles)
                 if kind == "m"}
    out = {}
    total = circuit.num_measurements
    for i in range(len(program.out_bits)):
        if i in folded:
            # deferred bit = the patch's terminal readout: deterministic
            # pre-readout parity XOR the readout records over the logical
            # operator's data support (all read in that letter)
            q, letter = folded[i]
            nm = _pn(q)
            pre = exp.prereadout_parity.get((nm, letter))
            if pre is None:
                out[i] = None            # genuinely random free output
                continue
            support = [dq for rec in exp.system.logical_ops
                       if rec.get('patch_name') == nm
                       and rec.get('type') == letter
                       for dq in rec['pauli']]
            recs = sorted(set(pre)
                          ^ {exp.final_readout_recs[dq] for dq in support})
        else:
            pre = exp.step_joint_records_pre.get(m_step_of[i])
            post = exp.step_joint_records_post.get(m_step_of[i])
            if pre is None or post is None:
                out[i] = None
                continue
            recs = sorted(set(pre) ^ set(post))
        idx = circuit.num_observables
        circuit.append("OBSERVABLE_INCLUDE",
                       [stim.target_rec(r - total) for r in recs], [idx])
        out[i] = idx
    return out


# --------------------------------------------------------------- pass 2b shim
def to_experiment_inputs(program: PPMProgram, distance: int = 3,
                         assignment: str = "row_major", placement=None,
                         orientation=None):
    """Structural mapping of a PPMProgram onto SequentialPPMExperiment inputs.

    Returns ``(specs, steps, initial_states, final_measure_states, program)``.
    Data patches are named ``q{i}`` (ppm_import convention), gadget ancillas
    ``y{k}``.  Ancilla lifecycle: initial state 'Y' (realised on the
    experiment side via the Gidney in-place birth), final state 'X' (the
    gadget's X readout IS the ancilla's terminal measurement).  Data patches get placeholder 'Z' initial/final states, as
    in ppm_import.to_experiment; the observable assembly that turns PPM
    records + XOR rules into the program's classical bits is experiment-side
    work (task 3), not done here.

    Every joint 'mpp' becomes one direct-joint PPMStep; the weight-1 ancilla
    X readouts are folded into final_measure_states instead of steps.
    """
    from circls.core.sequential_ppm_ls import PPMStep
    from circls.interop.ir.ppm_import import patch_name
    from circls.compiler.mapping import place_patches, place_patches_optimized

    def name(q: int) -> str:
        if q < program.num_data:
            return patch_name(q)
        return f"y{q - program.num_data}"

    folded = folded_out_bits(program)
    folded_qubits = {q for q, _ in folded.values()}
    steps: List[PPMStep] = [PPMStep(list(it))
                            for it in program_step_interactions(program)]

    names = ({nm for s in steps for nm, _ in s.interaction_type}
             | {name(q) for q in folded_qubits})
    first_letters: dict = {}
    for s in steps:
        for nm, letter in s.interaction_type:
            first_letters.setdefault(nm, letter)
    # Square Sparse floor (circls.compiler.mapping); the assignment policy is a knob:
    # 'row_major' (baseline) or 'optimized' (circls.compiler.assignment portfolio).
    if orientation is not None:
        # result-injection hook (docs/API_HOOKS.md): an externally chosen
        # BIRTH orientation per patch.  Partial dicts are fine; composes
        # with placement= and with assignment='optimized' (the optimizer
        # plans its cells against the forced orientations).
        legal = {"X_horizontal", "X_vertical"}
        extra = sorted(set(orientation) - names)
        if extra:
            raise ValueError(f"orientation names unknown patches: {extra}")
        bad = {nm: o for nm, o in orientation.items() if o not in legal}
        if bad:
            raise ValueError(
                f"orientation values must be in {sorted(legal)}: {bad}")
        ybad = sorted(nm for nm in orientation if nm.startswith("y"))
        if ybad:
            raise ValueError(
                "orientation cannot override |Y> patches (the Gidney "
                f"birth layout is protocol-fixed X_vertical): {ybad}")
    if placement is not None:
        # result-injection hook (docs/API_HOOKS.md): an externally
        # computed {patch_name: coarse cell} — validate loudly, then
        # place with the caller's cells.
        if assignment == "optimized":
            raise ValueError(
                "placement= and assignment='optimized' together — the "
                "injected placement replaces the assignment search")
        missing = sorted(names - set(placement))
        if missing:
            raise ValueError(f"placement missing patches: {missing}")
        extra = sorted(set(placement) - names)
        if extra:
            raise ValueError(f"placement names unknown patches: {extra}")
        cells = {}
        for nm in names:
            c = tuple(placement[nm])
            if len(c) != 2 or not all(isinstance(v, int) and v >= 0
                                      for v in c):
                raise ValueError(
                    f"placement cell for {nm} must be two ints >= 0: "
                    f"{placement[nm]!r}")
            cells[nm] = c
        if len(set(cells.values())) != len(cells):
            raise ValueError("placement assigns two patches to one cell")
        specs = place_patches(names, distance,
                              first_letters=first_letters, cells=cells,
                              orients=orientation)
    elif assignment == "optimized":
        specs = place_patches_optimized(
            names, [s.interaction_type for s in steps], distance,
            first_letters=first_letters, orients=orientation)
    elif assignment == "row_major":
        specs = place_patches(names, distance, first_letters=first_letters,
                              orients=orientation)
    else:
        raise ValueError(f"unknown assignment policy {assignment!r}")
    initial_states = {nm: ("Y" if nm.startswith("y") else "Z") for nm in names}
    final_measure_states = {nm: ("X" if nm.startswith("y") else "Z")
                            for nm in names}
    for q, letter in folded.values():
        final_measure_states[name(q)] = letter    # the deferred measurement
    return specs, steps, initial_states, final_measure_states, program
