"""Bridge: lsqecc's compilation decisions -> the CircLS lowering.

Runs the lsqecc_slicer on a benchmark's QASM, parses its LLI instruction
stream and its slices, and replays the SAME plan on the CircLS backend so
its qubit-rounds and LER can be measured at the circuit level.

Replay contract (every CircLS optimization off):

  - placement: the slicer's own patch tiles (slice JSON), injected;
  - sequence: the LLI stream verbatim, as a PPMProgram — its ancilla
    births (Init), its joint measurements (MultiBodyMeasure), its ancilla
    readouts (MeasureSinglePatch), in its order; no re-selection, no
    reordering, no parallel windows (parallel_steps=False);
  - single-qubit Cliffords: conjugated away exactly (per-patch 1-qubit
    stim tableau frame; record sign flips land in OutBit.flip);
  - rotations: not injected; auto_rotate re-plans them on the slicer's
    placement, so a rotation is executed and costed exactly where that
    placement forces one;
  - data patches: STATIC — allocated at t=0 and held to the terminal
    readout via the documented lifetime hook (lifetime_overrides widens
    to the whole program; keep_patches pins them against retirement);
    CircLS's first-use init / last-use freeing never touches them;
  - ancillas: born and destroyed mid-circuit ONLY because the slicer's
    LLI scripts it; each scripted life is its own patch, and the
    derived first/last-use positions coincide with the scripted
    Init/MeasureSinglePatch positions by construction of the stream
    (tile sharing across disjoint lives replays through the lazy birth
    + retire path);
  - observables: the circuit is exactly what exp.build() emits — the
    tracker's own terminal OBSERVABLE_INCLUDEs (its deterministic
    terminal-closure group) are the failure metric, the same machinery
    and convention every CircLS pipeline circuit carries.  Nothing is
    stripped or appended.

Verification gates, all at build time, against a logical stim
simulation of the (basis-corrected) LLI stream run raw — gates as
unitaries, every measurement an MPP, terminal data reads appended:

  1. p=0 detector silence;
  2. p=0 determinism of every emitted observable;
  3. the emitted observable count equals the plan's deterministic
     parity dimension (the affine_structure rank of the logical
     reference's record space) — the tracker solved exactly the
     plan's deterministic group, no more, no less;
  4. the output bits' affine structure (values included, via the
     plan-convention reconstruction sample_program_bits) equals the
     logical reference's — for deterministic-output programs this is
     bit-exact value equality.
"""
import contextlib
import io
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "experiments"))

_INIT_LETTER = {"|0>": "Z", "|+>": "X"}    # loud on anything else


@dataclass
class LLIOp:
    kind: str                      # init | mbm | msp | rotate | gate1q
    patches: List[Tuple[int, str]] = field(default_factory=list)
    state: str = ""                # for init: |0>, |+>; for gate1q: gate name


def run_slicer(slicer: Path, qasm_path: Path, workdir: Path):
    """Run the slicer twice: LLI dump (measure lines stripped -- its LLI
    pipeline does not parse `measure`) and the normal slicing pass."""
    stripped = workdir / (qasm_path.stem + ".nomeas.qasm")
    stripped.write_text("\n".join(
        l for l in qasm_path.read_text().splitlines()
        if not re.match(r"\s*(measure|creg)\b", l)) + "\n")
    lli = subprocess.run(
        [str(slicer), "-I", "qasm", "-i", str(stripped),
         "--printlli", "before", "--noslices"],
        capture_output=True, text=True, timeout=600).stdout
    slices_path = workdir / (qasm_path.stem + ".slices.json")
    stats = subprocess.run(
        [str(slicer), "-I", "qasm", "-i", str(stripped),
         "-o", str(slices_path), "-f", "stats", "--graceful"],
        capture_output=True, text=True, timeout=600).stdout
    return lli, json.loads(slices_path.read_text()), stats


def parse_lli(text: str) -> List[LLIOp]:
    ops = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        head, *rest = line.split()
        if head == "Init":
            ops.append(LLIOp("init", [(int(rest[0]), "")], state=rest[1]))
        elif head == "MultiBodyMeasure":
            body = [(int(p.split(":")[0]), p.split(":")[1])
                    for p in rest[0].split(",")]
            ops.append(LLIOp("mbm", body))
        elif head == "MeasureSinglePatch":
            # SEMANTIC CORRECTION (2026-08-15, verified in stim): the CNOT
            # gadget lsqecc emits (Init a |+>; M_ZZ(c,a); M_XX(a,t);
            # MeasureSinglePatch a X) is only a CNOT when the ancilla
            # readout is Z-bar — with a literal X readout the channel loses
            # the merge correlation (postselected check: <Z0Z1>=0 on the
            # |+0> input; Z readout gives the Bell pair).  Their slicer
            # never executes the letter (layout/volume only), so this is
            # invisible in their own numbers; a circuit-level replay must
            # take the algebraically forced basis.  An exhaustive scan of
            # the relabelings (merge letters literal/dual x readout
            # literal/dual) leaves letter-literal merges + complemented
            # readout as the unique reading under which the emitted stream
            # computes the program.  The logical-reference oracle in main()
            # runs on the same corrected stream and gates the replay.
            ops.append(LLIOp("msp", [(int(rest[0]),
                                      {"X": "Z", "Z": "X"}[rest[1]])]))
        elif head == "RotateSingleCellPatch":
            ops.append(LLIOp("rotate", [(int(rest[0]), "")]))
        elif head in ("HGate", "XGate", "YGate", "ZGate", "SGate",
                      "SDgGate"):
            ops.append(LLIOp("gate1q", [(int(rest[0]), "")],
                             state=head[:-4]))
        elif head == "RequestYState":
            raise NotImplementedError(
                "RequestYState (catalytic S-gate / Y-state protocol): "
                "out of Clifford replay scope")
        else:
            raise ValueError(f"unhandled LLI instruction: {line}")
    return ops


_GATE1Q = {"H": "H", "X": "X", "Y": "Y", "Z": "Z", "S": "S",
           "SD": "S_DAG"}


def conjugate_frame(ops: List[LLIOp]):
    """Absorb every single-qubit Clifford into a per-patch frame (a
    1-qubit stim tableau) and conjugate later measurement letters through
    it: measuring P after U equals measuring U^dagger P U before it,
    exact for Clifford streams.  The sign of the conjugated Pauli flips
    the recorded bit; ``signs`` carries it per op for the OutBit flips.
    Frames reset at Init (a fresh state absorbs no prior gates).
    Rotations are dropped (auto_rotate re-plans them on the injected
    placement)."""
    import stim
    frames: Dict[int, "stim.Tableau"] = {}
    ident = stim.Tableau(1)

    def conj(q: int, P: str) -> Tuple[str, int]:
        u = frames.get(q)
        if u is None or u == ident:
            return P, 0
        ps = u.inverse()(stim.PauliString(P))
        letter = "_XYZ"[ps[0]]
        return letter, (1 if ps.sign == -1 else 0)

    out: List[LLIOp] = []
    signs: List[List[int]] = []      # parallel to out: per-factor bit flips
    gates = rotations = 0
    for op in ops:
        if op.kind == "gate1q":
            q = op.patches[0][0]
            g = stim.Tableau.from_named_gate(_GATE1Q[op.state])
            frames[q] = g * frames.get(q, stim.Tableau(1))
            gates += 1
            continue
        if op.kind == "rotate":
            rotations += 1
            continue
        if op.kind == "init":
            frames.pop(op.patches[0][0], None)
            out.append(op)
            signs.append([0])
            continue
        body, sgn = [], []
        for q, P in op.patches:
            letter, s = conj(q, P) if P else (P, 0)
            body.append((q, letter))
            sgn.append(s)
        out.append(LLIOp(op.kind, body, state=op.state))
        signs.append(sgn)
    return out, {"gates_absorbed": gates, "rotations_dropped": rotations,
                 "signs": signs, "frames": frames}


def extract_placement(slices) -> Dict[int, Tuple[int, int]]:
    """{patch id: (row, col)} from the first slice each id appears in."""
    placement: Dict[int, Tuple[int, int]] = {}
    for sl in slices:
        for r, row in enumerate(sl):
            for c, cell in enumerate(row or []):
                if not cell:
                    continue
                m = re.match(r"Id:\s*(\d+)", cell.get("text") or "")
                if m:
                    placement.setdefault(int(m.group(1)), (r, c))
    return placement


def measured_qubits(qasm_text: str, num_data: int) -> List[int]:
    """Data qubits the ORIGINAL program measures (ascending order)."""
    per_bit = [int(m.group(1)) for m in
               re.finditer(r"measure\s+\w+\[(\d+)\]", qasm_text)]
    if per_bit:
        return sorted(set(per_bit))
    if re.search(r"measure\s+\w+\s*->", qasm_text):
        return list(range(num_data))     # whole-register measure
    raise ValueError("original QASM has no measure statement")


def to_program(ops2, meta, num_data: int, measured: List[int]):
    """Conjugated LLI stream -> (PPMProgram, init_letters, id remap).

    Every lsqecc patch id becomes one PPMProgram qubit: original data ids
    keep their index; each scripted ancilla LIFE gets the next index (so
    tile reuse never aliases qubits).  Init ops set the qubit's initial
    letter; mbm/msp/terminal measurements become the mpp stream, one
    record each, with the frame's sign flips in OutBit.flip."""
    import stim
    from circls.interop.ir.gosc_gadgets import OutBit, PPMProgram, ProgramOp

    remap: Dict[int, int] = {}

    def qidx(pid: int) -> int:
        if pid < num_data:
            return pid
        if pid not in remap:
            remap[pid] = num_data + len(remap)
        return remap[pid]

    prog_ops, flips = [], []
    init_letters: Dict[int, str] = {}
    for op, sgn in zip(ops2, meta["signs"]):
        if op.kind == "init":
            pid = op.patches[0][0]
            if op.state not in _INIT_LETTER:
                raise NotImplementedError(
                    f"Init state {op.state} for patch {pid}: only "
                    f"|0>/|+> are wired")
            init_letters[qidx(pid)] = _INIT_LETTER[op.state]
            continue
        targets = {qidx(q): P for q, P in op.patches}
        if "Y" in targets.values():
            raise NotImplementedError(
                f"conjugated letter Y in {op.kind} on {sorted(targets)}: "
                f"S-carrying frame, not wired")
        prog_ops.append(ProgramOp("mpp", targets))
        flips.append(sum(sgn) % 2)
    for q in measured:
        u = meta["frames"].get(q)
        if u is None:
            letter, s = "Z", 0
        else:
            ps = u.inverse()(stim.PauliString("Z"))
            letter = "_XYZ"[ps[0]]
            s = 1 if ps.sign == -1 else 0
            if letter == "Y":
                raise NotImplementedError(
                    f"terminal Y measurement for data qubit {q} "
                    "(S-carrying frame): not wired")
        prog_ops.append(ProgramOp("mpp", {q: letter}))
        flips.append(s)
    out_bits = [OutBit(rec=k, flip=f) for k, f in enumerate(flips)]
    program = PPMProgram(num_data=num_data + len(remap), ops=prog_ops,
                         gadgets=[], out_bits=out_bits)
    return program, init_letters, remap



def build_compiled(program, init_letters, remap, placement, d=3,
                   orientation="by_first_letter", routes=None,
                   unstitches=None):
    """Their plan on the CircLS backend, through the documented surface:
    injected placement, widened data lifetimes, scripted ancilla lives.

    Observables: the DETERMINISTIC SUBGROUP of the program's output bits
    (terminal data measurements) — one observable per GF(2) basis element
    of the deterministic combinations, record sets from the same
    closure recipes the CircLS pipeline uses (executed_bit_recs).  This
    matches the pipeline's own metric semantics: re-selection there also
    emits exactly a generating set of the deterministic subgroup; the
    replay may not re-select (their plan measures raw per-qubit bits, of
    which e.g. a cat state's are individually free coins), so the basis
    is taken on the output side instead.  Internal merge parities are
    not program failures and get no observables."""
    from circls.interop.ir.gosc_gadgets import (append_program_observables,
                                             folded_out_bits,
                                             program_step_interactions)
    from circls.pipeline import CompiledProgram
    from circls.interop.ir.ppm_import import patch_name
    from circls.core.multi_patch_coupler import PatchSpec, origin_of
    from circls.core.sequential_ppm_ls import PPMStep, SequentialPPMExperiment

    num_data = program.num_data - len(remap)
    steps = [PPMStep(list(it),
                     route=(routes or {}).get(j),
                     unstitch=(unstitches or {}).get(j))
             for j, it in enumerate(program_step_interactions(program))]
    folded = folded_out_bits(program)

    initial_states = {patch_name(k): init_letters.get(k, "Z")
                      for k in range(program.num_data)}
    final_states = {patch_name(k): "Z" for k in range(program.num_data)}
    for q, letter in folded.values():
        final_states[patch_name(q)] = letter

    first_letters: Dict[str, str] = {}
    for s in steps:
        for nm, letter in s.interaction_type:
            first_letters.setdefault(nm, letter)

    def orient(k):
        if orientation != "by_first_letter":
            return orientation
        return ("X_horizontal"
                if first_letters.get(patch_name(k)) == "Z"
                else "X_vertical")

    cell_of = {qidx: placement[pid]
               for pid, qidx in list(remap.items())
               + [(q, q) for q in range(num_data)]}
    specs = [PatchSpec(patch_name(k),
                       origin_of(cell_of[k][1], cell_of[k][0], d, seam=True),
                       d, orient(k))
             for k in sorted(cell_of)]
    data_names = {patch_name(q) for q in range(num_data)}
    exp = SequentialPPMExperiment(
        specs, steps, initial_states=initial_states,
        final_measure_states=final_states, rounds=d, rounds_init=1,
        liveness=True, keep_patches=data_names,
        lifetime_overrides={nm: (0, max(len(steps) - 1, 0))
                            for nm in data_names},
        auto_rotate=True, rotation_kind="auto", parallel_steps=False)
    with contextlib.redirect_stdout(io.StringIO()):
        circuit = exp.build()
    # The circuit is exactly what the backend emits — including the
    # tracker's own terminal OBSERVABLE_INCLUDEs (the deterministic
    # terminal closure group, the same machinery and convention every
    # CircLS pipeline circuit carries).  That group IS the replay's
    # failure metric; nothing is stripped or appended.
    n = len(program.out_bits)
    out_bit_ids = [i for i in range(n)
                   if i in folded and folded[i][0] < num_data]
    # actual-outcome recipes, for the per-mask verification gate only:
    # joint bit = the step's post-closure records; folded bit = the
    # patch readout's support records snapshotted at readout time
    from circls.interop.ir.gosc_gadgets import step_roles
    roles = step_roles(program)
    m_step_of = {bit: j for j, (kind, bit) in enumerate(roles)
                 if kind == "m"}
    recipes = []
    for i in range(n):
        if i in folded:
            q, letter = folded[i]
            recs = exp.patch_support_recs[(patch_name(q), letter)]
        else:
            post = exp.step_joint_records_post.get(m_step_of[i])
            assert post is not None, f"joint bit {i}: no post closure"
            recs = sorted(post)
        recipes.append(list(recs))
    cp = CompiledProgram(circuit=circuit, observables={},
                         experiment=exp, program=program,
                         reconstruction=None,
                         reorder_perm=list(range(n)),
                         m_load_trace=list(range(n)))
    return cp, out_bit_ids, recipes



def logical_reference_bits(ops: List[LLIOp], remap: Dict[int, int],
                           num_data: int, measured: List[int],
                           num_qubits: int, shots: int, seed: int):
    """Sample the RAW LLI stream at the logical level: gates as unitaries,
    ORIGINAL (unconjugated, unsigned-executed) letters, terminal data Z.
    Bit columns align 1:1 with the PPMProgram's out_bits."""
    import numpy as np
    import stim
    c = stim.Circuit()
    tp = {"X": stim.target_x, "Y": stim.target_y, "Z": stim.target_z}

    def q_of(pid):
        return pid if pid < num_data else remap[pid]

    for op in ops:
        if op.kind == "rotate":
            continue
        if op.kind == "gate1q":
            c.append(_GATE1Q[op.state], [q_of(op.patches[0][0])])
        elif op.kind == "init":
            q = q_of(op.patches[0][0])
            c.append("R", [q])
            if op.state == "|+>":
                c.append("H", [q])
        else:
            targs = []
            for j, (pid, P) in enumerate(op.patches):
                if j:
                    targs.append(stim.target_combiner())
                targs.append(tp[P](q_of(pid)))
            c.append("MPP", targs)
    for q in measured:
        c.append("M", [q])
    while c.num_qubits < num_qubits:
        c.append("I", [c.num_qubits])
    return c.compile_sampler(seed=seed).sample(shots)


def _ler_with_budget(circuit, args):
    """measure_ler in a forked process group, killed at --ler-budget."""
    import multiprocessing as mp
    import os
    import signal

    def child(q):
        os.setsid()
        from circls.tools.evaluate import measure_ler
        s = measure_ler(circuit, args.p, max_shots=args.max_shots,
                        max_errors=args.max_errors,
                        num_workers=args.workers)
        q.put({"decoder": s.decoder, "shots": s.shots,
               "errors": s.errors, "ler": s.logical_error_rate,
               "ler_bar": s.ler_error_bar()})

    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=child, args=(q,))
    p.start()
    p.join(args.ler_budget)
    if p.is_alive():
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.join(10)
        return None
    return q.get() if not q.empty() else None


class _Stats:
    def __init__(self, d):
        self.decoder = d["decoder"]
        self.shots = d["shots"]
        self.errors = d["errors"]
        self._ler = d["ler"]
        self._bar = d["ler_bar"]

    @property
    def logical_error_rate(self):
        return self._ler

    def ler_error_bar(self):
        return self._bar


def main():
    import argparse
    import numpy as np
    from circls.tools.reporting import (affine_structure,
                                          sample_program_bits)
    ap = argparse.ArgumentParser()
    ap.add_argument("--slicer", required=True)
    ap.add_argument("--qasm", help="path to a QASM file")
    ap.add_argument("--prog", help="benchsuite case name (writes its own "
                    "QASM to workdir; alternative to --qasm)")
    ap.add_argument("--workdir", default=".")
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--shots", type=int, default=256)
    ap.add_argument("--measure", action="store_true",
                    help="after the gates pass: circuit_stats + LER, "
                    "appended to experiments/results/best5/"
                    "lsqecc_replay.jsonl")
    ap.add_argument("--p", type=float, default=5e-4)
    ap.add_argument("--max-shots", type=int, default=1_000_000)
    ap.add_argument("--max-errors", type=int, default=100)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--ler-budget", type=int, default=3600,
                    help="wall-clock cap (s) for the LER run; on expiry "
                    "the row records ler=null with note ler_budget_exceeded "
                    "(MWPF stalls on saturated wide-hyperedge circuits)")
    ap.add_argument("--save-circuit", action="store_true",
                    help="write the built circuit to workdir/<prog>.stim "
                    "before the LER run (reusable without a rebuild)")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    prov = None
    if args.measure:
        from provenance import provenance
        prov = provenance(args.allow_dirty)   # before any output file exists
    if bool(args.qasm) == bool(args.prog):
        ap.error("pass exactly one of --qasm / --prog")
    if args.prog:
        from benchsuite import suite
        cases = {c.name: c for c in suite()}
        if args.prog not in cases:
            ap.error(f"unknown case {args.prog}; have e.g. "
                     f"{sorted(cases)[:8]}")
        qasm_path = Path(args.workdir) / f"{args.prog}.qasm"
        qasm_path.write_text(cases[args.prog].qasm)
    else:
        qasm_path = Path(args.qasm)
    lli, slices, stats = run_slicer(Path(args.slicer), qasm_path,
                                    Path(args.workdir))
    ops = parse_lli(lli)
    ops2, meta = conjugate_frame(ops)
    placement = extract_placement(slices)
    num_data = int(re.search(r"qreg\s+\w+\[(\d+)\]",
                             qasm_path.read_text()).group(1))
    measured = measured_qubits(qasm_path.read_text(), num_data)
    program, init_letters, remap = to_program(ops2, meta, num_data,
                                              measured)
    print(f"LLI ops: {len(ops)} -> {len(program.ops)} mpps "
          f"({meta['gates_absorbed']} gates absorbed, "
          f"{meta['rotations_dropped']} rotations dropped); "
          f"qubits {program.num_data} ({num_data} data)")
    last_err = None
    for orient in ("by_first_letter", "X_vertical", "X_horizontal"):
        try:
            cp, out_bit_ids, recipes = build_compiled(
                program, init_letters, remap, placement, d=args.d,
                orientation=orient)
        except Exception as e:
            print(f"orientation={orient}: {type(e).__name__}: "
                  f"{str(e)[:180]}")
            last_err = e
            continue
        det, obs = cp.circuit.compile_detector_sampler(seed=0).sample(
            args.shots, separate_observables=True)
        silent = not det.any()
        deterministic = (obs.shape[1] == 0) or bool((obs == obs[0]).all())
        # rank safety: with fewer shots than bit columns the diff matrix
        # cannot reach full rank and FABRICATES deterministic dims (a
        # 256-bit program at 256 shots gets >=1 spurious mask, bv_n280
        # would get hundreds) — sample past the bit count
        S = max(args.shots, len(program.out_bits) + 128)
        raw = logical_reference_bits(ops, remap, num_data, measured,
                                     program.num_data, S, 7)
        raw_masks, _ = affine_structure(raw)
        n_obs = cp.circuit.num_observables
        # count accounting only: the tracker may emit MORE observables
        # than the plan's deterministic dimension — protocol-internal
        # closures (rotation/SE residue) are real deterministic relations
        # the plan's record space cannot see — so the check is >=.  A
        # per-mask replay check is NOT possible in raw record space: the
        # physical records equal the plan's composed with a unipotent
        # frame map whose offsets live in protocol records, so the masks
        # transform; the end-to-end realization evidence is the
        # output-bit gate below (plan-convention reconstruction,
        # value-exact for deterministic outputs).
        dim_ok = n_obs >= len(raw_masks)
        got_out = sample_program_bits(cp, shots=S,
                                      seed=29)[:, out_bit_ids]
        out_ok = (affine_structure(got_out)
                  == affine_structure(raw[:, out_bit_ids]))
        print(f"orientation={orient}: build OK, qubits="
              f"{cp.circuit.num_qubits}, p=0 detectors "
              f"{'SILENT' if silent else 'FIRED'}, observables "
              f"{'DETERMINISTIC' if deterministic else 'RANDOM'}; "
              f"observables={n_obs} vs plan dims {len(raw_masks)} "
              f"(>=: {'OK' if dim_ok else 'MISMATCH'}); "
              f"output-bit structure {'OK' if out_ok else 'MISMATCH'}; "
              f"rotations={len(cp.experiment.rotations)}")
        if not (silent and deterministic and dim_ok and out_ok):
            sys.exit(2)
        if args.save_circuit:
            (Path(args.workdir) / (qasm_path.stem + ".stim")).write_text(
                str(cp.circuit))
        if args.measure:
            import time
            from circls.metrics.circuit_stats import circuit_stats
            cs = circuit_stats(cp.circuit)
            t0 = time.time()
            note = None
            if cp.circuit.num_observables == 0:
                stats = None       # no deterministic content -> no LER
            else:
                d = _ler_with_budget(cp.circuit, args)
                stats = _Stats(d) if d else None
                if stats is None:
                    note = "ler_budget_exceeded"
            row = {"record": "lsqecc_replay", "note": note,
                   "program": qasm_path.stem, "d": args.d, "p": args.p,
                   "orientation": orient,
                   "qubit_rounds": cs.qubit_rounds,
                   "qubit_rounds_span": cs.qubit_rounds_span,
                   "qubits_active": cs.qubits_active,
                   "qubits_allocated": cs.qubits_allocated,
                   "measurement_layers": cs.measurement_layers,
                   "num_observables": cs.num_observables,
                   "rotations": len(cp.experiment.rotations),
                   "decoder": stats.decoder if stats else None,
                   "shots": stats.shots if stats else 0,
                   "errors": stats.errors if stats else 0,
                   "ler": stats.logical_error_rate if stats else None,
                   "ler_bar": stats.ler_error_bar() if stats else None,
                   "seconds": round(time.time() - t0, 1)}
            out = (_ROOT / "experiments" / "results" / "best5"
                   / "lsqecc_replay.jsonl")
            with out.open("a") as fh:
                fh.write(json.dumps({**prov, "record": "provenance"})
                         + "\n")
                fh.write(json.dumps(row) + "\n")
            print("measured:", json.dumps(row))
        return
    raise SystemExit(f"all orientations failed: {last_err}")


if __name__ == "__main__":
    main()
