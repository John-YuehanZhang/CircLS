"""Public evaluation API: ``verify`` and ``measure_ler``.

The two calls that complete the compile story::

    out = compile_qasm(qasm, distance=3)
    report = verify(out)                       # circuit-level checks
    stats = measure_ler(out, p=1e-3)           # real logical error rate

``verify`` runs the same checks the test suite runs: no detector
triggers at p = 0, the program observables are deterministic, the
graphlike distance stays d (on circuits small enough to search), and
the compiled program is logically right in two record-free senses: a
logical-level simulation of the expanded PPM program, with the
compiler's own correction and reconstruction rules, carries the same
affine structure as a stabilizer simulation of the source QASM
(``logical_program``), and the noise-free circuit realises the
program's logical state before its terminal readout, every expected
logical stabilizer peeking at |<S>| = 1 (``logical_state``).  None of
these compares the value of a physical outcome with the source truth:
a Pauli-frame error (a logical Pauli inserted into the circuit) passes
all four checks.

``measure_ler`` injects the uniform circuit-level noise pass used
throughout the paper's measurements and hands the noisy circuit to
the vendored LightStim ``SimulationPipeline`` (PyMatching by
default; pass ``decoder='mwpf'`` for circuits whose error model does
not decompose into graphlike components).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import stim

# ── uniform circuit-level noise (the paper's shared noise pass) ──────────────

_1Q = {"H", "X", "Y", "Z", "S", "S_DAG", "SQRT_X", "SQRT_X_DAG",
       "SQRT_Y", "SQRT_Y_DAG", "C_XYZ", "C_ZYX", "H_XY", "H_XZ", "H_YZ", "I"}
_2Q = {"CX", "CY", "CZ", "XCX", "XCY", "XCZ", "YCX", "YCY", "YCZ",
       "SWAP", "ISWAP", "ISWAP_DAG", "CNOT", "ZCX", "ZCY", "ZCZ",
       "SQRT_XX", "SQRT_XX_DAG", "SQRT_YY", "SQRT_YY_DAG",
       "SQRT_ZZ", "SQRT_ZZ_DAG"}
_MEAS = {"M": "M", "MZ": "MZ", "MX": "MX", "MY": "MY",
         "MR": "MR", "MRZ": "MRZ", "MRX": "MRX", "MRY": "MRY"}
_RESET = {"R": "X_ERROR", "RZ": "X_ERROR", "RX": "Z_ERROR", "RY": "X_ERROR"}
_ANNOT = {"DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS",
          "TICK", "MPAD"}


def inject_uniform_noise(circuit: stim.Circuit, p: float) -> stim.Circuit:
    """Uniform circuit-level depolarizing noise at rate ``p``.

    After every 1q gate DEPOLARIZE1, after every 2q gate DEPOLARIZE2,
    measurement flip p (``M(p)``), reset flip p (X_ERROR after R,
    Z_ERROR after RX), and DEPOLARIZE1 in every TICK span, including
    empty spans, on every qubit that is gated somewhere in the circuit
    but not in that span (qubits that never appear in a gate get no
    noise).  This is the model of experiments/noise_inject.py, the pass
    behind every LER in the paper, and the two give byte-identical
    output on circuits that carry no noise, no inverted (``!``)
    measurement targets and no Pauli-product measurements; on those
    inputs noise_inject.py raises while this pass copies the
    instruction through without adding noise."""
    flat = circuit.flattened()
    gated = set()
    for inst in flat:
        if inst.name not in _ANNOT:
            for t in inst.targets_copy():
                if t.is_qubit_target:
                    gated.add(t.value)
    out = stim.Circuit()
    moment = []

    def flush():
        busy = set()
        for inst in moment:
            nm = inst.name
            ts = inst.targets_copy()
            qs = [t.value for t in ts if t.is_qubit_target]
            if nm not in _ANNOT:
                busy.update(qs)
            if nm in _MEAS:
                out.append(_MEAS[nm], ts, p)
            else:
                out.append(inst)
                if nm in _1Q:
                    out.append("DEPOLARIZE1", qs, p)
                elif nm in _2Q:
                    out.append("DEPOLARIZE2", qs, p)
                elif nm in _RESET:
                    out.append(_RESET[nm], qs, p)
        idle = sorted(gated - busy)
        if idle:
            out.append("DEPOLARIZE1", idle, p)
        moment.clear()

    for inst in flat:
        if inst.name == "TICK":
            flush()
            out.append("TICK")
        else:
            moment.append(inst)
    flush()
    return out


# ── verify ───────────────────────────────────────────────────────────────────

@dataclass
class VerifyReport:
    silent: bool                       # no detector triggers at p = 0
    deterministic: bool                # program observables are constant
    distance: Optional[bool]           # graphlike distance == d (None: skipped)
    logical: Optional[bool]            # logical_program AND logical_state
    #                                    (None: neither could run)
    detail: str = ""
    logical_program: Optional[bool] = None   # program simulation == source
    #                                          truth (None: no source QASM)
    logical_state: Optional[bool] = None     # circuit realises the program's
    #                                          logical state (None: skipped)

    @property
    def ok(self) -> bool:
        return (self.silent and self.deterministic
                and self.distance is not False
                and self.logical is not False)

    @property
    def skipped(self) -> list:
        """Names of the checks that did NOT run (ok treats them as
        passing).  distance skips above DISTANCE_DETECTOR_CAP;
        logical_program skips when the compile carried no source QASM
        (the compile_ppm_sequence entry); logical_state skips when the
        terminal readout cannot be cut at an instruction boundary."""
        out = []
        if self.distance is None:
            out.append("distance")
        if self.logical_program is None:
            out.append("logical_program")
        if self.logical_state is None:
            out.append("logical_state")
        return out

    def __repr__(self):
        base = (f"VerifyReport(ok={self.ok}, silent={self.silent}, "
                f"deterministic={self.deterministic}, "
                f"distance={self.distance}, logical={self.logical}")
        if self.logical is not None:
            base += (f" [program={self.logical_program}, "
                     f"state={self.logical_state}]")
        if self.skipped:
            base += f", skipped={self.skipped}"
        if self.detail:
            base += f", detail={self.detail!r}"
        return base + ")"


#: distance search cost grows quickly; above this many detectors the
#: check reports None ("skipped") unless forced
DISTANCE_DETECTOR_CAP = 5000


def verify(compiled, shots: int = 256, seed: int = 0,
           distance_p: float = 1e-3,
           force_distance: bool = False) -> VerifyReport:
    """Circuit-level verification of a :class:`CompiledProgram`."""
    c = compiled.circuit
    det, obs = c.compile_detector_sampler(seed=seed).sample(
        shots, separate_observables=True)
    silent = not det.any()
    deterministic = (obs.shape[1] == 0) or bool((obs == obs[0]).all())
    notes = []

    distance = None
    if not force_distance and c.num_detectors > DISTANCE_DETECTOR_CAP:
        # never skip silently: the d = 5 t_as_s distance gate went unrun
        # for weeks because nothing surfaced this branch (the sub-distance
        # seam defect hid behind it, found 2026-08-21)
        print(f"verify: distance check SKIPPED "
              f"({c.num_detectors} detectors > cap {DISTANCE_DETECTOR_CAP}; "
              f"pass force_distance=True to run it)")
    if force_distance or c.num_detectors <= DISTANCE_DETECTOR_CAP:
        try:
            noisy = inject_uniform_noise(c, distance_p)
            noisy.detector_error_model(decompose_errors=True)
            d = compiled.experiment.patches[0].distance
            distance = len(noisy.shortest_graphlike_error()) == d
        except ValueError as e:
            notes.append(f"distance check skipped: {str(e)[:80]}")
    else:
        notes.append(f"distance check skipped: {c.num_detectors} detectors "
                     f"> cap {DISTANCE_DETECTOR_CAP}")

    # logical correctness, two record-free checks (reporting.py):
    #   program: the expanded PPM program, simulated at the logical level
    #            with the compiler's own correction/reconstruction rules,
    #            has the affine structure of the source program;
    #   state:   the noise-free circuit realises the program's logical
    #            state before the terminal readout (every expected
    #            logical stabilizer peeks at |<S>| = 1).
    from circls.tools.reporting import (logical_program_structure,
                                          logical_state_check,
                                          source_truth_structure)
    logical_program = None
    qasm = getattr(compiled, "source_qasm", None)
    if qasm is not None:
        truth = source_truth_structure(qasm, shots=shots, seed=seed)
        got = logical_program_structure(compiled, shots=shots, seed=seed)
        logical_program = got == truth
        if not logical_program:
            notes.append("logical_program: the program's affine structure "
                         "differs from the source program's")
    state = logical_state_check(compiled)
    logical_state = state.ok
    if state.detail:
        notes.append(f"logical_state: {state.detail}")
    ran = [x for x in (logical_program, logical_state) if x is not None]
    logical = all(ran) if ran else None

    return VerifyReport(silent=silent, deterministic=deterministic,
                        distance=distance, logical=logical,
                        detail="; ".join(notes),
                        logical_program=logical_program,
                        logical_state=logical_state)


# ── measure_ler ──────────────────────────────────────────────────────────────

def measure_ler(compiled, p: float, decoder: Optional[str] = None,
                max_shots: int = 1_000_000, max_errors: int = 100,
                num_workers: int = 4):
    """Inject uniform noise at ``p`` and measure the logical error rate.

    Returns the LightStim ``SimulationStats`` (``.logical_error_rate``,
    ``.ler_error_bar()``, shot and error counts).  ``decoder`` is a
    LightStim ``DecoderConfig`` name; the default tries PyMatching and
    falls back to MWPF when stim cannot decompose the error model."""
    from lightstim.simulation.decoder_backend import (DecoderConfig,
                                                      SimulationPipeline)
    circuit = getattr(compiled, "circuit", compiled)
    noisy = inject_uniform_noise(circuit, p)
    names = [decoder] if decoder else ["pymatching", "mwpf"]
    last_err = None
    for name in names:
        try:
            if name == "pymatching":       # fail fast on non-graphlike DEMs
                noisy.detector_error_model(decompose_errors=True)
            pipeline = SimulationPipeline(
                decoder_config=DecoderConfig(name),
                max_shots=max_shots, max_errors=max_errors,
                num_workers=num_workers)
            return pipeline.run(noisy)
        except ValueError as e:
            if ("Failed to decompose" not in str(e) and
                    "observable ids larger than 63" not in str(e) and
                    "more than 64 terms" not in str(e)):
                raise
            last_err = e
    raise last_err
