"""Pass 2a oracle: the gadget-expanded PPM program (|Y> ancillas + all-X/Z
joint measurements + static XOR correction rules) must reproduce the exact
joint outcome distribution of the ORIGINAL Y-containing measurement sequence.

Dense simulation again (<= 8 qubits): init_y as the S.H unitary, every mpp as
a signed-projector branch; program bits assembled from records via the OutBit
and Gadget XOR rules.  This validates, end to end: the sweep, the |Y>-resource
Fig. 11b gadget form, the hand-derived correction condition c = b1^b2^kappa,
and the static frame-to-XOR compilation.
"""
import random

import numpy as np
import pytest

from circls.interop.ir.gosc_gadgets import (
    append_program_observables, expand_gadgets, step_roles, to_experiment_inputs)
from circls.interop.ir.pauli_algebra import y_free_sweep
from circls.interop.ir.pauli_ir import PauliCircuit, PauliOp
from tests.test_yfree_sweep import _joint_distribution, _mat

_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_S = np.diag([1, 1j]).astype(complex)


def _embed(u, q, n):
    out = np.array([[1]], dtype=complex)
    for i in range(n):
        out = np.kron(out, u if i == q else np.eye(2))
    return out


def _program_distribution(program):
    """Joint distribution over the program's out_bits, exact dense sim."""
    n = program.num_qubits
    psi0 = np.zeros(2 ** n, dtype=complex)
    psi0[0] = 1.0
    dist = {}

    def finish(recs, prob):
        r = recs
        cond = [r[g.rec_joint] ^ r[g.rec_x] ^ g.kappa for g in program.gadgets]
        bits = tuple(r[ob.rec] ^ ob.flip ^
                     (sum(cond[k] for k in ob.gadgets) % 2)
                     for ob in program.out_bits)
        dist[bits] = dist.get(bits, 0.0) + prob

    def run(i, psi, recs, prob):
        if prob < 1e-15:
            return
        if i == len(program.ops):
            finish(recs, prob)
            return
        op = program.ops[i]
        if op.kind == "init_y":
            (a, letter), = op.targets.items()
            assert letter == "Y"
            psi = _embed(_S @ _H, a, n) @ psi     # |0> -> |+i>
            run(i + 1, psi, recs, prob)
            return
        assert op.kind == "mpp"
        m = _mat(op.targets, n)
        for bit in (0, 1):
            proj = (np.eye(2 ** n) + (1 - 2 * bit) * m) / 2
            phi = proj @ psi
            p = float(np.vdot(phi, phi).real)
            if p > 1e-15:
                run(i + 1, phi / np.sqrt(p), recs + [bit], prob * p)

    run(0, psi0, [], 1.0)
    return dist


def _assert_program_equals_circuit(circuit: PauliCircuit):
    swept = y_free_sweep(circuit)
    program = expand_gadgets(swept)
    if program.num_qubits > 8:
        pytest.skip(f"dense oracle capped at 8 qubits, got {program.num_qubits}")
    d_ref = _joint_distribution(circuit)
    d_prog = _program_distribution(program)
    for k in set(d_ref) | set(d_prog):
        assert abs(d_ref.get(k, 0.0) - d_prog.get(k, 0.0)) < 1e-10, (
            f"bitstring {k}: original {d_ref.get(k, 0)} vs program "
            f"{d_prog.get(k, 0)}")
    return program


# ------------------------------------------------------------------ structure
def test_expand_structure_fig10():
    p = {0: "Y", 2: "Y", 3: "Z", 4: "Y", 5: "Y"}
    swept = y_free_sweep(PauliCircuit(6, [PauliOp("m", p, 1)]))
    prog = expand_gadgets(swept)
    assert [o.kind for o in prog.ops] == [
        "init_y", "mpp", "mpp", "init_y", "mpp", "mpp", "mpp"]
    g0, g1 = prog.gadgets
    assert g0.mask == {0: "Z"} and g0.ancilla == 6
    assert g1.mask == {2: "Z", 4: "Z", 5: "Z"} and g1.ancilla == 7
    assert (g0.rec_joint, g0.rec_x, g0.kappa) == (0, 1, 1)
    assert (g1.rec_joint, g1.rec_x, g1.kappa) == (2, 3, 1)
    # joint measurements carry Z on the ancilla; letters all X/Z
    assert prog.ops[1].targets == {0: "Z", 6: "Z"}
    assert prog.ops[4].targets == {2: "Z", 4: "Z", 5: "Z", 7: "Z"}
    (ob,) = prog.out_bits
    assert ob.rec == 4 and ob.flip == 1 and ob.gadgets == [0, 1]


def test_expand_rejects_bad_ops():
    # X masks and sign +1 became VALID with the t_as_s proxy front end
    # (kappa = 0); Y content, empty masks and pi/8 ops stay loud errors
    prog = expand_gadgets(PauliCircuit(2, [PauliOp("s", {0: "X"}, -1),
                                           PauliOp("s", {0: "Z"}, 1)]))
    assert [g.kappa for g in prog.gadgets] == [1, 0]
    with pytest.raises(ValueError, match="pi/4"):
        expand_gadgets(PauliCircuit(2, [PauliOp("s", {}, -1)]))
    with pytest.raises(ValueError, match="contains Y"):
        expand_gadgets(PauliCircuit(2, [PauliOp("m", {0: "Y"}, 1)]))
    with pytest.raises(ValueError, match="unsupported"):
        expand_gadgets(PauliCircuit(2, [PauliOp("t", {0: "Z"}, 1)]))


# --------------------------------------------------------------------- oracle
def test_program_matches_original_fig10():
    p = {0: "Y", 2: "Y", 3: "Z", 4: "Y", 5: "Y"}
    prog = _assert_program_equals_circuit(PauliCircuit(6, [PauliOp("m", p, 1)]))
    assert prog.num_ancilla == 2


def test_program_matches_original_cascade():
    """Mask re-introduces Y downstream; the second gadget's correction must
    land on the right later bits."""
    c = PauliCircuit(2, [
        PauliOp("m", {0: "Y"}, 1),
        PauliOp("m", {0: "X", 1: "Z"}, 1),
        PauliOp("m", {1: "X"}, -1),
    ])
    prog = _assert_program_equals_circuit(c)
    assert prog.num_ancilla >= 2


def test_program_matches_original_random():
    rng = random.Random(71)
    checked = 0
    for _ in range(40):
        n = rng.choice([2, 3])
        ops = [PauliOp("m",
                       {q: rng.choice("XYZ") for q in range(n)
                        if rng.random() < 0.75} or {0: rng.choice("XYZ")},
                       rng.choice([1, -1]))
               for _ in range(rng.randrange(1, 4))]
        c = PauliCircuit(n, ops)
        if expand_gadgets(y_free_sweep(c)).num_qubits > 8:
            continue
        _assert_program_equals_circuit(c)
        checked += 1
    assert checked >= 15


def test_program_end_to_end_from_nwqec():
    pytest.importorskip("nwqec")
    from circls.interop.nwqec.frontend import load_clifford_mpauli
    from tests.test_gosc_pauli_algebra import _random_clifford
    rng = random.Random(83)
    checked = 0
    for _ in range(10):
        qasm, _circ = _random_clifford(rng, 3, 15)
        loaded = load_clifford_mpauli(qasm)
        if expand_gadgets(y_free_sweep(loaded)).num_qubits > 8:
            continue
        _assert_program_equals_circuit(loaded)
        checked += 1
    assert checked >= 3


# --------------------------------------------------------- pass 2b structural
def test_experiment_inputs_structural():
    p = {0: "Y", 2: "Y", 3: "Z", 4: "Y", 5: "Y"}
    prog = expand_gadgets(y_free_sweep(PauliCircuit(6, [PauliOp("m", p, 1)])))
    specs, steps, init, final, back = to_experiment_inputs(prog)
    assert back is prog
    assert [s.interaction_type for s in steps] == [
        [("q0", "Z"), ("y0", "Z")],
        [("q2", "Z"), ("q4", "Z"), ("q5", "Z"), ("y1", "Z")],
        [("q0", "X"), ("q2", "X"), ("q3", "Z"), ("q4", "X"), ("q5", "X")],
    ]
    assert init["y0"] == init["y1"] == "Y"
    assert final["y0"] == final["y1"] == "X"     # gadget X readout = terminal
    assert all(l in ("X", "Z") for s in steps for _, l in s.interaction_type)
    assert {sp.name for sp in specs} == {nm for s in steps
                                         for nm, _ in s.interaction_type}


# ------------------------------------------- program-bit observable assembly
def test_program_observable_end_to_end_on_lattice():
    """Full-chain acceptance: repeated Y(x)Y measurement pair -> sweep ->
    gadget expansion -> real lattice experiment (2 data + 2 born-|Y> ancilla
    patches) -> append_program_observables.  Bit 0 is a genuinely free output
    (gated to None); bit 1's closure (the two outcomes must agree) becomes a
    stim observable that is noiseless-deterministic and holds fault distance
    d=3 under circuit-level noise everywhere."""
    import contextlib
    import io
    from circls.core.sequential_ppm_ls import PPMStep, SequentialPPMExperiment
    from circls.core.routed_multi_patch_ls import PatchSpec, origin_of
    from lightstim.noise.config import NoiseConfig

    D = 3
    prog = expand_gadgets(y_free_sweep(PauliCircuit(2, [
        PauliOp("m", {0: "Y", 1: "Y"}, 1),
        PauliOp("m", {0: "Y", 1: "Y"}, 1)])))
    assert [r for r in step_roles(prog)] == [
        ("gadget", 0), ("gadget", 1), ("m", 0), ("m", 1)]

    def sp(nm, a, b):
        return PatchSpec(nm, origin_of(a, b, D, seam=True), D, "X_vertical")

    px = [sp("q0", 0, 0), sp("q1", 2, 0), sp("y0", 0, 2), sp("y1", 2, 2)]
    seq = [PPMStep([("q0", "Z"), ("y0", "Z")]),
           PPMStep([("q1", "Z"), ("y1", "Z")]),
           PPMStep([("q0", "X"), ("q1", "X")]),
           PPMStep([("q0", "X"), ("q1", "X")])]
    states = {"q0": "Z", "q1": "Z", "y0": "Y", "y1": "Y"}
    finals = {"q0": "Z", "q1": "Z", "y0": "X", "y1": "X"}

    exp = SequentialPPMExperiment(px, seq, initial_states=states,
                                  final_measure_states=finals,
                                  rounds=D, rounds_init=1)
    with contextlib.redirect_stdout(io.StringIO()):
        c = exp.build()
    obs_map = append_program_observables(c, exp, prog)
    assert obs_map[0] is None            # free random output, not a parity
    assert obs_map[1] == c.num_observables - 1
    det, obs = c.compile_detector_sampler(seed=0).sample(
        2048, separate_observables=True)
    assert not det.any()
    for j in range(c.num_observables):   # every observable deterministic
        assert (obs[:, j] == obs[0, j]).all()

    expn = SequentialPPMExperiment(px, seq, initial_states=states,
                                   final_measure_states=finals,
                                   rounds=D, rounds_init=1,
                                   noise_params=NoiseConfig(
                                       p_1q=1e-3, p_2q=1e-3, p_meas=1e-3,
                                       p_reset=1e-3, p_idle=1e-3))
    with contextlib.redirect_stdout(io.StringIO()):
        cn = expn.build()
    append_program_observables(cn, expn, prog)
    cn.detector_error_model(decompose_errors=True)
    assert len(cn.shortest_graphlike_error()) == D


# ----------------------------------------------------------- M2: handshake
def test_append_observables_rejects_mispaired_program():
    """M2: a mispaired (exp, program) or edited step list must fail loudly,
    never silently mis-bind bits."""
    import contextlib, io
    from circls.core.sequential_ppm_ls import SequentialPPMExperiment
    prog_a = expand_gadgets(y_free_sweep(PauliCircuit(2, [
        PauliOp("m", {0: "Y", 1: "Z"}, 1), PauliOp("m", {0: "X", 1: "X"}, 1)])))
    prog_b = expand_gadgets(y_free_sweep(PauliCircuit(2, [
        PauliOp("m", {0: "Z", 1: "Z"}, 1)])))
    specs, steps, init, final, _ = to_experiment_inputs(prog_a)
    exp = SequentialPPMExperiment(specs, steps, initial_states=init,
                                  final_measure_states=final,
                                  rounds=3, rounds_init=1)
    with contextlib.redirect_stdout(io.StringIO()):
        c = exp.build()
    with pytest.raises(ValueError, match="mismatch"):
        append_program_observables(c, exp, prog_b)      # wrong program
    # reordered steps: same count, different content order
    from circls.core.sequential_ppm_ls import PPMStep
    exp2 = SequentialPPMExperiment(specs, list(reversed(steps)),
                                   initial_states=init,
                                   final_measure_states=final,
                                   rounds=3, rounds_init=1)
    with contextlib.redirect_stdout(io.StringIO()):
        c2 = exp2.build()
    with pytest.raises(ValueError, match="mismatch at step"):
        append_program_observables(c2, exp2, prog_a)
    # correct pairing still works
    assert append_program_observables(c, exp, prog_a) is not None


# --------------------------------------------------- m1: free-prefix peephole
def test_drop_free_prefix_shape_and_oracle():
    """Leading pure-Z masks on |0...0> are identity: dropping them must
    remove their gadgets and leave the joint distribution untouched;
    mid-stream 's' ops (after the first m) are kept."""
    import random
    from circls.interop.ir.pauli_algebra import drop_free_prefix
    # benchmark shape: one leading mask -> zero gadgets after the peephole
    swept = y_free_sweep(PauliCircuit(2, [PauliOp("m", {0: "Y", 1: "X"}, -1),
                                          PauliOp("m", {0: "Z", 1: "Z"}, 1)]))
    assert swept.ops[0].kind == "s"
    slim = drop_free_prefix(swept)
    assert all(op.kind == "m" for op in slim.ops[:1])
    assert expand_gadgets(slim).num_ancilla < expand_gadgets(swept).num_ancilla
    # cascade case keeps its MID-stream masks
    casc = y_free_sweep(PauliCircuit(2, [PauliOp("m", {0: "Y"}, 1),
                                         PauliOp("m", {0: "X", 1: "Z"}, 1)]))
    slim_c = drop_free_prefix(casc)
    assert any(op.kind == "s" for op in slim_c.ops)     # later mask survives
    # oracle: distribution equality on random Y-heavy streams
    rng = random.Random(313)
    checked = 0
    for _ in range(30):
        n = rng.choice([2, 3])
        ops = [PauliOp("m",
                       {q: rng.choice("XYZ") for q in range(n)
                        if rng.random() < 0.75} or {0: rng.choice("XYZ")},
                       rng.choice([1, -1]))
               for _ in range(rng.randrange(1, 4))]
        c = PauliCircuit(n, ops)
        prog = expand_gadgets(drop_free_prefix(y_free_sweep(c)))
        if prog.num_qubits > 8:
            continue
        d_ref = _joint_distribution(c)
        d_prog = _program_distribution(prog)
        for k in set(d_ref) | set(d_prog):
            assert abs(d_ref.get(k, 0.0) - d_prog.get(k, 0.0)) < 1e-10
        checked += 1
    assert checked >= 15
