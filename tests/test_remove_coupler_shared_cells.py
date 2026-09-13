"""``QECSystem.remove_coupler`` frees only the cells no other registration
references, and un-declares the freed ones.

Regression for the batch-two KeyError failures (2026-09-07): fredkin_n3 under
the ``static`` configuration (liveness off, ``first_use_init=False``) at every
d = 3..11 and multiply_n13 d=3 under ``reselect_only``.  With
``first_use_init=False`` every step's coupler is registered UP FRONT, so a
later registration reuses the earlier couplers' DORMANT indices (add_patch's
dormant-index reuse) and a runtime litinski rotation grows a data patch onto a
dormant coupler cell.  When ``_invalidate_downstream_registration`` then
removed a stale up-front coupler, ``remove_coupler`` pruned ``qubit_coords``,
the owner maps and the category sets for EVERY index of the coupler's
``local_to_global_map`` — a survivor's live syndrome qubit included — and the
next SE round raised ``KeyError`` on it (fredkin static d=3: index 449, q0's
post-rotation boundary lobe; multiply reselect_only d=3: index 1093).  With
the pruning restricted to the coupler's EXCLUSIVE cells, those cells were
still declared by the QUBIT_COORDS written at setup and never touched, and
the S4==S5 self-check rejected the circuit (108 idle qubits at fredkin d=3),
so the freed declarations are retracted too — from the front block, keeping
later define-by-run declarations in that block.
"""
import contextlib
import io
import re

import pytest
import stim

from lightstim.qec_code.surface_code.rotated import RotatedSurfaceCode
from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.ir.builder import CircuitBuilder


# ── unit level: QECSystem.remove_coupler on shared cells ─────────────────────

def _coupler(system, name, offset, d=3):
    """register_coupler's registration half with a plain patch as the body"""
    system.add_patch(RotatedSurfaceCode(distance=d), offset=offset, name=name,
                     is_active=False)
    system.coupler_patches[name] = system.patches[name][0]
    return set(system.local_to_global_map[name].values())


def _idx(system, name):
    return set(system.local_to_global_map[name].values())


def _uids(system, name):
    return set(system.patches[name][0]._registered_stabilizer_uids)


def _in_category(system, q):
    return q in system.data_indices or q in system.syndrome_indices


def test_remove_coupler_keeps_cells_shared_with_a_surviving_coupler():
    system = QECSystem()
    system.add_patch(RotatedSurfaceCode(distance=3), name="Q", offset=(0, 0))
    c1 = _coupler(system, "C1", (10, 0))
    c2 = _coupler(system, "C2", (10, 0))        # same cells: dormant reuse
    assert c2 == c1
    uids = _uids(system, "C1")
    assert uids == _uids(system, "C2")           # signature-deduplicated records
    assert all(system.stabilizers[u]["patch_name"] == "C2" for u in uids)

    system.remove_coupler("C2")

    assert "C2" not in system.coupler_patches
    assert "C2" not in system.local_to_global_map
    assert _idx(system, "C1") == c1
    for q in c1:
        assert q in system.qubit_coords            # the F3a/F3b KeyError site
        assert system.index_map[system.qubit_coords[q]] == q
        assert system.index_to_owner_map[q] == "C1"
        assert system.coord_to_owner_map[system.qubit_coords[q]] == "C1"
        assert _in_category(system, q)
    # the shared records go back to the survivor instead of being orphaned
    assert all(system.stabilizers[u]["patch_name"] == "C1" for u in uids)
    assert not system.active_stabilizer_indices & uids
    # and a later registration on the same cells reuses cells and records
    c3 = _coupler(system, "C3", (10, 0))
    assert c3 == c1
    assert _uids(system, "C3") == uids


def test_remove_coupler_keeps_cells_taken_over_by_a_data_patch():
    # the rotation case: a data patch grown onto a dormant coupler cell owns
    # the index now; removing the stale coupler must not strip it
    system = QECSystem()
    system.add_patch(RotatedSurfaceCode(distance=3), name="Q", offset=(0, 0))
    c1 = _coupler(system, "C1", (10, 0))
    system.add_patch(RotatedSurfaceCode(distance=3), name="P", offset=(10, 0))
    assert _idx(system, "P") == c1
    p_uids = _uids(system, "P")
    assert p_uids <= system.active_stabilizer_indices

    system.remove_coupler("C1")

    assert _idx(system, "P") == c1
    for q in c1:
        assert q in system.qubit_coords
        assert system.index_to_owner_map[q] == "P"
        assert system.coord_to_owner_map[system.qubit_coords[q]] == "P"
        assert _in_category(system, q)
    assert p_uids <= system.active_stabilizer_indices
    assert all(system.stabilizers[u]["patch_name"] == "P" for u in p_uids)


def test_remove_coupler_frees_exclusive_cells_as_before():
    system = QECSystem()
    system.add_patch(RotatedSurfaceCode(distance=3), name="Q", offset=(0, 0))
    q = _idx(system, "Q")
    c1 = _coupler(system, "C1", (10, 0))
    system.remove_coupler("C1")
    assert not (c1 & set(system.qubit_coords))
    assert not (c1 & set(system.index_to_owner_map))
    assert not any(_in_category(system, x) for x in c1)
    assert set(system.index_map.values()) == q
    c2 = _coupler(system, "C2", (10, 0))         # fresh indices, old ones orphaned
    assert not (c2 & c1)


# ── builder level: retracting the freed cells' declarations ──────────────────

def _declared(circuit):
    out = set()
    for inst in circuit:
        if isinstance(inst, stim.CircuitInstruction) and inst.name == "QUBIT_COORDS":
            out.update(t.value for t in inst.targets_copy())
    return out


def _front_block_holds_every_declaration(circuit):
    """every QUBIT_COORDS sits in the leading run of QUBIT_COORDS"""
    in_front = True
    for inst in circuit:
        is_qc = isinstance(inst, stim.CircuitInstruction) and inst.name == "QUBIT_COORDS"
        if not is_qc:
            in_front = False
        elif not in_front:
            return False
    return True


def _system_with_builder():
    system = QECSystem()
    system.add_patch(RotatedSurfaceCode(distance=3), name="Q", offset=(0, 0))
    c1 = _coupler(system, "C1", (10, 0))
    tracker = SyndromeTracker(num_qubits=system.num_qubits,
                              expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=True)
    system.register_tracker(tracker)
    system.register_builder(builder)
    builder.write_coordinates()                  # setup declared C1's cells
    builder.circuit.append("H", sorted(_idx(system, "Q")))
    return system, builder, c1


def test_remove_coupler_retracts_the_freed_cells_declarations():
    system, builder, c1 = _system_with_builder()
    q = _idx(system, "Q")
    assert _declared(builder.circuit) == q | c1

    system.remove_coupler("C1")                  # C1's cells are exclusive

    assert _declared(builder.circuit) == q == set(system.qubit_coords)
    assert not (c1 & builder._written_coord_indices)
    assert builder.circuit[-1].name == "H"
    # a later registration still lands its declarations in the front block
    c2 = _coupler(system, "C2", (20, 0))
    assert _declared(builder.circuit) == set(system.qubit_coords) == q | c2
    assert _front_block_holds_every_declaration(builder.circuit)
    assert builder.circuit[-1].name == "H"


def test_remove_coupler_keeps_declarations_of_shared_cells():
    system, builder, c1 = _system_with_builder()
    q = _idx(system, "Q")
    assert _coupler(system, "C2", (10, 0)) == c1
    before = str(builder.circuit)
    system.remove_coupler("C2")                  # every cell survives in C1
    assert str(builder.circuit) == before
    assert _declared(builder.circuit) == q | c1 == set(system.qubit_coords)


def test_retract_coordinates_is_a_no_op_for_undeclared_indices():
    system, builder, c1 = _system_with_builder()
    before = str(builder.circuit)
    assert builder.retract_coordinates(set()) == 0
    assert builder.retract_coordinates({10 ** 6, 10 ** 6 + 1}) == 0
    assert str(builder.circuit) == before


# ── end to end: the batch-two cells that raised KeyError ─────────────────────
# TopoLS's tt2 programs (docs/benchmark/<name>_tt2.qasm) with the terminal
# register measurement appended — the Table 2 / batch-two input text; neither
# contains an s/sdg statement, so the "s-deletion" convention is a no-op here.

FREDKIN_N3_TT2 = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
x q[0];
x q[1];
cx q[2], q[1];
cx q[0], q[1];
h q[2];
t q[0];
t q[1];
t q[2];
cx q[2], q[1];
cx q[0], q[2];
t q[1];
cx q[0], q[1];
t q[2];
t q[1];
cx q[0], q[2];
cx q[2], q[1];
t q[1];
h q[2];
cx q[2], q[1];
creg c[3];
measure q -> c;
"""

MULTIPLY_N13_TT2 = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[13];
x q[0];
x q[1];
x q[2];
x q[4];
h q[5];
cx q[0], q[5];
t q[5];
cx q[2], q[5];
t q[5];
cx q[0], q[5];
t q[5];
cx q[2], q[5];
t q[0];
t q[5];
cx q[2], q[0];
t q[2];
t q[0];
cx q[2], q[0];
h q[5];
h q[6];
cx q[1], q[6];
t q[6];
cx q[2], q[6];
t q[6];
cx q[1], q[6];
t q[6];
cx q[2], q[6];
t q[1];
t q[6];
cx q[2], q[1];
t q[2];
t q[1];
cx q[2], q[1];
h q[6];
h q[7];
cx q[0], q[7];
t q[7];
cx q[3], q[7];
t q[7];
cx q[0], q[7];
t q[7];
cx q[3], q[7];
t q[0];
t q[7];
cx q[3], q[0];
t q[3];
t q[0];
cx q[3], q[0];
h q[7];
h q[8];
cx q[1], q[8];
t q[8];
cx q[3], q[8];
t q[8];
cx q[1], q[8];
t q[8];
cx q[3], q[8];
t q[1];
t q[8];
cx q[3], q[1];
t q[3];
t q[1];
cx q[3], q[1];
h q[8];
h q[9];
cx q[0], q[9];
t q[9];
cx q[4], q[9];
t q[9];
cx q[0], q[9];
t q[9];
cx q[4], q[9];
t q[0];
t q[9];
cx q[4], q[0];
t q[4];
t q[0];
cx q[4], q[0];
h q[9];
h q[10];
cx q[1], q[10];
t q[10];
cx q[4], q[10];
t q[10];
cx q[1], q[10];
t q[10];
cx q[4], q[10];
t q[1];
t q[10];
cx q[4], q[1];
t q[4];
t q[1];
cx q[4], q[1];
h q[10];
cx q[6], q[11];
cx q[7], q[11];
cx q[8], q[12];
cx q[9], q[12];
creg c[13];
measure q -> c;
"""

# batch two's configurations (experiments/ablation.py compile_kwargs):
# "static" = compile_kwargs("no_live") + first_use_init=False (tclass_dscaling);
# "reselect_only" = compile_kwargs("reselect_only") (multiply is a
# scheduling-unfaithful case, so step_scheduling is off there anyway)
STATIC_KW = dict(assignment="optimized", measure_reduction=True,
                 first_use_init=False, step_scheduling=True, parallel_steps=True)
RESELECT_ONLY_KW = dict(assignment="row_major", measure_reduction=True,
                        first_use_init=False, step_scheduling=False,
                        parallel_steps=False)

_G1 = {"h": "H", "s": "S", "sdg": "S_DAG", "x": "X", "z": "Z", "y": "Y"}
_G2 = {"cx": "CX", "cz": "CZ", "swap": "SWAP"}
_SKIP = {"openqasm", "include", "qreg", "creg", "measure", "barrier",
         "t", "tdg", "s", "sdg"}


def _gf2_rank(rows):
    basis = {}
    rank = 0
    for r in rows:
        v = int("".join(map(str, r)), 2) if r else 0
        while v:
            b = v.bit_length() - 1
            if b in basis:
                v ^= basis[b]
            else:
                basis[b] = v
                rank += 1
                break
    return rank


def _skeleton_rank(qasm, n):
    """number of deterministic Z-parity output bits of the program with every
    t/tdg/s/sdg deleted, run on |0..0> — the |+>-proxy program's skeleton
    (batch two's skeleton_ranks()[0], an own stim tableau)"""
    c = stim.Circuit()
    c.append("I", range(n))
    for line in qasm.splitlines():
        for st in line.split("//")[0].split(";"):
            st = st.strip()
            if not st:
                continue
            head = st.split()[0].lower()
            if head in _SKIP:
                continue
            args = [int(a) for a in re.findall(r"\[(\d+)\]", st)]
            c.append(_G1[head] if head in _G1 else _G2[head], args)
    rows = [c.to_tableau().z_output(i) for i in range(n)]
    return n - _gf2_rank([[1 if rows[i][j] in (1, 2) else 0 for j in range(n)]
                          for i in range(n)])


def _observable_rank(circ):
    """GF(2) rank of the OBSERVABLE_INCLUDE record sets (distinct observables)"""
    tot = 0
    recsets = {}
    for inst in circ.flattened():
        if inst.name.startswith("M"):
            tot += (sum(1 for t in inst.targets_copy() if not t.is_combiner)
                    if inst.name == "MPP" else len(inst.targets_copy()))
        if inst.name == "OBSERVABLE_INCLUDE":
            j = int(inst.gate_args_copy()[0])
            recsets.setdefault(j, set())
            for t in inst.targets_copy():
                recsets[j] ^= {tot + t.value}
    vecs = [recsets[j] for j in sorted(recsets)]
    if not vecs:
        return 0
    allrec = sorted(set().union(*vecs))
    return _gf2_rank([[1 if r in v else 0 for r in allrec] for v in vecs])


def _compile_xproxy(monkeypatch, qasm, d, kw):
    """batch two's recipe: compile_qasm(..., t_as_s=True) with the X-state
    proxy (every gadget ancilla born |+> instead of |Y>), remove_coupler spied"""
    import circls.pipeline as P
    orig = P.to_experiment_inputs

    def xproxy(*a, **k):
        specs, steps, init, final, prog = orig(*a, **k)
        init = {nm: ("X" if v == "Y" else v) for nm, v in init.items()}
        return specs, steps, init, final, prog
    monkeypatch.setattr(P, "to_experiment_inputs", xproxy)
    removed = []
    real_remove = QECSystem.remove_coupler

    def spy(self, name):
        removed.append(name)
        return real_remove(self, name)
    monkeypatch.setattr(QECSystem, "remove_coupler", spy)
    with contextlib.redirect_stdout(io.StringIO()):
        cp = P.compile_qasm(qasm, distance=d, t_as_s=True, **kw)
    return cp, removed


def _check_compiled(cp, qasm, n):
    from circls.metrics.experiment_stats import experiment_stats
    circ = cp.circuit
    det, obs = circ.compile_detector_sampler(seed=1).sample(
        64, separate_observables=True)
    assert not det.any(), "p=0: a detector fired"
    assert all((obs[:, j] == obs[0, j]).all() for j in range(obs.shape[1])), \
        "p=0: an observable is not deterministic"
    experiment_stats(cp.experiment, circ)       # the S4==S5 self-check
    assert _observable_rank(circ) == _skeleton_rank(qasm, n)
    assert _front_block_holds_every_declaration(circ)
    circ.detector_error_model()


# The layout on which the 2026-09-07 failure was observed: fredkin_n3 under
# STATIC_KW as the optimizer placed it before its spectral start got a
# deterministic tie-break (2026-09-12).  Interchangeable |Y> patches made the
# optimizer's choice depend on the BLAS kernel's last-bit rounding, so this
# layout came out on AVX-512 machines only and CI (no AVX-512) never entered
# the revocation path.  The test is about remove_coupler, not about the
# optimizer, so the layout is pinned here (placement= replaces the
# assignment search; orientation= is what the optimizer had planned for the
# data patches -- the |Y> patches' orientation is protocol-fixed).
FREDKIN_STATIC_PLACEMENT = {
    "q0": (3, 5), "q1": (5, 3), "q2": (3, 3), "y0": (7, 3), "y1": (7, 5),
    "y2": (1, 5), "y3": (3, 7), "y4": (7, 1), "y5": (1, 1), "y6": (5, 7),
    "y7": (5, 5), "y8": (3, 1), "y9": (1, 3), "y10": (5, 1)}
FREDKIN_STATIC_ORIENTATION = {"q0": "X_horizontal", "q1": "X_vertical", "q2": "X_vertical"}
FREDKIN_STATIC_PINNED_KW = {**{k: v for k, v in STATIC_KW.items() if k != "assignment"},
                            "placement": FREDKIN_STATIC_PLACEMENT,
                            "orientation": FREDKIN_STATIC_ORIENTATION}


def test_fredkin_n3_static_compiles_after_a_downstream_coupler_is_revoked(monkeypatch):
    cp, removed = _compile_xproxy(monkeypatch, FREDKIN_N3_TT2, 3, FREDKIN_STATIC_PINNED_KW)
    # the failing path: an unplanned rotation revoked an up-front coupler
    assert removed, "no coupler was revoked: the regression path was not exercised"
    assert cp.placement == FREDKIN_STATIC_PLACEMENT
    _check_compiled(cp, FREDKIN_N3_TT2, 3)


@pytest.mark.slow
def test_multiply_n13_reselect_only_compiles_after_downstream_couplers_are_revoked(monkeypatch):
    cp, removed = _compile_xproxy(monkeypatch, MULTIPLY_N13_TT2, 3, RESELECT_ONLY_KW)
    assert removed, "no coupler was revoked: the regression path was not exercised"
    _check_compiled(cp, MULTIPLY_N13_TT2, 13)
