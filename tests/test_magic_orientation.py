"""|+>-proxy gadget ancillas face the step that consumes them.

``mapping.magic_orientations`` picks each ancilla's birth orientation from
the position of its partners (Z through an E/W seam wants X_horizontal,
through a N/S seam X_vertical); ``compile_qasm(magic_proxy="X")`` applies
it, while the default ``magic_proxy="Y"`` keeps the Gidney birth pinned to
X_vertical and the compiled circuit unchanged."""
import contextlib
import io

import pytest

pytest.importorskip("nwqec")

import circls.pipeline as P
from circls.compiler.mapping import magic_orientations
from circls.tools.evaluate import verify

# a 2-qubit Clifford+T program: one gadget ancilla y0 consumed by one joint
# PPM; c[1] is deterministic (the T sits between the two CNOTs), so verify()
# has an observable to check the distance of
QASM_T = ('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\n'
          'h q[0];\ncx q[0], q[1];\nt q[1];\ncx q[0], q[1];\n'
          'measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n')


def test_rule_faces_the_partner():
    steps = [[("q0", "Z"), ("y0", "Z")]]
    east = {"q0": (3, 1), "y0": (1, 1)}
    north = {"q0": (1, 3), "y0": (1, 1)}
    diagonal = {"q0": (3, 3), "y0": (1, 1)}
    assert magic_orientations(east, steps, ["y0"]) == {"y0": "X_horizontal"}
    assert magic_orientations(north, steps, ["y0"]) == {"y0": "X_vertical"}
    assert magic_orientations(diagonal, steps, ["y0"]) == {"y0": "X_vertical"}


def test_rule_nearest_partner_weighs_most():
    steps = [[("q0", "Z"), ("q1", "X"), ("y0", "Z")]]
    cells = {"y0": (1, 1), "q0": (3, 1), "q1": (1, 7)}       # q0 east at 2, q1 north at 6
    assert magic_orientations(cells, steps, ["y0"]) == {"y0": "X_horizontal"}
    cells = {"y0": (1, 1), "q0": (7, 1), "q1": (1, 3)}       # q0 east at 6, q1 north at 2
    assert magic_orientations(cells, steps, ["y0"]) == {"y0": "X_vertical"}


def test_rule_only_touches_the_named_ancillas():
    steps = [[("q0", "Z"), ("y0", "Z")], [("q1", "Z"), ("y1", "Z")]]
    cells = {"q0": (3, 1), "y0": (1, 1), "q1": (5, 3), "y1": (5, 1)}
    assert magic_orientations(cells, steps, ["y1"]) == {"y1": "X_vertical"}


def test_rule_uses_the_ancilla_letter():
    # an X measurement on the ancilla through an E/W seam wants X_vertical
    steps = [[("q0", "Z"), ("y0", "X")]]
    cells = {"q0": (3, 1), "y0": (1, 1)}
    assert magic_orientations(cells, steps, ["y0"]) == {"y0": "X_vertical"}


def test_tied_vote_is_settled_by_the_exact_judge():
    # toffoli_n3's PPM 0 geometry: partners north (q1) and west (q2) at the
    # same distance tie the vote; the judge scores both orientations on the
    # consuming step and takes the strictly cheaper one
    from circls.compiler.assignment import ExactScorer, legal_axis
    steps = [[("q1", "Z"), ("q2", "X"), ("y0", "Z")]]
    cells = {"q0": (3, 3), "q1": (5, 3), "q2": (3, 5), "y0": (5, 5)}
    orients = {"q0": "X_horizontal", "q1": "X_horizontal", "q2": "X_vertical"}
    got = magic_orientations(cells, steps, ["y0"], orients=orients)["y0"]
    judge = ExactScorer([(2 * (k % 4) + 1, 2 * (k // 4) + 1) for k in range(16)])
    cost = {o: judge.step_cost([cells[p] for p, _ in steps[0]],
                               [legal_axis(l, o if p == "y0" else orients[p])
                                for p, l in steps[0]])
            for o in ("X_vertical", "X_horizontal")}
    assert cost["X_horizontal"] != cost["X_vertical"]
    assert got == min(cost, key=cost.get)


def test_tied_vote_and_tied_judge_keep_the_default():
    steps = [[("q0", "Z"), ("y0", "Z")]]
    cells = {"q0": (3, 3), "y0": (1, 1)}      # diagonal partner, symmetric board
    assert magic_orientations(cells, steps, ["y0"], orients={"q0": "X_horizontal"}) == {"y0": "X_vertical"}


def _compile(**kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return P.compile_qasm(QASM_T, distance=3, t_as_s=True, **kw)


def _orients(cp):
    return {s.name: s.orientation for s in cp.experiment.patches
            if s.name.startswith("y")}


@pytest.mark.parametrize("assignment", ["row_major", "optimized"])
def test_x_proxy_orientation_matches_the_rule_and_verifies(assignment):
    cp = _compile(magic_proxy="X", assignment=assignment)
    init = cp.experiment.initial_states
    assert all(v == "X" for k, v in init.items() if k.startswith("y"))
    steps = [s.interaction_type for s in cp.experiment.ppm_sequence]
    want = magic_orientations(cp.placement, steps, sorted(_orients(cp)))
    assert _orients(cp) == want
    rep = verify(cp)
    assert rep.silent and rep.deterministic and rep.distance is True


def test_y_proxy_default_stays_pinned_and_rejects_override():
    cp = _compile()
    assert set(_orients(cp).values()) == {"X_vertical"}
    assert all(v == "Y" for k, v in cp.experiment.initial_states.items()
               if k.startswith("y"))
    with pytest.raises(ValueError, match="cannot override"):
        _compile(orientation={"y0": "X_horizontal"})


def test_x_proxy_accepts_an_explicit_override():
    cp = _compile(magic_proxy="X", orientation={"y0": "X_horizontal"})
    assert _orients(cp) == {"y0": "X_horizontal"}
    cp = _compile(magic_proxy="X", orientation={"y0": "X_vertical"})
    assert _orients(cp) == {"y0": "X_vertical"}


def test_bad_proxy_letter_is_rejected():
    with pytest.raises(ValueError, match="magic_proxy"):
        _compile(magic_proxy="Z")
