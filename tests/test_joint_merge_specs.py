"""The unified wall generator vs the four frozen legacy hand specs, plus the
rule-table classifier and the multi-patch wall selector (joint_merge.py)."""
import pytest

from circls.core.joint_merge import (
    _wall_checks, assign_walls, classify_seam, patch_type, PatchView,
    SeamRuleError, TEXTBOOK, CONJUGATE)

from tests.test_wall_rules import same_type_wall_spec, mirror_mixed_wall_spec
from tests.test_mixed_wall import wall_spec as fig6_wall_spec
from tests.test_mixed_wall_horizontal import horizontal_wall_spec

pytestmark = pytest.mark.smoke


def _norm(spec):
    return sorted((tuple(B), tuple(sorted(p.items())), tuple(sorted(kf.items())))
                  for B, p, kf in spec)


@pytest.mark.parametrize("d", [3, 5])
@pytest.mark.parametrize("tag,legacy,phis,axis", [
    ("row2-FIG1", same_type_wall_spec, (0, 1), 'horizontal'),
    ("row3-mirror", mirror_mixed_wall_spec, (1, 1), 'horizontal'),
    ("row3-FIG6", fig6_wall_spec, (0, 0), 'horizontal'),
    ("row3-horizontal", horizontal_wall_spec, (0, 0), 'vertical'),
])
def test_generator_matches_frozen_legacy_spec(d, tag, legacy, phis, axis):
    # the four hand-derived wall specs are ONE closed form of
    # (seam axis, the two patches' checkerboard phases, d, origin)
    gen = _wall_checks(d, phis[0], phis[1], 1, 1, axis)
    assert _norm(gen) == _norm(legacy(d)), tag


@pytest.mark.parametrize("dx,dy", [(4, 0), (0, 8), (12, 4)])
def test_generator_translates(dx, dy):
    # kf/foot coords are absolute: a shifted origin is exactly the shifted spec
    d = 3
    base = _wall_checks(d, 0, 0, 1, 1, 'horizontal')
    moved = _wall_checks(d, 0, 0, 1 + dx, 1 + dy, 'horizontal')

    def shift(spec):
        return _norm([
            ((B[0] + dx, B[1] + dy),
             {(x + dx, y + dy): P for (x, y), P in p.items()},
             {k: ((v[0] + dx, v[1] + dy) if isinstance(v, tuple) else v)
              for k, v in kf.items()})
            for B, p, kf in spec])

    assert shift(base) == _norm(moved)


def _view(name, cell, d, typ, orientation, phase=0):
    """A synthetic PatchView at coarse cell ``cell`` with the given type."""
    pitch = 2 * d + 2
    origin = (cell[0] * pitch + 1, cell[1] * pitch + 1)
    from circls.core.joint_merge import (
        _reference_lobes)
    tb_ref, cj_ref = _reference_lobes(d)
    local = tb_ref if typ == TEXTBOOK else cj_ref
    lobes = frozenset((x + origin[0] - 1, y + origin[1] - 1) for x, y in local)
    lr = 'X' if orientation == 'X_horizontal' else 'Z'
    return PatchView(name=name, origin=origin, distance=d, lobes=lobes,
                     phase=phase, lr=lr, tb='Z' if lr == 'X' else 'X',
                     orientation=orientation)


def test_classifier_rows():
    d = 3
    # stacked cells (0,0)/(0,1): seam line horizontal; the measured logical
    # must run horizontally on both patches
    a_tb = _view("A", (0, 0), d, TEXTBOOK, 'X_horizontal')
    b_tb = _view("B", (0, 1), d, TEXTBOOK, 'X_horizontal')
    b_cj = _view("B", (0, 1), d, CONJUGATE, 'X_horizontal')
    r1 = classify_seam(a_tb, 'X', b_tb, 'X')
    assert (r1.row, r1.construction) == (1, 'merge')
    r2 = classify_seam(a_tb, 'X', b_cj, 'X')
    assert (r2.row, r2.construction) == (2, 'wall')
    # mixed: A measures X (X̄ horizontal), B measures Z (Z̄ horizontal = X̄ vertical)
    b_cj_v = _view("B", (0, 1), d, CONJUGATE, 'X_vertical')
    b_tb_v = _view("B", (0, 1), d, TEXTBOOK, 'X_vertical')
    r3 = classify_seam(a_tb, 'X', b_cj_v, 'Z')
    assert (r3.row, r3.construction) == (3, 'wall')
    r4 = classify_seam(a_tb, 'X', b_tb_v, 'Z')
    assert (r4.row, r4.construction) == (4, 'merge')


def test_classifier_rejects_perpendicular_logical():
    d = 3
    a = _view("A", (0, 0), d, TEXTBOOK, 'X_vertical')   # X̄ vertical
    b = _view("B", (0, 1), d, TEXTBOOK, 'X_horizontal')
    with pytest.raises(SeamRuleError, match="PARALLEL"):
        classify_seam(a, 'X', b, 'X')                   # horizontal seam


def test_classifier_rejects_non_adjacent():
    d = 3
    a = _view("A", (0, 0), d, TEXTBOOK, 'X_horizontal')
    b = _view("B", (0, 2), d, TEXTBOOK, 'X_horizontal')
    with pytest.raises(SeamRuleError, match="cell-adjacent"):
        classify_seam(a, 'X', b, 'X')


def test_assign_walls_minority_class():
    d = 3
    views = {
        "T1": _view("T1", (0, 0), d, TEXTBOOK, 'X_horizontal'),
        "C1": _view("C1", (0, 1), d, CONJUGATE, 'X_horizontal'),
        "C2": _view("C2", (1, 0), d, CONJUGATE, 'X_horizontal'),
        "C3": _view("C3", (1, 1), d, CONJUGATE, 'X_horizontal'),
    }
    target = [("T1", "X"), ("C1", "X"), ("C2", "X"), ("C3", "X")]
    seams = [("T1", "C1"), ("C1", "C2"), ("C2", "C3"), ("T1", "C2")]
    walls = assign_walls(views, target, seams)
    # 1 textbook + 3 conjugate -> the textbook patch's seams carry the walls
    assert walls == {("T1", "C1"): True, ("C1", "C2"): False,
                     ("C2", "C3"): False, ("T1", "C2"): True}
    # all same type -> no walls anywhere
    views2 = dict(views, T1=_view("T1", (0, 0), d, CONJUGATE, 'X_horizontal'))
    assert set(assign_walls(views2, target, seams).values()) == {False}
