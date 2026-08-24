"""Bus-basis (majority) rule for subset joint measurements.

The corridor's native basis is the MAJORITY target basis; only the minority-basis
patches attach through mixed (XZ) domain walls.  In particular a pure-X or pure-Z
joint uses NO mixed stabilizers at all (the pure-Z corridor is the exact CSS dual
of the pure-X one).  Ties keep the historical X-bus behaviour.
"""

import pytest

from circls.core.routed_multi_patch_ls import PatchSpec
from circls.core.multi_patch_coupler import (
    origin_of, route_and_build)

pytestmark = pytest.mark.smoke

D = 3


def _joint(paulis, spacing=2):
    # coarse-grid corridor placements (the architecture's geometry): target
    # patch interiors are inviolable, walls live on access seams - the old
    # fine-grid abutting form repainted patch interiors and is retired
    patches = [PatchSpec(f"P{i}", origin_of(spacing * i, 0, D, seam=True), D,
                         "X_horizontal")
               for i, P in enumerate(paulis)]
    target = [(f"P{i}", P) for i, P in enumerate(paulis)]
    r = route_and_build(patches, target, seam=True)
    assert r.status == "ok", (paulis, r.status, getattr(r, "message", ""))
    return r


def _n_mixed(lay):
    return sum(1 for c in lay.checks if c["type"] == "M")


def _wall_patches(lay, target_names, placed_cells):
    """Names of target patches with at least one mixed check on their boundary."""
    out = set()
    for ch in lay.checks:
        if ch["type"] != "M":
            continue
        for nm, cells in placed_cells.items():
            if any(q in cells for q in ch["pauli"]):
                out.add(nm)
    return out & set(target_names)


def test_pure_z_uses_no_mixed_wall():
    r = _joint("ZZ")
    assert _n_mixed(r.layout) == 0, "pure-Z joint must use a native Z bus (no walls)"
    assert all(r.layout.verify().values())


def test_pure_z_circuit_deterministic_and_not_collapsed():
    r = _joint("ZZ")
    c = r.layout.build_circuit(rounds=D, p=0.0)
    det, obs = c.compile_detector_sampler(seed=1).sample(200, separate_observables=True)
    assert not det.any() and not obs.any()
    cn = r.layout.build_circuit(rounds=D, p=1e-3)
    d_eff = len(cn.shortest_graphlike_error(canonicalize_circuit_errors=True))
    # dual of the pure-X joint: the tracked logical spans the bus; with 1 tile the
    # X-bus measures d_eff = d*t + (d-1) = 5 at d=3 -- the Z dual must match.
    assert d_eff >= D, f"pure-Z joint collapsed: d_eff={d_eff}"


def test_pure_x_regression_unchanged():
    r = _joint("XX")
    assert _n_mixed(r.layout) == 0
    cn = r.layout.build_circuit(rounds=D, p=1e-3)
    # standalone-harness distance only (real joint distance is covered end-to-end
    # by the routed-pipeline tests); the split schedule keeps XX from collapsing.
    assert len(cn.shortest_graphlike_error(canonicalize_circuit_errors=True)) >= D


def test_majority_z_walls_only_on_x_patch():
    """XZZ: nZ=2 > nX=1 -> Z bus; the single X patch carries the only wall."""
    r = _joint("XZZ")
    from circls.core.multi_patch_coupler import place_patch
    placed = {f"P{i}": set(place_patch(
        PatchSpec(f"P{i}", origin_of(2 * i, 0, D), D, "X_horizontal"))["data"])
        for i, P in enumerate("XZZ")}
    walls = _wall_patches(r.layout, [f"P{i}" for i in range(3)], placed)
    assert _n_mixed(r.layout) > 0
    assert walls == {"P0"}, f"walls must sit only on the minority (X) patch, got {walls}"
    assert all(r.layout.verify().values())


def test_tie_keeps_x_bus():
    """XZ tie: historical behaviour (X bus, wall on the Z patch)."""
    r = _joint("XZ")
    from circls.core.multi_patch_coupler import place_patch
    placed = {f"P{i}": set(place_patch(
        PatchSpec(f"P{i}", origin_of(2 * i, 0, D), D, "X_horizontal"))["data"])
        for i, P in enumerate("XZ")}
    walls = _wall_patches(r.layout, ["P0", "P1"], placed)
    assert walls == {"P1"}, f"tie must keep the X bus (wall on Z patch), got {walls}"


def test_majority_x_walls_only_on_z_patch():
    """ZXX: nX=2 > nZ=1 -> X bus; wall only on the Z patch (position-independent)."""
    r = _joint("ZXX")
    from circls.core.multi_patch_coupler import place_patch
    placed = {f"P{i}": set(place_patch(
        PatchSpec(f"P{i}", origin_of(2 * i, 0, D), D, "X_horizontal"))["data"])
        for i, P in enumerate("ZXX")}
    walls = _wall_patches(r.layout, [f"P{i}" for i in range(3)], placed)
    assert walls == {"P0"}, f"walls must sit only on the minority (Z) patch, got {walls}"


def test_majority_z_memory_basis_follows_bus():
    """XZZ (Z bus): the memory experiment must run in the bus basis and track a
    bus-basis logical — the X-memory form is ill-posed on a Z bus (a single
    measurement flip becomes an undetected observable flip)."""
    for paulis in ("XZZ", "ZXZ"):
        r = _joint(paulis)
        c0 = r.layout.build_circuit(rounds=D, p=0.0)
        det, obs = c0.compile_detector_sampler(seed=1).sample(200, separate_observables=True)
        assert not det.any() and not obs.any(), paulis
        cn = r.layout.build_circuit(rounds=D, p=1e-3)
        d_eff = len(cn.shortest_graphlike_error(canonicalize_circuit_errors=True))
        assert d_eff >= 3, f"{paulis}: memory basis must follow the bus, d_eff={d_eff}"


def test_intbasis_reduce_is_linear_and_no_subjoint_is_sound():
    """verify()'s no_subjoint gate XORs the residues ``B.reduce(single_k)``
    and reads a subset product as lying in span(B) iff those residues XOR to
    zero -- valid ONLY when reduce is LINEAR.  The old reduce returned at the
    first leading bit without a pivot, leaving lower pivot bits set: a
    non-canonical residue that broke linearity, so a proper subset product
    already in span(B) (e.g. the {0,1} pair product of a 4-body joint) escaped
    the gate and would ship as a silent sub-product measurement
    (bug 2026-08-24).  Pin the property the gate depends on."""
    from circls.core.multi_patch_coupler import _IntBasis

    def lcg(seed):
        x = seed
        while True:
            x = (1103515245 * x + 12345) & 0x7fffffff
            yield x

    g = lcg(2026)
    for _ in range(3000):
        B, vecs = _IntBasis(), []
        for _ in range(next(g) % 6):
            v = next(g) % 512
            B.add(v)
            vecs.append(v)
        a, b = next(g) % 512, next(g) % 512
        assert (B.reduce(a) ^ B.reduce(b)) == B.reduce(a ^ b), \
            "reduce must be linear (no_subjoint XORs residues)"
        span = {0}
        for v in vecs:
            span |= {s ^ v for s in span}
        w = next(g) % 512
        assert (B.reduce(w) == 0) == (w in span), \
            "reduce(v)==0 must hold iff v is in span(B)"

    # the finding's exact replica: {0,1} pair product (0b0110 ^ 0b0100 =
    # 0b0010) IS in span(B={0b0010}); the gate must now flag it.
    B = _IntBasis()
    B.add(0b0010)
    singles = [0b0110, 0b0100, 0x40, 0x80]
    full, piv2, ok = (1 << len(singles)) - 1, {}, True
    for k, s in enumerate(singles):
        r, m = B.reduce(s), 1 << k
        while r:
            bb = r.bit_length() - 1
            if bb not in piv2:
                piv2[bb] = (r, m)
                break
            pr, pm = piv2[bb]
            r ^= pr
            m ^= pm
        if not r and m != full:
            ok = False
    assert ok is False, \
        "no_subjoint must catch the {0,1} subset product lying in span(B)"
