"""Hook-benign SE scheduling for the rotated bent-joint builder.

The syndrome-extraction schedule:

* per orientation-domain benign direction tables (fixes the transpose mismatch
  that halved the circuit distance of X_horizontal layouts);
* T=5 floating idle slot when two orientation domains meet (seam band);
* mixed (domain-wall) checks interleaved into the same coupling window with
  per-corner slot alignment (hook order inherited from the benign tables).
"""

import pytest

from circls.core.routed_multi_patch_ls import PatchSpec
from circls.core.multi_patch_coupler import place_patch
from lightstim.qec_code.surface_code.rotated.bent_joint_se import (
    RotatedBentJointMeasurement, assign_hook_benign_schedule, BENIGN_TABLES)
from circls.core.multi_patch_coupler import (
    origin_of, route_and_build)

pytestmark = pytest.mark.smoke

D = 3


# ── helpers ──────────────────────────────────────────────────────────────────

def _joint(paulis, d=D, spacing=2):
    patches = [PatchSpec(f"P{i}", origin_of(spacing * i, 0, d, seam=True), d,
                         "X_horizontal")
               for i, P in enumerate(paulis)]
    r = route_and_build(patches, [(f"P{i}", P) for i, P in enumerate(paulis)],
                        seam=True)
    assert r.status == "ok", (paulis, r.status)
    return r.layout


def _d_eff(circuit):
    return len(circuit.shortest_graphlike_error(canonicalize_circuit_errors=True))


def _assert_deterministic(circuit):
    det, obs = circuit.compile_detector_sampler(seed=1).sample(200, separate_observables=True)
    assert not det.any() and not obs.any()


def _assert_no_tick_collision(circuit):
    """No qubit may be touched by two instructions inside one tick slice."""
    used = set()
    for inst in circuit.flattened():
        if inst.name == "TICK":
            used = set()
            continue
        if inst.name in ("QUBIT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS"):
            continue
        if inst.name in ("DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Z_ERROR"):
            continue
        for t in inst.targets_copy():
            if t.is_qubit_target:
                assert t.value not in used, (inst.name, t.value)
                used.add(t.value)


def _round_ticks(circuit):
    """TICKs of the full circuit (proxy for round depth at rounds=1)."""
    return sum(1 for i in circuit.flattened() if i.name == "TICK")


# ── the transpose fix: single patch reaches full distance ────────────────────

@pytest.mark.parametrize("d", [3, 5])
def test_hook_benign_single_patch_full_distance(d):
    res = place_patch(PatchSpec("A", origin_of(0, 0, d), d, "X_horizontal"))
    for basis, sup in [("X", res['x_support']), ("Z", res['z_support'])]:
        b = RotatedBentJointMeasurement(list(res['data']), res['checks'], list(sup))
        c0 = b.circuit(rounds=d, p=0.0, basis=basis)
        _assert_deterministic(c0)
        c = b.circuit(rounds=d, p=1e-3, basis=basis)
        assert _d_eff(c) == d, (basis, d)


def test_hook_benign_x_vertical_full_distance():
    """X_vertical domains reach full distance too (the untransposed frame)."""
    d = 3
    res = place_patch(PatchSpec("A", origin_of(0, 0, d), d, "X_vertical"))
    domains = {q: "X_vertical" for q in res['data']}
    b = RotatedBentJointMeasurement(list(res['data']), res['checks'],
                                    list(res['x_support']), domains=domains)
    c = b.circuit(rounds=d, p=1e-3, basis="X")
    assert _d_eff(c) == d


# ── joint layouts: determinism, collision-freedom, acceptance ────────────────

@pytest.mark.parametrize("paulis", ["XX", "ZZ", "XZ"])
def test_hook_benign_joint_clean(paulis):
    lay = _joint(paulis)
    c0 = lay.build_circuit(rounds=D, p=0.0)
    _assert_deterministic(c0)
    _assert_no_tick_collision(c0)


# ── mixed checks: interleaved, bounded depth ─────────────────────────────────

def test_mixed_checks_bounded_depth():
    lay = _joint("XZX")   # minority-Z middle patch attaches via a recolour
    n_mixed = sum(1 for ch in lay.checks if ch['type'] == 'M')
    # corridor geometry: one d-cell mixed column on the minority access seam
    # (the abutting form's historical count was 5)
    assert n_mixed >= D
    t_new = _round_ticks(lay.build_circuit(rounds=1, p=0.0))
    # depth is bounded by the interleaved coupling window (R+H+T+H+M), not by
    # the number of mixed checks.
    assert t_new <= 26


# ── two orientation domains: T=5 floating idle slot (scheduler unit test) ────

def test_two_domain_schedule_is_t5_and_order_preserving():
    DIRS = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
    W, H, SEAM = 21, 13, 10
    data = {(x, y) for x in range(1, W + 1, 2) for y in range(1, H + 1, 2)}
    checks, domains = [], {}
    for q in data:
        domains[q] = "X_vertical" if q[0] < SEAM else "X_horizontal"
    for cx in range(0, W + 2, 2):
        for cy in range(0, H + 2, 2):
            pauli = {(cx + a, cy + b): ('X' if ((cx + cy) // 2) % 2 == 1 else 'Z')
                     for a, b in DIRS if (cx + a, cy + b) in data}
            if len(pauli) == 4:
                t = next(iter(pauli.values()))
                checks.append({'syn': (cx, cy), 'type': t, 'pauli': pauli,
                               'corners': sorted(pauli)})
    T, tickmaps = assign_hook_benign_schedule(checks, domains)
    assert T == 5
    # no data collision at any tick
    seen = set()
    for k, tm in tickmaps.items():
        for q, t in tm.items():
            assert (t, q) not in seen
            seen.add((t, q))
    # gate ORDER per check equals its domain benign order (idle only shifts positions)
    for k, tm in tickmaps.items():
        ch = checks[k]
        orient = domains[next(iter(ch['pauli']))]
        table = BENIGN_TABLES[orient][ch['type']]
        want = [ (ch['syn'][0] + d[0], ch['syn'][1] + d[1]) for d in table ]
        got = [q for q, _ in sorted(tm.items(), key=lambda kv: kv[1])]
        assert got == want, (ch['syn'],)


def test_single_domain_schedule_is_t4():
    lay = _joint("XX")
    pure = [ch for ch in lay.checks if ch['type'] in ('X', 'Z')]
    T, _ = assign_hook_benign_schedule(pure, dict(lay.domains))
    assert T == 4


def test_multi_patch_generator_populates_domains():
    """Routed layouts must carry a domains map for the hook-benign scheduler."""
    from circls.core.multi_patch_coupler import route_and_build
    d = 3
    patches = [PatchSpec("A", origin_of(0, 0, d), d, "X_vertical"),
               PatchSpec("B", origin_of(0, 2, d), d, "X_vertical")]
    r = route_and_build(patches, [("A", "X"), ("B", "X")])
    assert r.status == "ok"
    lay = r.layout
    assert lay.domains, "routed layout must carry a domains map"
    a_cells = set(place_patch(patches[0])["data"])
    assert all(lay.domains.get(q) == "X_vertical" for q in a_cells if q in lay.domains)
    assert all(q in lay.domains for q in lay.data)
