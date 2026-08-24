"""Seam-column routing mode (``route_and_build(seam=True)``, progress §11).

The seam-column design is the standard-lattice-surgery corridor generalised:
coarse pitch ``2d+2``, one freshly-initialized data column/row between every
pair of edge-adjacent occupied cells, patches keep their TEXTBOOK standalone
construction verbatim (rule 0 / native lock, automatic in seam mode).

Circuits are built by the IR atomic-operation engine
(:class:`RoutedMultiPatchLSExperiment`: QECSystem + RotatedRoutedMultiPatchCoupler
+ CircuitBuilder + SyndromeTracker; noise via ``lightstim.noise``).

The headline physics: the state channel reaches the FULL code distance ``d``
with the plain textbook conjugate corridor init — no basis seal needed; the
m-consuming (teleportation) channel's distance is ``min(rounds, d)``.
"""

import contextlib
import io

import pytest

from lightstim.noise.config import NoiseConfig
from circls.core.routed_multi_patch_ls import (
    RoutedMultiPatchLSExperiment, PatchSpec, origin_of, route_and_build)
from circls.core.multi_patch_coupler import \
    BentLayoutError

pytestmark = pytest.mark.smoke

D = 3
NP = NoiseConfig(p_1q=1e-3, p_2q=1e-3, p_meas=1e-3, p_reset=1e-3, p_idle=1e-3)


def _min_weight(patches, target, init, meas, **ov):
    kw = dict(initial_states=init, measure_states=meas,
              rounds_init=3, rounds=D, noise_params=NP)
    kw.update(ov)
    with contextlib.redirect_stdout(io.StringIO()):
        exp = RoutedMultiPatchLSExperiment(patches, target, **kw)
        c = exp.build()
    errs = c.search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=6,
        dont_explore_edges_with_degree_above=6,
        dont_explore_edges_increasing_symptom_degree=True)
    return len(errs), c


def _spec(nm, a, b, o):
    return PatchSpec(nm, origin_of(a, b, D, seam=True), D, o)


def test_seam_straight_full_distance_textbook_init():
    patches = [_spec("Z1", 0, 0, "X_horizontal"),
               _spec("Z2", 2, 0, "X_horizontal")]
    w, c = _min_weight(patches, [("Z1", "Z"), ("Z2", "Z")],
                       {"Z1": "X", "Z2": "Z"}, {"Z1": "X", "Z2": "Z"})
    assert w == D and c.num_observables == 1


def test_seam_L_bend_full_distance():
    patches = [_spec("Z1", 0, 0, "X_horizontal"),
               _spec("Z2", 1, 1, "X_vertical")]
    w, _ = _min_weight(patches, [("Z1", "Z"), ("Z2", "Z")],
                       {"Z1": "X", "Z2": "Z"}, {"Z1": "X", "Z2": "Z"})
    assert w == D


def test_seam_T_junction_three_patches():
    patches = [_spec("Z1", 0, 0, "X_horizontal"),
               _spec("Z2", 1, 1, "X_vertical"),
               _spec("Z3", 2, 0, "X_horizontal")]
    w, c = _min_weight(patches, [("Z1", "Z"), ("Z2", "Z"), ("Z3", "Z")],
                       {"Z1": "X", "Z2": "Z", "Z3": "Z"},
                       {"Z1": "X", "Z2": "Z", "Z3": "Z"})
    assert w == D and c.num_observables == 2


def test_seam_with_obstacles_routes_clear():
    patches = [_spec("Z1", 0, 0, "X_horizontal"),
               _spec("Z2", 2, 0, "X_horizontal"),
               _spec("B1", 0, 2, "X_horizontal"),
               _spec("B2", 2, 2, "X_horizontal")]
    w, _ = _min_weight(patches, [("Z1", "Z"), ("Z2", "Z")],
                       {"Z1": "X", "Z2": "Z"}, {"Z1": "X", "Z2": "Z"})
    assert w == D


def test_seam_grid_rejects_abutting_origin():
    with pytest.raises(ValueError, match="not on the coarse routing grid"):
        route_and_build([PatchSpec("Z1", origin_of(1, 0, D), D, "X_horizontal"),
                         _spec("Z2", 2, 0, "X_horizontal")],
                        [("Z1", "Z"), ("Z2", "Z")], seam=True)


def test_seam_orientation_rule_rejects_perpendicular_logical():
    # measuring Z through an E/W seam with X_vertical: that adjacency runs
    # PERPENDICULAR to the measured logical — the explicit route is an
    # honest loud reject (restored original semantics; the old 'repair'
    # rebuilt both patches mid-life, which the architecture forbids)
    patches = [_spec("Q1", 0, 0, "X_vertical"),
               _spec("Q2", 2, 0, "X_horizontal")]
    with pytest.raises(BentLayoutError, match="PARALLEL|parallel-law"):
        route_and_build(patches, [("Q1", "Z"), ("Q2", "Z")], seam=True,
                        route=[(1, 0)])
    # auto-routing solves the same measurement through a legal face
    r = route_and_build(patches, [("Q1", "Z"), ("Q2", "Z")], seam=True)
    assert r.status == 'ok'
    assert all(r.layout.verify().values())


def test_seam_teleport_observable_min_rounds_d():
    # m-consuming teleportation channel (XX joint, source init X, destination
    # read out in X): the tracker folds the banked joint outcome m into the
    # observable automatically.  Its distance is min(rounds, d) — a
    # length-``rounds`` timelike chain of measurement errors on one chain
    # check flips the banked m with no closing detector (conjugate corridor
    # readout), so the merge must run d rounds to reach full distance.
    patches = [_spec("X1", 0, 0, "X_vertical"),
               _spec("X2", 2, 0, "X_vertical")]
    target = [("X1", "X"), ("X2", "X")]
    init, meas = {"X1": "X", "X2": "Z"}, {"X1": "Z", "X2": "X"}
    w, c = _min_weight(patches, target, init, meas)
    assert c.num_observables == 1
    assert w == D
    w1, _ = _min_weight(patches, target, init, meas, rounds=1)
    assert w1 == 1


def test_emv_knapsack_dijkstra_matches_dumb_oracle():
    # Cross-check from corridor-routing-model.md §6: the group-native
    # "Dijkstra with a knapsack" (EMV send-and-split, double node-weight
    # correction, multi-source seeding, no virtual-node reduction) must agree
    # case by case, on random instances, with the dumb oracle (<=2^k
    # face-by-face combinations x the same DP over singleton groups).
    import itertools
    import random
    import networkx as nx
    from circls.compiler.routing import (
        emv_group_steiner)

    def dumb(G, groups):
        best = None
        for combo in itertools.product(*groups):
            sol = emv_group_steiner(G, [[c] for c in combo])
            if sol and (best is None or sol[0] < best):
                best = sol[0]
        return best

    rng = random.Random(11)
    for _trial in range(400):
        free = [(x, y) for x in range(4) for y in range(3)
                if rng.random() < 0.75]
        G = nx.Graph()
        G.add_nodes_from(free)
        for (x, y) in free:
            for nb in ((x + 1, y), (x, y + 1)):
                if nb in G:
                    G.add_edge((x, y), nb)
        if not free:
            continue
        groups = [rng.sample(free, min(len(free), rng.choice((1, 2))))
                  for _ in range(rng.choice((2, 3)))]
        a = emv_group_steiner(G, groups)
        assert (a[0] if a else None) == dumb(G, groups)
