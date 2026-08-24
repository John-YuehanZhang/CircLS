"""Routed multi-patch lattice-surgery experiment — rotated seam-column design.

THE experiment-definition home for the routed subset joint measurement
``M(∏ᵢ P̄ᵢ)`` on rotated surface-code patches (seam-column corridor design,
rule-0 native lock, phased protocol A0 → merge → destructive readout;
minority patches register in the conjugate convention — no morph).

Layering (mirrors ``two_patch_ls.TwoPatchLSExperiment`` over
``unrotated/two_patch_coupler``):

* geometry / routing / stabilizer rules live in
  ``circls.core.multi_patch_coupler``;
* the SE engine lives in ``...rotated.bent_joint_se``;
* THIS module owns the experiment layer: the user-facing ``PatchSpec``,
  the phased circuit assembly (``build_phased_subset_circuit``), the
  report front-end (``report_subset_joint``) and (Phase 2) the atomic-op
  driven ``RoutedMultiPatchLSExperiment``.
"""

from .multi_patch_coupler import (   # noqa: F401
    PatchSpec, origin_of, route_and_build)


import numpy as np

from .multi_patch_coupler import (
    cell, _specs_to_cells, acceptance_of_layout, ACCEPTANCE_ITEMS,
    conjugate_patch_records)

__all__ = ["PatchSpec", "origin_of", "route_and_build",
           "report_subset_joint", "RoutedMultiPatchLSExperiment"]


def _print_acceptance(a, label):
    """(2) The :data:`ACCEPTANCE_ITEMS` PASS checklist."""
    n_ok = sum(1 for x in a['items'].values() if x is True)
    print(f"{label}   ->  {'PASS' if a['accept'] else 'FAIL'}   "
          f"({n_ok}/{len(ACCEPTANCE_ITEMS)} conditions)")
    print(f"    data qubits = {a['data']}   independent stabilizers (rank) = {a['n_stab']}   "
          f"readout chain = {a['readout_chain_len']} stabs")
    print(f"    remaining logical dof  =  data - stabilizers  =  {a['data']} - {a['n_stab']}  =  "
          f"{a['k']}   (want N-1 = {a['N']-1})")
    for k, x in a['items'].items():
        print(f"    [{'True ' if x is True else str(x)}] {k}")


def _draw_path(patches, target, route, title, seam=False):
    """(1) The routed path on the coarse-cell grid: target patches coloured (X̄ blue / Z̄ orange),
    obstacles grey-hatched, the ancilla corridor in green."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Rectangle
    patch_at, _o, d = _specs_to_cells(patches, target, seam=seam)
    tset = dict(target)
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    def box(a, b, **kw):
        qs = cell(a, b, d, seam); x0, x1 = min(q[0] for q in qs), max(q[0] for q in qs)
        y0, y1 = min(q[1] for q in qs), max(q[1] for q in qs)
        ax.add_patch(Rectangle((x0 - 1, y0 - 1), x1 - x0 + 2, y1 - y0 + 2, **kw))
        return (x0 + x1) / 2, (y0 + y1) / 2
    for (a, b) in route:                                        # ancilla corridor
        box(a, b, facecolor='#bfe6bf', edgecolor='#2ca02c', lw=1.4, zorder=1)
    for p in patches:                                           # patches
        a, b = patch_at[p.name]
        if p.name in tset:
            cx, cy = box(a, b, facecolor=('#6f9cf0' if tset[p.name] == 'X' else '#ef7a45'),
                         edgecolor='white', lw=1, zorder=2)
            ax.text(cx, cy, p.name, ha='center', va='center', color='white', fontsize=9, fontweight='bold', zorder=3)
        else:
            cx, cy = box(a, b, facecolor='0.86', edgecolor='0.6', hatch='xxx', lw=0.5, zorder=1)
            ax.text(cx, cy, p.name, ha='center', va='center', color='0.4', fontsize=8, zorder=3)
    ax.set_aspect('equal'); ax.invert_yaxis(); ax.autoscale_view(); ax.margins(0.08); ax.axis('off')
    ax.set_title(f"{title}  —  routed path (coarse cells)", fontsize=11)
    ax.legend(handles=[mpatches.Patch(color='#6f9cf0', label='target X̄'),
                       mpatches.Patch(color='#ef7a45', label='target Z̄'),
                       mpatches.Patch(facecolor='#bfe6bf', edgecolor='#2ca02c', label='ancilla corridor'),
                       mpatches.Patch(facecolor='0.86', edgecolor='0.6', hatch='xxx', label='obstacle')],
              loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=8.5, frameon=False)
    plt.tight_layout(); plt.show()


def _show_layout(lay, title):
    """(3) Data-qubit layout: X/Z/mixed stabilizers, data qubits, logicals, gold readout chain."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Polygon, Circle, Rectangle
    COL = {'X': '#e23b3b', 'Z': '#2f6fd0', 'M': '#8b3fd0'}
    FILL = {'X': '#f6b8b8', 'Z': '#b8cdf0', 'M': '#d9c2f2'}; GOLD = '#f0a000'
    CHAIN = lay.readout_chain
    allc = [c for ch in lay.checks for c in ch['corners']] + list(lay.data)
    x0, x1 = min(p[0] for p in allc) - 1.6, max(p[0] for p in allc) + 1.6
    y0, y1 = min(p[1] for p in allc) - 1.6, max(p[1] for p in allc) + 1.6
    fig, ax = plt.subplots(figsize=(min(13, 0.85 + (x1 - x0) * 0.42), min(13, 0.85 + (y1 - y0) * 0.42)))
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor='#f6f6f9', ec='none', zorder=0))
    for ch in lay.checks:
        t, pts, syn = ch['type'], ch['corners'], ch['syn']; hl = syn in CHAIN; ec = GOLD if hl else COL[t]
        if len(pts) >= 3:
            cx, cy = np.mean([p[0] for p in pts]), np.mean([p[1] for p in pts])
            order = sorted(pts, key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))
            ax.add_patch(Polygon(order, closed=True, facecolor=FILL[t], edgecolor=ec,
                                 lw=2.4 if hl else 0.6, alpha=0.9 if hl else 0.30, zorder=3 if hl else 2))
        elif len(pts) == 2:
            (a, b), (c, dd) = pts
            ax.plot([a, syn[0], c], [b, syn[1], dd], color=ec, lw=7 if hl else 5,
                    alpha=0.7 if hl else 0.28, solid_capstyle='round', zorder=3 if hl else 2)
        if t == 'M':
            for c, P in ch['pauli'].items():
                ax.plot([syn[0], c[0]], [syn[1], c[1]], color=COL[P], lw=2.4, zorder=5, alpha=0.9, solid_capstyle='round')
        ax.add_patch(Rectangle((syn[0] - 0.18, syn[1] - 0.18), 0.36, 0.36, facecolor=COL[t],
                     edgecolor=GOLD if hl else 'white', lw=2 if hl else 0.8, zorder=6, alpha=1 if hl else 0.55))
    for q in lay.data:
        ax.add_patch(Circle(q, 0.14, facecolor='#1a1a1a', edgecolor='white', lw=0.6, zorder=8))
    xshades = ['#c01616', '#7a0d0d', '#e0552a', '#9c1b5a', '#5a1b9c']; xi = 0
    for nm, P, sup in lay.logicals:
        col = '#13346e' if P == 'Z' else xshades[xi % len(xshades)]; xi += (P == 'X')
        srt = sorted(sup)
        ax.plot([q[0] for q in srt], [q[1] for q in srt], color=col, lw=5, zorder=10, solid_capstyle='round')
        mx, my = srt[len(srt) // 2]
        ax.text(mx, my - 0.8, fr'$\bar {P}_{{{nm[1:]}}}$', color=col, fontsize=13, fontweight='bold', ha='center', zorder=11,
                bbox=dict(boxstyle='round,pad=0.12', fc='white', ec=col, lw=1))
    handles = [mpatches.Patch(color=COL['X'], label='X stabilizer'), mpatches.Patch(color=COL['Z'], label='Z stabilizer'),
               mpatches.Patch(color=COL['M'], label='MIXED (XZ) wall'),
               mpatches.Patch(facecolor='#fff3d6', edgecolor=GOLD, lw=2, label='readout chain (product = joint)')]
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.005), ncol=2, fontsize=9)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect('equal'); ax.invert_yaxis(); ax.axis('off')
    ax.set_title(title, fontsize=12, pad=46); plt.tight_layout(); plt.show()


def _default_title(target):
    """``[("X1","X"), ("B2","X")]`` -> ``"Subset joint  M(X̄1 X̄_B2)"`` (notebook naming style)."""
    def term(nm, P):
        return f"{P}̄{nm[1:]}" if nm[:1] == P and nm[1:].isdigit() else f"{P}̄_{nm}"
    return "Subset joint  M(" + " ".join(term(nm, P) for nm, P in target) + ")"








# =============================================================================
# RoutedMultiPatchLSExperiment — the atomic-operation experiment class
# =============================================================================

from typing import Dict, List, Optional, Tuple

import stim

from lightstim.ir.experiment import QECExperiment
from lightstim.ir.qec_system import QECSystem
from lightstim.qec_code.surface_code.rotated import RotatedSurfaceCode
from .multi_patch_coupler import (
    RotatedRoutedMultiPatchCoupler)
from lightstim.qec_code.surface_code.rotated.bent_joint_se import se_round_chunk

_CONJ = {'X': 'Z', 'Z': 'X'}
_FLIP_O = {'X_horizontal': 'X_vertical', 'X_vertical': 'X_horizontal'}


class RoutedMultiPatchLSExperiment(QECExperiment):
    """Routed multi-patch joint measurement ``M(∏ᵢ P̄ᵢ)`` on rotated patches.

    The seam-column phased protocol, driven END-TO-END by IR atomic
    operations (cf. ``TwoPatchLSExperiment``):

    1. every target patch is added to the :class:`QECSystem` — majority
       (bus-Pauli) patches as the TEXTBOOK :class:`RotatedSurfaceCode`,
       minority patches in the CONJUGATE CONVENTION
       (:func:`conjugate_patch_records`: transposed geometry with X/Z
       swapped — exactly the structure the routed merged check set assigns
       to the minority region) — and initialised per ``initial_states``
       (bases are literal; there is no hidden frame);
    2. ``rounds_init`` standalone SE rounds (``builder.apply_syndrome_extraction``);
    3. ``builder.activate_coupler`` pauses the superseded native checks and
       activates the routed merged check set
       (:class:`RotatedRoutedMultiPatchCoupler`); patch interiors — minority
       included — match the merged set natively (rule-0 lock);
    4. corridor + seam data initialised in the conjugate of the bus basis
       (``builder.initialize``), then ``rounds`` merged SE rounds;
    5. destructive readout per ``measure_states`` (corridor in its init
       basis — the split-family textbook corridor)
       (``builder.apply_data_readout``);
    6. optional noise via ``builder.build_noisy_circuit``.

    The protocol contains NO transversal-H morph: the conjugate-convention
    registration replaces it (verified equal in observables and graphlike
    distance on straight / comb / L / T geometries at d = 3 and 5, minus the
    morph's noisy 1q layer).

    Detectors and OBSERVABLE_INCLUDEs come from the ``SyndromeTracker``:
    bare logicals of (init==P, measure==P) patches appear automatically, and
    teleportation-style record folding (the banked joint outcome m entering
    an observable) is handled by the tracker's decomposition — no hand-written
    detector rules.
    """

    def __init__(self,
                 patches: List[PatchSpec],
                 interaction_type: List[Tuple[str, str]],
                 *,
                 initial_states: Optional[Dict[str, str]] = None,
                 measure_states: Optional[Dict[str, str]] = None,
                 rounds_init: int = 1,
                 rounds: int = 3,
                 route: Optional[List[Tuple[int, int]]] = None,
                 seam: bool = True,
                 bus: Optional[str] = None,
                 noise_params=None,
                 noise_model: Optional[str] = 'circuit_level'):
        super().__init__(extraction_block_class=None, rounds=rounds,
                         noise_params=noise_params, noise_model=noise_model)
        assert rounds_init >= 1, (
            f"rounds_init must be >= 1 (a freshly initialised patch needs at least one "
            f"standalone SE round to establish its stabilizers before the merge); "
            f"got {rounds_init}")
        self.patches = list(patches)
        self.interaction_type = list(interaction_type)
        self.rounds_init = rounds_init
        self.route = route
        self.seam = seam
        self.coupler_name = "routed_bus"

        P_of = dict(self.interaction_type)
        n_x = sum(1 for _, P in self.interaction_type if P == 'X')
        # ``bus`` (optional): a PLANNED native basis.  When None, the historical
        # majority target basis (tie -> X).  Forcing the non-majority basis conjugates
        # the OTHER patches (minority = those whose Pauli != bus) — valid physics (the
        # joint reduces to a pure-bus merge), just with more mixed walls.
        self._route_bus = bus  # None unless explicitly forced; passed to route_and_build
        self.bus = bus if bus is not None else (
            'X' if n_x >= len(self.interaction_type) - n_x else 'Z')
        self.minority_names = frozenset(nm for nm, P in self.interaction_type
                                        if P != self.bus)
        # default: every target prepared AND read out in its own joint Pauli
        self.initial_states = {nm: P_of.get(nm, 'Z') for nm in
                               (s.name for s in self.patches)}
        self.initial_states.update({k: v.upper() for k, v in
                                    (initial_states or {}).items()})
        self.measure_states = {nm: P_of.get(nm, 'Z') for nm in
                               (s.name for s in self.patches)}
        self.measure_states.update({k: v.upper() for k, v in
                                    (measure_states or {}).items()})
        self.subset_route = None                    # cache, filled by show()/build()

    def show(self, *, path=True, layout=True, acceptance=False, title=None):
        """Draw the routed-path figure and the stabilizer-layout figure for
        this experiment (the same two figures ``report_subset_joint`` shows);
        ``acceptance=True`` additionally prints the acceptance checklist.
        Routes on demand, so it works before or after :meth:`build`."""
        r = self.subset_route
        if r is None:
            r = route_and_build(self.patches, self.interaction_type,
                                seam=self.seam, route=self.route, bus=self._route_bus)
            if r.status != 'ok':
                raise ValueError(f'route_and_build failed: {r.status} — {r.message}')
            self.subset_route = r
        if title is None:
            title = _default_title(self.interaction_type)
        if path:
            _draw_path(self.patches, self.interaction_type, sorted(r.tree),
                       title, seam=self.seam)
        if acceptance:
            _print_acceptance(acceptance_of_layout(r.layout), title)
        if layout:
            _show_layout(r.layout, title)

    def build(self) -> stim.Circuit:
        specs, target = self.patches, self.interaction_type
        r = self.subset_route                       # reuse show()'s routing if any
        if r is None:
            r = route_and_build(specs, target, seam=self.seam, route=self.route,
                                bus=self._route_bus)
        if r.status != 'ok':
            raise ValueError(f'route_and_build failed: {r.status} — {r.message}')
        self.subset_route = r

        # -- 1. system assembly: majority patches textbook, minority patches
        #       in the conjugate convention (transposed geometry + typeswap) --
        self.system = QECSystem()
        reg_orients = {}
        for s in specs:
            geo = _FLIP_O[s.orientation] if s.name in self.minority_names \
                else s.orientation
            p = RotatedSurfaceCode(distance=s.distance)
            if geo == 'X_horizontal':
                p.transpose_coords()
            if s.name in self.minority_names:
                # the patch runs the merged (painted) structure from round 0;
                # its logical directions match the DECLARED orientation
                conjugate_patch_records(p)
            reg_orients[s.name] = s.orientation
            self.system.add_patch(
                p, name=s.name, offset=(s.origin[0] - 1, s.origin[1] - 1))
        self.system.register_coupler(
            RotatedRoutedMultiPatchCoupler(),
            patch_names=[nm for nm, _ in target], name=self.coupler_name,
            specs=specs, target=target, subset_route=r, seam=self.seam,
            minority_names=self.minority_names)
        self._setup_experiment()
        self.builder.write_coordinates()

        # -- 2. patch init (bases literal) + A0 rounds -----------------------
        owner = self.system.index_to_owner_map
        init_dict, domains_pre = {}, {}
        for q in self.system.data_indices:
            nm = owner.get(q)
            if nm not in self.initial_states:
                continue                        # coupler data: initialised at merge
            init_dict[q] = self.initial_states[nm]
            domains_pre[tuple(self.system.qubit_coords[q])] = reg_orients[nm]
        self.builder.initialize(init_dict=init_dict, n=self.system.num_qubits)
        chunk = se_round_chunk(self.system, domains=domains_pre)
        self.builder.apply_syndrome_extraction(circuit_chunk=chunk,
                                               rounds=self.rounds_init)

        # -- 3./4. merge: activate coupler, init corridor, merged rounds -----
        self.builder.activate_coupler(self.coupler_name)
        coupler_patch = self.system.coupler_patches[self.coupler_name]
        l2g = self.system.local_to_global_map[self.coupler_name]
        corridor_basis = _CONJ[self.bus]
        coupler_init = {l2g[q]: corridor_basis
                        for q in coupler_patch.data_indices}
        if coupler_init:
            self.builder.initialize(init_dict=coupler_init,
                                    n=self.system.num_qubits)
        lay = r.layout
        if any(c.get('kf') for c in lay.checks):
            # a seam wall's stretched (kf relay) checks cannot be expressed
            # by the bent hook-benign chunk — same rule as the sequential
            # pipeline: wall steps run the K&F diagonal schedule
            from lightstim.qec_code.surface_code.rotated.diagonal_se import (
                DiagonalSurfaceCodeExtractionBlock)
            chunk = DiagonalSurfaceCodeExtractionBlock(self.system).circuit
        else:
            domains_merged = dict(getattr(lay, 'domains', {}) or {})
            chunk = se_round_chunk(self.system, domains=domains_merged)
        self.builder.apply_syndrome_extraction(circuit_chunk=chunk,
                                               rounds=self.rounds)

        # -- 5. destructive readout -------------------------------------------
        meas = dict(coupler_init)                       # corridor in init basis
        for q in self.system.data_indices:
            nm = owner.get(q)
            if nm in self.measure_states:
                meas[q] = self.measure_states[nm]
        self.builder.apply_data_readout(final_measurements=meas)

        # -- 6. noise ----------------------------------------------------------
        if self.noise_params is not None:
            return self.builder.build_noisy_circuit(
                noise_params=self.noise_params, noise_model=self.noise_model)
        return self.builder.circuit








def report_subset_joint(patches, interaction_type, title=None, *, route=None,
                        show_path=True, show_acceptance=True, show_layout=True,
                        rounds=2, rounds_init=1, initial_states=None,
                        measure_states=None, seam=False):
    """Route, verify and display a subset joint ``M(∏ᵢ P̄ᵢ)``.

    Shows the routed-path coarse-cell figure, the 12-item acceptance
    checklist and the data-qubit layout, then returns the verified
    :class:`SubsetRoute` with ``.circuit`` attached — built by the IR
    atomic-operation engine (:class:`RoutedMultiPatchLSExperiment`;
    detectors/observables from the ``SyndromeTracker``)."""
    exp = RoutedMultiPatchLSExperiment(
        patches, interaction_type,
        initial_states=initial_states, measure_states=measure_states,
        rounds_init=rounds_init, rounds=rounds, route=route, seam=seam)
    circuit = exp.build()
    r = exp.subset_route
    if title is None:
        title = _default_title(interaction_type)
    how = ('standard construction' if r.how == 'standard'
           else f'convex-corner cut at {r.cut}')
    if show_path:
        _draw_path(patches, interaction_type, sorted(r.tree), title, seam=seam)
    if show_acceptance:
        _print_acceptance(acceptance_of_layout(r.layout), f"{title}   [{how}]")
    if show_layout:
        _show_layout(r.layout, title)
    r.circuit = circuit
    return r
