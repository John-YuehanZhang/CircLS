"""Layer 2 of the resource aggregator: tile-level statistics for a
``SequentialPPMExperiment``.

Zero-intrusion design: nothing in ``build()`` is instrumented.  Everything is
derived from the built circuit (per-qubit holding intervals, via
``circuit_stats``) plus the experiment's geometry — qubit coordinates, patch
specs, and the ownership map (which survives retirement by design).

Tile convention: the coarse seam-grid cell, pitch ``P = 2d + 2``; qubit
``(x, y)`` belongs to tile ``(x // P, y // P)``.  Seam-column qubits
(``x = P·k − 1``) attach to the cell on their left/bottom — whenever a seam
is active that neighbouring cell is occupied too, so no phantom tiles arise.

Metrics produced (S1/V3 reinstated
once mapping moved inside the pipeline):
  S1 ``tiles_bbox``          — bounding-box area of the touched tiles,
  S2 ``tiles_touched``       — tiles ever occupied,
  S3 ``tiles_peak``          — max tiles occupied in any one round,
  T2 ``logical_timesteps``   — measurement_layers / d,
  V1 ``volume_blocks``       — Σ_t occupied_tiles(t) / d,
  V3 ``bbox_volume_blocks``  — tiles_bbox × measurement_layers / d
                               (TopoLS/DASCOT-style accounting, dialogue only),
  L1 ``live_tile_rounds``    — Σ over data patches of (retire − birth + 1),
  L2 ``utilization``         — L1 / (S3 × T1),
  L4 ``reuse_rate``          — freed patch cells later re-occupied / cells freed,
  T4 ``compile_seconds``     — caller-supplied (see ``timed_build``).

Built-in self-check: our compiler must not declare qubits nobody uses —
``qubits_active == qubits_allocated`` is enforced (S4==S5 gate); the V1↔V2
cross-account density ``qubit_rounds / tile_rounds`` is exposed for the
report-level sanity band.
"""
from __future__ import annotations

import dataclasses
import time
from collections import Counter
from typing import Dict, Optional, Tuple

import stim

from circls.metrics.circuit_stats import CircuitStats, circuit_stats


@dataclasses.dataclass(frozen=True)
class ExperimentStats:
    circuit: CircuitStats
    distance: int                           # uniform code distance d
    tiles_bbox: int                         # S1: bounding-box area of touched tiles
    tiles_touched: int                      # S2
    tiles_peak: int                         # S3
    logical_timesteps: float                # T2 = measurement_layers / d
    tile_rounds: int                        # Σ_t occupied_tiles(t)  (V1 numerator)
    volume_blocks: float                    # V1 = tile_rounds / d
    bbox_volume_blocks: float               # V3 = tiles_bbox × measurement_layers / d
    live_tile_rounds: int                   # L1
    utilization: float                      # L2
    reuse_rate: Optional[float]             # L4; None when nothing was freed mid-run
    compile_seconds: Optional[float]        # T4
    qubits_per_tile_round: float            # V2 / tile_rounds — V1↔V2 density (sanity band)
    patch_live: Dict[str, Tuple[int, int]]  # patch -> (birth_round, retire_round)
    tile_occupancy: Dict[Tuple[int, int], Tuple[int, ...]]  # tile -> sorted occupied rounds


def timed_build(exp) -> Tuple[stim.Circuit, float]:
    """Build the experiment's circuit, returning ``(circuit, seconds)``."""
    t0 = time.perf_counter()
    circuit = exp.build()
    return circuit, time.perf_counter() - t0


def experiment_stats(exp, circuit: stim.Circuit, *,
                     compile_seconds: Optional[float] = None) -> ExperimentStats:
    cs = circuit_stats(circuit)
    if cs.qubits_active != cs.qubits_allocated:
        # S4==S5 gate: our compiler must not declare qubits nobody uses.
        idle = cs.qubits_allocated - cs.qubits_active
        raise ValueError(
            f"S4==S5 self-check failed: {idle} declared qubit(s) are never "
            f"touched by any instruction — wasted coordinates or a builder bug "
            f"(active={cs.qubits_active}, allocated={cs.qubits_allocated})")

    dists = {s.distance for s in exp.patches}
    if len(dists) != 1:
        raise ValueError(f"experiment_stats needs a uniform distance, got {sorted(dists)}")
    d = dists.pop()
    pitch = 2 * d + 2                     # seam-column grid (the sequential pipeline)
    coords = exp.system.qubit_coords

    # ── tile occupancy: union of the tile's qubits' holding intervals ────────
    tile_sets: Dict[Tuple[int, int], set] = {}
    for q, ivals in cs.occupancy.items():
        x, y = coords[q]
        tile = (int(x) // pitch, int(y) // pitch)
        rs = tile_sets.setdefault(tile, set())
        for lo, hi in ivals:
            rs.update(range(lo, hi + 1))

    n_rounds = cs.measurement_layers
    per_round: Counter = Counter()
    for rs in tile_sets.values():
        per_round.update(rs)
    tiles_peak = max(per_round.values(), default=0)
    tile_rounds = sum(len(rs) for rs in tile_sets.values())
    if tile_sets:
        aa = [a for a, _ in tile_sets]
        bb = [b for _, b in tile_sets]
        tiles_bbox = (max(aa) - min(aa) + 1) * (max(bb) - min(bb) + 1)
    else:
        tiles_bbox = 0

    # ── patch live ranges: each data qubit's FIRST holding interval is the
    #    patch's own (later intervals on the same qubit are corridor reuse) ──
    patch_names = {s.name for s in exp.patches}
    owner = exp.system.index_to_owner_map
    patch_rounds: Dict[str, set] = {}
    for q in exp.system.data_indices:
        nm = owner.get(q)
        if nm not in patch_names or q not in cs.occupancy:
            continue
        lo, hi = cs.occupancy[q][0]
        patch_rounds.setdefault(nm, set()).update(range(lo, hi + 1))
    patch_live = {nm: (min(rs), max(rs)) for nm, rs in patch_rounds.items()}
    live_tile_rounds = sum(hi - lo + 1 for lo, hi in patch_live.values())

    utilization = (live_tile_rounds / (tiles_peak * n_rounds)
                   if tiles_peak and n_rounds else 0.0)

    # ── L4: patch cells freed before the end — were they re-occupied? ───────
    from circls.core.multi_patch_coupler import cell_index
    freed = reused = 0
    for s in exp.patches:
        lohi = patch_live.get(s.name)
        if lohi is None or lohi[1] >= n_rounds:
            continue                       # lived to the final round — never freed
        freed += 1
        cell = cell_index(s.origin, d, seam=True)
        if any(r > lohi[1] for r in tile_sets.get(cell, ())):
            reused += 1
    reuse_rate = (reused / freed) if freed else None

    return ExperimentStats(
        circuit=cs,
        distance=d,
        tiles_bbox=tiles_bbox,
        tiles_touched=len(tile_sets),
        tiles_peak=tiles_peak,
        logical_timesteps=n_rounds / d,
        tile_rounds=tile_rounds,
        volume_blocks=tile_rounds / d,
        bbox_volume_blocks=tiles_bbox * n_rounds / d,
        live_tile_rounds=live_tile_rounds,
        utilization=utilization,
        reuse_rate=reuse_rate,
        compile_seconds=compile_seconds,
        qubits_per_tile_round=(cs.qubit_rounds / tile_rounds) if tile_rounds else 0.0,
        patch_live=patch_live,
        tile_occupancy={t: tuple(sorted(rs)) for t, rs in tile_sets.items()},
    )
