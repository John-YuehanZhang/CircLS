"""Report assembly for the resource aggregator: JSON export, a readable
summary table, and the report-level sanity checks."""
from __future__ import annotations

import dataclasses
import json
from typing import List, Optional

from circls.metrics.circuit_stats import CircuitStats
from circls.metrics.experiment_stats import ExperimentStats


def sanity_notes(es: ExperimentStats) -> List[str]:
    """Report-level checks (the hard S4==S5 gate already ran inside
    ``experiment_stats``).  Returns human-readable warnings; empty = clean."""
    notes = []
    d = es.distance
    lo, hi = d * d, 2 * (d + 1) ** 2       # one patch's worth of qubits per tile
    if es.tile_rounds and not (lo <= es.qubits_per_tile_round <= hi):
        notes.append(
            f"V1<->V2 density {es.qubits_per_tile_round:.1f} qubits/tile-round "
            f"outside the sanity band [{lo}, {hi}] for d={d}")
    if es.volume_blocks > es.bbox_volume_blocks + 1e-9:
        notes.append("V1 exceeds V3 (occupied volume larger than its bounding box?)")
    return notes


def to_dict(es: ExperimentStats, *, detail: bool = False) -> dict:
    """JSON-safe dict.  ``detail=True`` keeps per-tile occupancy and per-patch
    live ranges; the per-qubit maps stay out either way (bulky)."""
    cs = es.circuit
    out = {
        "S1_tiles_bbox": es.tiles_bbox,
        "S2_tiles_touched": es.tiles_touched,
        "S3_tiles_peak": es.tiles_peak,
        "S4_qubits_active": cs.qubits_active,
        "S5_qubits_allocated": cs.qubits_allocated,
        "T1_rounds": cs.measurement_layers,
        "T2_logical_timesteps": es.logical_timesteps,
        "T4_compile_seconds": es.compile_seconds,
        "V1_volume_blocks": es.volume_blocks,
        "V2_qubit_rounds": cs.qubit_rounds,
        "V3_bbox_volume_blocks": es.bbox_volume_blocks,
        "L1_live_tile_rounds": es.live_tile_rounds,
        "L2_utilization": es.utilization,
        "L4_reuse_rate": es.reuse_rate,
        "d": es.distance,
        "internal": {
            "ticks": cs.ticks,
            "tile_rounds": es.tile_rounds,
            "qubit_rounds_span": cs.qubit_rounds_span,
            "qubits_per_tile_round": es.qubits_per_tile_round,
            "num_detectors": cs.num_detectors,
            "num_observables": cs.num_observables,
        },
        "sanity_notes": sanity_notes(es),
    }
    if detail:
        out["patch_live"] = dict(es.patch_live)
        out["tile_occupancy"] = {f"{a},{b}": list(rs)
                                 for (a, b), rs in es.tile_occupancy.items()}
    return out


def to_json(es: ExperimentStats, path: Optional[str] = None, *,
            detail: bool = False) -> str:
    text = json.dumps(to_dict(es, detail=detail), indent=2)
    if path is not None:
        with open(path, "w") as f:
            f.write(text)
    return text


def format_summary(es: ExperimentStats, title: str = "") -> str:
    cs = es.circuit
    rows = [
        ("S1 tiles_bbox", es.tiles_bbox),
        ("S2 tiles_touched", es.tiles_touched),
        ("S3 tiles_peak", es.tiles_peak),
        ("S4 qubits_active", cs.qubits_active),
        ("S5 qubits_allocated", cs.qubits_allocated),
        ("T1 rounds", cs.measurement_layers),
        ("T2 logical_timesteps", f"{es.logical_timesteps:.2f}"),
        ("T4 compile_seconds", "-" if es.compile_seconds is None
                               else f"{es.compile_seconds:.2f}"),
        ("V1 volume_blocks", f"{es.volume_blocks:.2f}"),
        ("V2 qubit_rounds", cs.qubit_rounds),
        ("V3 bbox_volume_blocks", f"{es.bbox_volume_blocks:.2f}"),
        ("L1 live_tile_rounds", es.live_tile_rounds),
        ("L2 utilization", f"{es.utilization:.3f}"),
        ("L4 reuse_rate", "-" if es.reuse_rate is None else f"{es.reuse_rate:.2f}"),
    ]
    w = max(len(k) for k, _ in rows)
    lines = [title] if title else []
    lines += [f"  {k:<{w}}  {v}" for k, v in rows]
    for note in sanity_notes(es):
        lines.append(f"  !! {note}")
    return "\n".join(lines)


def format_circuit_summary(cs: CircuitStats, title: str = "") -> str:
    """Layer-1-only table (for foreign circuits, e.g. a baseline's .stim)."""
    rows = [
        ("S4 qubits_active", cs.qubits_active),
        ("S5 qubits_allocated", cs.qubits_allocated),
        ("T1 rounds", cs.measurement_layers),
        ("V2 qubit_rounds", cs.qubit_rounds),
        ("   qubit_rounds_span", cs.qubit_rounds_span),
        ("   ticks", cs.ticks),
        ("   detectors/observables", f"{cs.num_detectors}/{cs.num_observables}"),
    ]
    w = max(len(k) for k, _ in rows)
    lines = [title] if title else []
    lines += [f"  {k:<{w}}  {v}" for k, v in rows]
    return "\n".join(lines)
