"""Component-circuit harnesses: idle/rotation/PPM circuits isomorphic to the
sequential pipeline.

Isomorphism convention (aligned with circls.core.sequential_ppm_ls._alloc_patch):
every patch is a ``RotatedSurfaceCode(distance=d)`` with
offset = origin_of(a, b, d, seam=True) - 1; SE uses
DiagonalSurfaceCodeExtractionBlock (the same schedule as the merge window).
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT.parent / "LightStim"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from lightstim.ir.qec_system import QECSystem                      # noqa: E402
from lightstim.ir.tracker import SyndromeTracker                   # noqa: E402
from lightstim.ir.builder import CircuitBuilder                    # noqa: E402
from lightstim.qec_code.surface_code.rotated import RotatedSurfaceCode  # noqa: E402
from lightstim.qec_code.surface_code.rotated.diagonal_se import (  # noqa: E402
    DiagonalSurfaceCodeExtractionBlock)
from lightstim.qec_code.surface_code.rotated.swap_rotation import rotate_patches_swap  # noqa: E402

from circls.core.routed_multi_patch_ls import origin_of                 # noqa: E402


def _system(coords: dict, d: int) -> QECSystem:
    system = QECSystem()
    for nm, (a, b) in coords.items():
        origin = origin_of(a, b, d, seam=True)
        p = RotatedSurfaceCode(distance=d)
        system.add_patch(p, name=nm, offset=(origin[0] - 1, origin[1] - 1))
    return system


def _builder(system: QECSystem) -> CircuitBuilder:
    tracker = SyndromeTracker(num_qubits=system.num_qubits,
                              expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker=tracker, system_config=system,
                             if_detector=True)
    system.register_tracker(tracker)
    system.register_builder(builder)
    builder.write_coordinates()
    return builder


def memory_builder(coords: dict, d: int, rounds_body: int,
                   rounds_init: int = 1, basis="Z") -> CircuitBuilder:
    """init -> baseline SE(rounds_init) -> SE(rounds_body) -> readout.
    rounds_body=0 gives the null circuit (init+baseline+readout cost only).
    ``basis``: one string for all, or {patch name: basis} to set it per patch."""
    system = _system(coords, d)
    builder = _builder(system)
    data = sorted(system.data_indices)
    owner = system.index_to_owner_map
    if isinstance(basis, dict):
        bmap = {q: basis[owner[q]] for q in data}
    else:
        bmap = {q: basis for q in data}
    builder.initialize(bmap, n=system.num_qubits)
    block = DiagonalSurfaceCodeExtractionBlock(system).circuit
    builder.apply_syndrome_extraction(block, rounds=rounds_init)
    if rounds_body:
        builder.apply_syndrome_extraction(block, rounds=rounds_body)
    builder.apply_data_readout(final_measurements=dict(bmap))
    return builder


def rotation_builder(coords: dict, d: int, rotate_names,
                     rounds_init: int = 1, rounds_after: int = None,
                     basis: str = "Z") -> CircuitBuilder:
    """init -> baseline SE -> SWAP rotation (concurrent) -> SE(rounds_after=d)
    -> readout."""
    if rounds_after is None:
        rounds_after = d
    system = _system(coords, d)
    builder = _builder(system)
    data = sorted(system.data_indices)
    builder.initialize({q: basis for q in data}, n=system.num_qubits)
    block = DiagonalSurfaceCodeExtractionBlock(system).circuit
    builder.apply_syndrome_extraction(block, rounds=rounds_init)
    rotate_patches_swap(system, builder, list(rotate_names))
    builder.apply_syndrome_extraction(
        DiagonalSurfaceCodeExtractionBlock(system).circuit, rounds=rounds_after)
    builder.apply_data_readout(final_measurements={q: basis for q in data})
    return builder


def match_idle_rounds(coords: dict, d: int, target_ticks: int,
                      rounds_init: int = 1, lo: int = 0, hi: int = None) -> int:
    """Pick rounds_body so the idle circuit's tick count is closest to
    target_ticks (matching the exposure duration)."""
    if hi is None:
        hi = 6 * d + rounds_init + 4
    best, best_gap = lo, None
    for r in range(lo, hi + 1):
        ticks = memory_builder(coords, d, r, rounds_init).circuit.num_ticks
        gap = abs(ticks - target_ticks)
        if best_gap is None or gap < best_gap:
            best, best_gap = r, gap
        if ticks >= target_ticks and best_gap == 0:
            break
    return best
