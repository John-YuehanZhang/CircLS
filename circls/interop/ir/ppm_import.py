"""Map an X/Z-only :class:`PauliCircuit` into LightStim's PPM machinery.

DIRECT-JOINT MODEL (model "a").  Each Pauli-product op ``M(P)`` becomes ONE
:class:`PPMStep` that joins the data patches in ``P`` DIRECTLY (patch q_i in
basis X or Z).

Geometry note: the produced experiment gives every patch a PLACEHOLDER
position/orientation — it is valid for schedule-level inspection, NOT for a
full geometric ``build()`` of arbitrary high-weight joints.

(History: this module used to also expose ``check_two_colorable``, a
2-colourability test of the per-PPM bus assignment.  The 2-colouring bus
planner was removed 2026-07-28 — under rule table v3 every measurement/type
combination is constructible and mid-sequence convention switches are realised
by physical rotations, so a global bus-consistency check has no application.)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from circls.core.sequential_ppm_ls import PPMStep, SequentialPPMExperiment
from circls.core.routed_multi_patch_ls import PatchSpec, origin_of

from circls.interop.ir.pauli_ir import PauliCircuit


def patch_name(qubit: int) -> str:
    """Stable patch name for a circuit qubit index."""
    return f"q{qubit}"


def to_ppm_steps(circuit: PauliCircuit, include_measurements: bool = True) -> List[PPMStep]:
    """Turn each X/Z op into a direct-joint :class:`PPMStep`.

    Identity ops (no support) are dropped.  Raises if any op still contains Y —
    run :func:`circls.interop.ir.yfree.to_y_free` first.
    """
    steps: List[PPMStep] = []
    for op in circuit.ops:
        if op.has_y():
            raise ValueError(
                f"to_ppm_steps needs X/Z-only ops but {op} contains Y; "
                f"run circls.interop.ir.yfree.to_y_free(circuit) first")
        if op.kind == "m" and not include_measurements:
            continue
        interaction = [(patch_name(q), op.paulis[q]) for q in op.qubits]
        if not interaction:
            continue
        steps.append(PPMStep(interaction))
    return steps


def _placeholder_specs(names, distance: int = 3) -> List[PatchSpec]:
    """Placeholder patch specs on a coarse line — geometry is unused by the
    2-colouring planner, only names/orientation labels matter."""
    return [
        PatchSpec(nm, origin_of(2 * i, 0, distance, seam=True), distance, "X_vertical")
        for i, nm in enumerate(sorted(names))
    ]


def to_experiment(circuit: PauliCircuit, distance: int = 3, **kw) -> SequentialPPMExperiment:
    """Build a schedule-level :class:`SequentialPPMExperiment` (direct-joint model).

    Every patch gets a placeholder layout and a dummy Z initial/final state, which
    is sufficient for schedule-level inspection.  Extra keyword args pass
    through to the experiment constructor.
    """
    steps = to_ppm_steps(circuit)
    names = {nm for s in steps for nm, _ in s.interaction_type}
    specs = _placeholder_specs(names, distance)
    states = {nm: "Z" for nm in names}
    return SequentialPPMExperiment(
        specs, steps, initial_states=states, final_measure_states=states, **kw)
