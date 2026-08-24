"""Interop front-ends: turn external fault-tolerant compilers' Pauli-product
output into LightStim :class:`SequentialPPMExperiment` inputs, and check whether
the resulting per-PPM bus assignment is 2-colourable.

ISOLATION CONTRACT — this subpackage is OPTIONAL and MUST NOT be imported by the
LightStim core (``lightstim/__init__.py`` never imports ``lightstim.interop``).
Layout: :mod:`circls.interop.ir` is the front-end-agnostic layer — the
:class:`PauliCircuit` IR contract and the PBC -> PPM transformation
chain; :mod:`circls.interop.nwqec` is the NWQEC-specific adapter.
Adapters for other front-ends get sibling packages next to ``nwqec``.
The external-tool front-ends (``nwqec.frontend``, ``ir.yfree``) import
their heavy third-party dependency (nwqec / lsqecc) LAZILY, inside the
functions that need it, so importing this package never pulls those in.

Pipeline (direct-joint model — see ir.ppm_import):
    nwqec Circuit --nwqec.frontend--> PauliCircuit (may contain Y)
                  --ir.yfree------->  PauliCircuit (X/Z only, Y removed)
                  --ir.ppm_import-->  SequentialPPMExperiment / 2-colour check
"""
from circls.interop.ir.pauli_ir import PauliOp, PauliCircuit
from circls.interop.ir.ppm_import import (
    to_ppm_steps, to_experiment, patch_name,
)
from circls.interop.ir.gosc_gadgets import OutBit, PPMProgram, ProgramOp

__all__ = [
    "PauliOp", "PauliCircuit",
    "to_ppm_steps", "to_experiment",
    "patch_name",
    "OutBit", "PPMProgram", "ProgramOp",
]
