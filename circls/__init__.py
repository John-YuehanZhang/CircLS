"""CircLS: Clifford QASM / PPM sequences -> verified stim circuits.

Routing, seam-table construction, scheduling and sequential-PPM
experiments on top of the vendored LightStim detector backend.
"""
__version__ = "0.1.0"

from .core.multi_patch_coupler import (PatchSpec, BentLayoutError,
                                  MultiPatchLayout, origin_of,
                                  route_and_build)
from .core.sequential_ppm_ls import PPMStep, SequentialPPMExperiment
from circls.tools.evaluate import inject_uniform_noise, measure_ler, verify
from circls.pipeline import (ProgramAnalysis, analyze_qasm,
                             compile_ppm_sequence, compile_qasm)

__all__ = ["compile_qasm", "compile_ppm_sequence", "analyze_qasm",
           "ProgramAnalysis", "verify",
           "measure_ler", "inject_uniform_noise",
           "PatchSpec", "BentLayoutError", "MultiPatchLayout",
           "origin_of", "route_and_build", "PPMStep",
           "SequentialPPMExperiment", "__version__"]
