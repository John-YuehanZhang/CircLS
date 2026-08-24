"""Resource-statistics aggregator.

Two layers:
  circuit_stats     — works on ANY ``stim.Circuit`` (ours or a baseline's):
                      S4 qubits_active / S5 qubits_allocated / T1 measurement
                      layers / ticks (internal) / V2 qubit_rounds.
  experiment_stats  — tile-level metrics for a ``SequentialPPMExperiment``
                      (S2/S3/V1/L1/L2/L4/T4), derived zero-intrusion from the
                      built circuit + the experiment's geometry.
"""
from circls.metrics.circuit_stats import CircuitStats, circuit_stats
from circls.metrics.experiment_stats import (ExperimentStats, experiment_stats,
                                             timed_build)

__all__ = ["CircuitStats", "circuit_stats",
           "ExperimentStats", "experiment_stats", "timed_build"]
