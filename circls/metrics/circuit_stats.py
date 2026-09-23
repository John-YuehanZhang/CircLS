"""Layer 1 of the resource aggregator: statistics from a bare ``stim.Circuit``.

Works on any circuit — ours or a baseline's (e.g. tqec's) — so cross-compiler
tables use one measuring stick.  Conventions:

* Rounds are delimited by *measurement layers*: the circuit is split into
  moments at ``TICK`` boundaries, and every moment containing at least one
  measurement instruction closes a round.  This matches both our builder
  (one ancilla-measurement moment per SE round) and tqec's blocks (one
  ``MX``/``M`` moment per round).
* ``qubits_active`` (S4) is the set of qubits touched by an *operational*
  instruction — unitary, reset, or measurement.  Noise channels and
  annotations (``QUBIT_COORDS``, ``DETECTOR``, …) do not make a qubit active.
* ``qubits_allocated`` (S5) counts every qubit index referenced anywhere,
  including coordinate declarations — the "declared universe" a noise model
  would bill (tqec's ``num_qubits`` convention).
* ``qubit_rounds`` (V2) bills *holding intervals*: a qubit holds state from a
  reset to its last touch before the next reset (idling included), and is
  free in between.  A qubit that is measured out and later re-reset (a
  retired cell reused as corridor) is NOT billed for the gap.  A destructive
  single-qubit measurement (``M``/``MX``/``MY``) closes the hold: measuring
  the same qubit again with no reset and no other operation in between (a
  redundant readout of a cell that was already measured out) does not extend
  it, while any gate on the qubit reopens the hold.  Product measurements
  (``MPP``, ``MXX``, ...) are not destructive and never close a hold.  For
  circuits without qubit reuse this equals the simpler first-to-last span,
  which is also reported (``qubit_rounds_span``).
"""
from __future__ import annotations

import bisect
import dataclasses
from typing import Dict, List, Tuple

import stim


@dataclasses.dataclass(frozen=True)
class CircuitStats:
    qubits_allocated: int        # S5: distinct qubit indices referenced anywhere
    qubits_active: int           # S4: distinct qubits touched by gates/resets/measurements
    measurement_layers: int      # T1 (circuit-level rule): number of moments containing a measurement
    ticks: int                   # internal field only (T3 demoted from the published set)
    qubit_rounds: int            # V2: sum over active qubits of holding-interval rounds
    qubit_rounds_span: int       # first-to-last variant (== qubit_rounds without reuse)
    num_detectors: int
    num_observables: int
    live_rounds: Dict[int, Tuple[int, int]]  # active qubit -> (first_round, last_round), 1-based
    occupancy: Dict[int, Tuple[Tuple[int, int], ...]]  # active qubit -> merged holding round-intervals


_DESTRUCTIVE = frozenset({"M", "MX", "MY"})   # canonical names of the single-qubit collapsing measurements


def _gate_kind(name: str, _cache: dict = {}) -> Tuple[bool, bool, bool, bool]:
    """Return ``(is_operational, produces_measurements, is_reset, is_destructive_measurement)`` for a gate."""
    hit = _cache.get(name)
    if hit is None:
        gd = stim.gate_data(name)
        operational = bool(gd.is_unitary or gd.is_reset or gd.produces_measurements)
        hit = (operational, bool(gd.produces_measurements), bool(gd.is_reset), gd.name in _DESTRUCTIVE)
        _cache[name] = hit
    return hit


def circuit_stats(circuit: stim.Circuit) -> CircuitStats:
    flat = circuit.flattened()

    allocated: set[int] = set()
    # per active qubit: holding segments in LAYER units [(start, last), ...];
    # a reset starts a new segment (MR extends the previous one to this layer
    # first, so ancilla measure+reset chains stay contiguous in round units).
    # ``closed`` holds the qubits whose current segment ended in a destructive
    # measurement: a further destructive measurement with no reset or gate in
    # between measures a qubit that holds nothing and does not extend the hold.
    segments: Dict[int, List[Tuple[int, int]]] = {}
    closed: set[int] = set()
    meas_layers: list[int] = []

    layer = 0
    for inst in flat:
        name = inst.name
        if name == "TICK":
            layer += 1
            continue
        targets = inst.targets_copy()
        for t in targets:
            qv = t.qubit_value
            if qv is not None:
                allocated.add(qv)
        operational, measures, resets, destructive = _gate_kind(name)
        if not operational:
            continue
        if measures and (not meas_layers or meas_layers[-1] != layer):
            meas_layers.append(layer)
        for t in targets:
            qv = t.qubit_value
            if qv is None:
                continue
            segs = segments.setdefault(qv, [])
            if resets:
                if segs and measures:      # MR: the measurement closes the old hold here
                    segs[-1] = (segs[-1][0], layer)
                segs.append((layer, layer))
                closed.discard(qv)
            elif destructive and qv in closed:
                continue                   # re-measuring a measured-out qubit: nothing is held
            elif segs:
                segs[-1] = (segs[-1][0], layer)
            else:
                segs.append((layer, layer))
            if destructive:
                closed.add(qv)
            else:
                closed.discard(qv)

    n_rounds = len(meas_layers)

    def round_end(lyr: int) -> int:
        # round r spans the moments after measurement layer r-1, up to and
        # including measurement layer r.  Ops past the last measurement layer
        # are clamped onto the final round.
        return min(bisect.bisect_left(meas_layers, lyr), n_rounds - 1) + 1

    def round_start(lyr: int) -> int:
        # A holding interval that STARTS in the same moment as a measurement
        # layer (the builder appends the next reset without a leading TICK)
        # begins holding strictly after that round: bill it from the next
        # round on.  For starts on non-measurement moments this equals
        # round_end.
        return min(bisect.bisect_right(meas_layers, lyr), n_rounds - 1) + 1

    live: Dict[int, Tuple[int, int]] = {}
    occupancy: Dict[int, Tuple[Tuple[int, int], ...]] = {}
    qubit_rounds = 0
    qubit_rounds_span = 0
    if n_rounds:
        for q, segs in segments.items():
            ivals: List[Tuple[int, int]] = []
            for s, e in segs:
                lo, hi = round_start(s), round_end(e)
                if lo > hi:                           # boundary-moment-only segment
                    lo = hi
                if ivals and lo <= ivals[-1][1]:      # overlap in round units
                    ivals[-1] = (ivals[-1][0], max(ivals[-1][1], hi))
                else:
                    ivals.append((lo, hi))
            occupancy[q] = tuple(ivals)
            live[q] = (ivals[0][0], ivals[-1][1])
            qubit_rounds += sum(hi - lo + 1 for lo, hi in ivals)
            qubit_rounds_span += ivals[-1][1] - ivals[0][0] + 1

    return CircuitStats(
        qubits_allocated=len(allocated),
        qubits_active=len(segments),
        measurement_layers=n_rounds,
        ticks=flat.num_ticks,
        qubit_rounds=qubit_rounds,
        qubit_rounds_span=qubit_rounds_span,
        num_detectors=circuit.num_detectors,
        num_observables=circuit.num_observables,
        live_rounds=live,
        occupancy=occupancy,
    )
