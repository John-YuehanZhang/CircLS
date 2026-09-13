"""The uniform circuit-level noise pass used for every circuit in the paper's
comparison, applied to an already-compiled noiseless circuit, so that no
compiler's built-in noise model takes part and both sides are sampled under
exactly the same model.

Model (uniform circuit-level, parameter p):
  - after every 1q unitary:  DEPOLARIZE1(p) on its targets
  - after every 2q unitary:  DEPOLARIZE2(p) on its pairs
  - every measurement:       flip probability p (M -> M(p), MX -> MX(p), ...)
  - after every reset:       X_ERROR(p) (Z_ERROR for RX)
  - idle qubits per TICK moment: DEPOLARIZE1(p) on every qubit that is
    gated somewhere in the circuit but idle in this moment (boundary
    qubits that are NEVER gated get no noise — they touch no detector).

The circuit is flattened first (REPEAT blocks expanded) so moments are
literal TICK spans.
"""
from __future__ import annotations

import stim

_1Q = {"H", "X", "Y", "Z", "S", "S_DAG", "SQRT_X", "SQRT_X_DAG",
       "SQRT_Y", "SQRT_Y_DAG", "C_XYZ", "C_ZYX", "H_XY", "H_XZ", "H_YZ", "I"}
_2Q = {"CX", "CY", "CZ", "XCX", "XCY", "XCZ", "YCX", "YCY", "YCZ",
       "SWAP", "ISWAP", "ISWAP_DAG", "CNOT", "ZCX", "ZCY", "ZCZ",
       "SQRT_XX", "SQRT_XX_DAG", "SQRT_YY", "SQRT_YY_DAG",
       "SQRT_ZZ", "SQRT_ZZ_DAG"}
_MEAS = {"M": "M", "MZ": "MZ", "MX": "MX", "MY": "MY",
         "MR": "MR", "MRZ": "MRZ", "MRX": "MRX", "MRY": "MRY"}
_RESET = {"R": "X_ERROR", "RZ": "X_ERROR", "RX": "Z_ERROR", "RY": "X_ERROR"}
_ANNOT = {"DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS",
          "TICK", "MPAD"}


def inject_uniform_noise(circuit: stim.Circuit, p: float) -> stim.Circuit:
    flat = circuit.flattened()
    gated = set()
    for inst in flat:
        if inst.name not in _ANNOT:
            for t in inst.targets_copy():
                if t.is_qubit_target:
                    gated.add(t.value)
    out = stim.Circuit()
    moment_touched: set = set()

    def close_moment():
        idle = sorted(gated - moment_touched)
        if idle:
            out.append("DEPOLARIZE1", idle, p)
        moment_touched.clear()

    for inst in flat:
        name = inst.name
        if name == "TICK":
            close_moment()
            out.append("TICK")
            continue
        qs = [t.value for t in inst.targets_copy() if t.is_qubit_target]
        if name in _MEAS:
            out.append(_MEAS[name], qs, p)
            moment_touched.update(qs)
            continue
        out.append(inst)
        if name in _ANNOT:
            continue
        moment_touched.update(qs)
        if name in _1Q:
            out.append("DEPOLARIZE1", qs, p)
        elif name in _2Q:
            out.append("DEPOLARIZE2", qs, p)
        elif name in _RESET:
            out.append(_RESET[name], qs, p)
        elif name.startswith("DEPOLARIZE") or "ERROR" in name:
            raise ValueError(f"input circuit already contains noise: {name}")
        else:
            raise ValueError(f"unhandled instruction {name!r} — extend the "
                             f"noise pass deliberately, never silently")
    close_moment()
    return out
