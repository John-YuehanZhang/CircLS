"""Build LaSsynth specifications from Clifford circuits / stabilizer states.

Two builders:

  unitary_spec(stim_circuit, n, box)   -> 2n ports (n in on bottom floor,
                                          n out on top floor), 2n stabilizers
                                          derived from the Clifford tableau.
  state_spec(stabilizers, box)         -> n ports (all outputs on top floor),
                                          n stabilizers.

Port ordering convention matches the demo notebook: the p-th character of a
stabilizer string is the Pauli acting on the p-th port in `ports`.
"""
import itertools
from typing import List, Sequence

import stim


def _grid_positions(n: int, max_i: int, max_j: int) -> List[List[int]]:
    """Place n patches on distinct (i,j) grid points, checkerboard first."""
    cells = [(i, j) for i in range(max_i) for j in range(max_j)]
    # prefer a checkerboard so patches are not immediately adjacent
    cells.sort(key=lambda c: ((c[0] + c[1]) % 2, c[0], c[1]))
    if n > len(cells):
        raise ValueError(f"{n} qubits do not fit in {max_i}x{max_j}")
    return [list(c) for c in cells[:n]]


def unitary_flows(circuit: stim.Circuit, n: int) -> List[str]:
    """Stabilizer flows of a Clifford unitary: P_in (x) U P_in U^dag on out.

    Returns 2n paulistrings of length 2n over ports [in_0..in_{n-1},
    out_0..out_{n-1}], using '.' for identity (LaSsynth's identity char).
    """
    t = circuit.to_tableau()
    strings = []
    for q in range(n):
        for basis in ("X", "Z"):
            img = t.x_output(q) if basis == "X" else t.z_output(q)
            inp = ["."] * n
            inp[q] = basis
            out = []
            for p in range(n):
                out.append(".XYZ"[img[p]])
            strings.append("".join(inp) + "".join(out))
    return strings


def unitary_spec(circuit: stim.Circuit, n: int, max_i: int, max_j: int,
                 max_k: int, z_basis_direction: str = "J") -> dict:
    pos = _grid_positions(n, max_i, max_j)
    ports = []
    for q in range(n):  # inputs on the bottom floor, pipe goes up
        ports.append({
            "location": [pos[q][0], pos[q][1], 0],
            "direction": "+K",
            "z_basis_direction": z_basis_direction,
        })
    for q in range(n):  # outputs on the top floor, pipe goes down
        ports.append({
            "location": [pos[q][0], pos[q][1], max_k],
            "direction": "-K",
            "z_basis_direction": z_basis_direction,
        })
    return {
        "max_i": max_i,
        "max_j": max_j,
        "max_k": max_k,
        "ports": ports,
        "stabilizers": unitary_flows(circuit, n),
    }


def state_spec(stabilizers: Sequence[str], max_i: int, max_j: int, max_k: int,
               z_basis_direction: str = "J") -> dict:
    n = len(stabilizers[0])
    pos = _grid_positions(n, max_i, max_j)
    ports = [{
        "location": [pos[q][0], pos[q][1], max_k],
        "direction": "-K",
        "z_basis_direction": z_basis_direction,
    } for q in range(n)]
    return {
        "max_i": max_i,
        "max_j": max_j,
        "max_k": max_k,
        "ports": ports,
        "stabilizers": list(stabilizers),
    }


def ghz_stabilizers(n: int) -> List[str]:
    """GHZ / cat state generators: X^n, and Z_q Z_{q+1}."""
    stabs = ["X" * n]
    for q in range(n - 1):
        s = ["."] * n
        s[q] = "Z"
        s[q + 1] = "Z"
        stabs.append("".join(s))
    return stabs
