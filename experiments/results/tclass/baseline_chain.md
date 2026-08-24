# Baseline chain on the nine S-proxy core programs

Date: 2026-08-19.  Question: can the paper's baseline (TopoLS + tqec)
produce a runnable stim circuit for the T-class programs compiled
through the S-state proxy?  Answer: no — 0 of 9.

## Protocol

Same chain and versions as the Clifford comparison
(`generate_topols_circuits.py`, `results/topols_circuits/manifest.json`):

* TopoLS at `4c12ea2` (the manifest's pinned SHA; the scratchpad
  checkout is at exactly this commit), MCTS with the code release's
  default settings (`b = 10`, 10000 iterations — the convention the
  appendix records for DJ-16, whose paper-reproducing settings crash).
* tqec at `5909ddec` (= the v0.2.0 release commit, reinstalled from a
  fresh clone of github.com/tqec/tqec).
* Input: the SAME substituted proxy QASM our t_as_s pipeline compiles
  (pure Clifford at the gate level: s/sdg/h/cx/x), unitary part only,
  written into `TopoLS/docs/benchmark/<name>_proxy.qasm`.
* Chain: `docs/prog.py -f <name>_proxy` (MCTS) ->
  `docs/2tqec.py -f <name>_proxy` (bgraph) -> our loader ->
  `fill_ports_for_minimal_simulation()` -> `compile_block_graph` ->
  `generate_stim_circuit(k=1)`.
* One assist beyond the Clifford loader: TopoLS emits bare `'S'`
  marker nodes for S rotations (`trans2tqec.py:437` — it never lowers
  them to a lattice-surgery construction); we mapped each marker
  (a degree-1 leaf) to tqec's own `YHalfCube` so the back end could
  try.  Ports and fills as in the Clifford protocol.

## Per-program outcome

| program          | stage reached | failure |
|------------------|---------------|---------|
| teleportation_n3 | tqec compile (both fills) | `NotImplementedError: Y cube is not implemented.` |
| qec_en_n5        | TopoLS MCTS   | `UnboundLocalError: 'block_state'` |
| toffoli_n3       | TopoLS MCTS   | `KeyError: 14` (same with `-zx 0` and seed 7) |
| fredkin_n3       | TopoLS MCTS   | `KeyError: 11` |
| bell_n4          | TopoLS MCTS   | `KeyError: 12` |
| adder_n4         | TopoLS MCTS   | `KeyError: 27` |
| simon_n6         | TopoLS MCTS   | `KeyError: 17` |
| multiply_n13     | TopoLS MCTS   | `KeyError: 27` |
| sat_n7           | TopoLS MCTS   | `UnboundLocalError: 'block_state'` |

A synthetic minimum (`h; s; cx` on two qubits) passes MCTS and
conversion; with the S marker mapped to a YHalfCube, tqec's
precondition check rejects the placement itself: "Z(1/2) spider must
connect to the time direction, but Z(1/2) at (2,-1,1) connects to
(1,-1,1)" — TopoLS's embedding does not know the Y cube's geometric
constraint, so even the marker positions are unusable.

## Reading

The failure is three-layered and version-pinned: the front end's
embedding search crashes on eight of nine proxy inputs; the converter
never lowers S to a native construction (bare markers only); and the
back end's Y-cube circuit generation is unimplemented at the pinned
release.  The inputs are pure-Clifford QASM — this is not a
magic-state limitation but an S-gate one, so the real T-class
programs are out of reach a fortiori.
