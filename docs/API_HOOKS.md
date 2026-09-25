# Customization hooks (design, 2026-08-13)

CircLS separates decisions (Section 3 of the paper) from the
stabilizer construction and circuit emission (Section 4).  The six
decision points below are open for customization; the construction,
the emitted circuit, and the verification suite are not.  Whatever a
hook injects, the compiled circuit still passes the same legality
checks and verification (p=0 silence, graphlike distance, logical
simulation match).

Two ways in, one contract:

- **Result injection** (primary): hand us the finished decision as
  plain data.  Your algorithm runs in your world; we validate and
  compile.
- **Strategy injection** (for state-dependent decisions): hand us a
  callable that we invoke inside the compile loop.  A strategy's
  return value is exactly the result-injection data structure.

## Entry points

```python
compile_qasm(qasm, distance=3, ..., hooks...)         # existing
compile_ppm_sequence(program, distance=3, ..., hooks...)  # NEW: enter
    # with a PPMProgram (any PBC front-end output), skip the QASM
    # front-end entirely.
```

Both share the same hook keywords.

## The six hooks

| hook | kind | signature / data | default |
|---|---|---|---|
| `reselector=` | result or callable | `MPauliProgram -> (MPauliProgram, recon)` | minimum-weight re-selection (`measure_reduction=True`) |
| `scheduler=` | result or callable | `Program -> (Program, perm)`; perm maps new op pos to old | lifetime-aware `schedule_ops` |
| `placement=` | result | `{patch_name: tile}` | `assignment=` portfolio ('row_major' / 'optimized') |
| `router=` | callable | `route(board_state, step) -> Corridor` | corridor search (Find Corridor) |
| `lifetime=` | result | `{patch_name: (init_layer, free_layer)}` — allocated at init_layer, freed at free_layer | first-use init / last-use free; `keep_patches` pins output patches |
| `orientation=` | result | `{patch_name: "X_horizontal" \| "X_vertical"}` — the patch's birth orientation | derived from the first-use letter; gadget ancillas under `magic_proxy='X'` face the step that consumes them (`mapping.magic_orientations`) |

Validation on injection (reject loudly, never repair silently):

- `reselector`: the returned measurement set must generate the same
  commuting group; checked by the same algebraic identity the test
  suite uses for the built-in re-selection.
- `scheduler`: only commuting ops may swap; anticommuting pairs must
  keep their order (re-measured on the returned permutation).
  Weight-1 fixed ops keep absolute positions.
- `placement`: tiles exist on the floor, no two patches share a
  tile, orientations admit a legal construction.
- `router`: returned corridor must be connected, on free tiles, and
  touch every measured patch; the construction rules then run on it
  unchanged (an illegal corridor fails the same checks as an
  internal one would).  Consulted only where a corridor search would
  run — see the precedence table in the router section.
- `lifetime`: init_layer <= first use, free_layer >= last use, and
  no patch in `keep_patches` may be freed early.
- `orientation`: a partial dict is fine (unnamed patches keep the
  derived orientation); values must be `X_horizontal` or
  `X_vertical`; gadget ancillas cannot be overridden under the default
  `magic_proxy='Y'` (the Gidney birth layout is protocol-fixed
  `X_vertical`); under `magic_proxy='X'` they may be, and otherwise
  default to `mapping.magic_orientations`.  Composes with `placement=` and with
  `assignment='optimized'` — the optimizer plans its cells against
  the forced orientations.  It fixes the BIRTH orientation only:
  with `auto_rotate` on, the rotation planner may still rotate the
  patch during execution (pass `auto_rotate=False` to freeze it),
  and an orientation that admits no legal construction fails the
  same construction checks as a derived one would.

## Interaction with existing flags

The existing booleans stay as shorthand: `measure_reduction=False`
== `reselector=None` (identity); `assignment="optimized"` is the
built-in placement search; `step_scheduling=False` freezes program
order.  A hook and its flag together are an error.

`step_scheduling=` and `parallel_steps=` are orthogonal:
`step_scheduling=False` keeps the PPM order verbatim but compatible
neighbors still merge into shared execution windows; fully serial,
order-verbatim execution needs `step_scheduling=False,
parallel_steps=False` (the ablation's no_schedpar configuration).

## Out of scope (deliberately closed)

The four-case seam table, the five corridor construction rules,
detector/observable annotation, and the verification suite.  These
define what a CircLS circuit IS.

Also closed: the internal router's search depth — the exact-search
group cap and `route_and_build`'s candidate-pool knobs (`per_z`,
`max_std`, `keepout`, ...).  They tune OUR searcher, not the routing
decision, and their semantics track the pool implementation.  To
control routing from outside, replace the decision with `router=` or
pin a corridor with `PPMStep.route`.

## Worked examples

Every snippet below is the pattern the test suite runs
(`tests/test_api_hooks.py`); copy it as-is.

### Inject your own placement (result injection)

```python
from circls.pipeline import analyze_qasm, compile_qasm

# front-end only: the PPM structure your algorithm optimizes against
info = analyze_qasm(qasm)      # .interactions, .patch_names, .first_letters
placement = my_mapper(info.interactions)   # {patch name: coarse cell}
out = compile_qasm(qasm, distance=3, placement=placement)
# missing patches, duplicate cells, or placement together with
# assignment="optimized" raise ValueError before anything compiles.
```

`analyze_qasm` runs only the front-end passes (load, re-selection,
ordering, Y-elimination, scheduling, gadget expansion) — pass it the
same flags you will pass to `compile_qasm` so the analyzed steps match
the compiled ones.

### Inject a re-selection (callable; a precomputed result works too)

```python
from circls.compiler.measure_reduce import Reconstruction

def my_reselect(raw):
    # rewrite the terminal measurement set any way you like, and say
    # how each original bit is reconstructed from the executed set:
    #   bit(original[k]) = const[k] XOR XOR(bit(executed[j])
    #                                        for j in inverse[k])
    red = ...            # your PauliCircuit of terminal m ops
    recon = Reconstruction(support=..., inverse=..., const=...)
    return red, recon

out = compile_qasm(qasm, distance=3, reselector=my_reselect)
# the return value must pass the exact operator identity; anything
# that drops information is rejected with ValueError.
```

To turn the built-in re-selection off entirely, keep using
`measure_reduction=False`.

### Inject an order (callable over the swept program)

```python
def my_schedule(circ):
    perm = my_solver(circ.ops)     # perm[new_pos] = old_pos
    return reorder(circ, perm), perm

out = compile_qasm(qasm, distance=3, scheduler=my_schedule)
# contract: fixed ops keep absolute positions, anticommuting pairs
# keep their order — violated permutations raise ValueError.
```

### Inject a router (strategy; the one state-dependent decision)

```python
def my_router(live_specs, step):
    # live_specs: the PatchSpecs alive at this layer (name, origin,
    # orientation); step.interaction_type lists (patch, Pauli).
    cells = my_corridor_search(live_specs, step)
    return cells         # list of coarse cells, or None to fall back

out = compile_qasm(qasm, distance=3, router=my_router)
# returned cells build through the same deterministic constructor and
# are gated by the same oracle as an internal route.
```

Per step the corridor is resolved in this order; the hook is
consulted only when every earlier source is absent:

1. `PPMStep.route` — an explicit corridor on the step (result
   injection) always wins.
2. A cell-adjacent two-patch step builds the zero-cell direct seam;
   there is no corridor to search, so the hook is not consulted.
3. A step inside a parallel window builds the corridor the window
   planner committed to jointly; the hook is not consulted.
4. The `router=` hook; returning `None` falls through.
5. The internal corridor search.

The hook is asked at most once per step (the answer is cached by the
step's position in the sequence), so it cannot return different
corridors for the planner's different candidate evaluations of the
same step.

### Widen lifetimes (result injection)

```python
# keep patches q0 and q1 alive for the whole program, whatever the
# liveness analysis says (overrides may only WIDEN, never narrow):
out = compile_qasm(qasm, distance=3, measure_reduction=False,
                   lifetime={"q0": (0, 7), "q1": (0, 7)})
```

The pair is a schedule, not just a bound: the patch is allocated and
initialized at `init_layer` (idling until its first use) and, under
`liveness=`, measured out at `free_layer` (idling after its last
use).  `init_layer == first use` and `free_layer == last use`
reproduce the derived behavior; `free_layer` at the final layer means
the patch survives to the terminal readout.  A patch in
`keep_patches` is never freed mid-sequence, whatever its window
says.

## Entering with a PPM sequence

`compile_ppm_sequence` accepts a `PPMProgram`
(`circls.interop.ir.gosc_gadgets`) — the output of any PBC front-end:

```python
from circls.interop.ir.gosc_gadgets import OutBit, PPMProgram, ProgramOp
from circls.pipeline import compile_ppm_sequence

prog = PPMProgram(
    num_data=3,
    ops=[
        # jointly measure X0 X1, then X1 X2, then read every qubit out
        ProgramOp("mpp", {0: "X", 1: "X"}),
        ProgramOp("mpp", {1: "X", 2: "X"}),
        ProgramOp("mpp", {0: "Z"}),
        ProgramOp("mpp", {1: "Z"}),
        ProgramOp("mpp", {2: "Z"}),
    ],
    gadgets=[],
    # program bit i = record[rec] XOR flip: here bit i is just the
    # i-th measurement record, unflipped
    out_bits=[OutBit(rec=k, flip=0) for k in range(5)],
)
out = compile_ppm_sequence(prog, distance=3)
```

Each `ProgramOp("mpp", targets)` consumes the next record index in
order; `OutBit(rec, flip, gadgets)` defines a classical program bit
as `record[rec] XOR flip XOR (gadget conditions)`.  The
`placement=`, `orientation=`, `lifetime=` and `router=` hooks apply unchanged;
re-selection and scheduling are front-end passes and do not run on
this entry.  Note one default difference: this entry leaves
`parallel_steps` at the experiment default `False`, while
`compile_qasm` defaults it to `True` — pass
`parallel_steps=True` explicitly to get shared execution windows for
a caller-supplied PPM order.
