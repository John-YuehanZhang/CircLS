"""Best-effort TQEC BlockGraph export for a CircLS-compiled schedule.

Phase A (any python with circls):   extract  -> <name>_blockstructure.json
Phase B (python with tqec):         export   -> <name>_blockgraph.dae/.html

Geometry is built from the SCHEDULE'S OWN SEMANTICS, not from geometric
adjacency: the time axis has one layer per scheduling batch (plus an
initialization layer below and a readout layer on top), a patch's live
range is a column of cubes, and every merge window becomes corridor
cubes in its batch's layer, piped only to the patches that PPM actually
joins.  Two consecutive merges reusing the same corridor tile become two
stacked cubes WITHOUT a pipe (the ancilla region is read out and
re-initialized between them).

Cube walls that the protocol fixes are pinned before the search: a
column's bottom face is its initialization basis, its top face its
readout basis, and a corridor cube's temporal faces are the merge
ancilla's init/readout (|+>/M_X for a Z-type joint, |0>/M_Z for X-type).
Remaining freedom is searched over TQEC's wall rules; among the valid
assignments we choose the one with the most deterministic correlation
surfaces (wrong wall orientations destroy determinism, they never create
it) and reject the export when every assignment has zero.  The number of
deterministic observables of the CircLS-compiled circuit is printed as
context (TQEC's surface-counting convention differs, so equality is not
required).  |Y>-initialized ancilla patches map to tqec's temporal Y
half-cube.
"""
import argparse
import json
import pathlib
import sys

HERE = pathlib.Path.cwd()          # outputs land in the caller's folder
ROOT = pathlib.Path(__file__).resolve().parents[2]

# Filled by extract_from_cp on every call: the mixed-radix digit sizes of
# the seam-side choice points (product = number of embeddings) and the
# patches whose merges need both wall letters on one axis.  A ZXCube has
# one letter per axis (opposite walls match) and temporal pipes carry the
# transverse letters through a column's whole lifetime, so a patch asked
# to open an X wall (Z-type merge) and a Z wall (X-type merge) on the
# same axis admits no assignment -- those embeddings are searched last.
LAST_SCAN = {"radix": [], "conflicts": []}


def _seam_legal(orient, P, t, p):
    """The compiler's parallel-law rule for a seam between patch cell ``p``
    (orientation ``orient``, measured Pauli ``P``) and corridor cell ``t``:
    with X_horizontal a Z measurement attaches east/west and an X
    measurement north/south, with X_vertical the reverse (mirrors
    ``multi_patch_coupler._legal_attach_groups``).  Unknown orientation or
    letter: no restriction."""
    if orient not in ("X_horizontal", "X_vertical") or P not in ("X", "Z"):
        return True
    seam_ew = (t[0] != p[0])
    want = "X_horizontal" if (P == "Z") == seam_ew else "X_vertical"
    return orient == want


def _orient_for_step(exp, i):
    """Per-patch orientation in effect for PPM ``i`` (rotation-aware when
    the experiment exposes it, else the declared orientations)."""
    eff = getattr(exp, "_eff_orient", None)
    if callable(eff):
        try:
            v = eff(i)
            if isinstance(v, dict):
                return dict(v)
        except TypeError:
            pass
    elif isinstance(eff, dict):
        v = eff.get(i)
        if isinstance(v, dict):
            return dict(v)
        if eff and all(isinstance(k, str) for k in eff):
            return dict(eff)
    return dict(getattr(exp, "_orient", {}) or {})


def extract_from_cp(cp, name, distance, out_dir=None, contact_variant=0):
    """Build the block structure for a compiled program (.experiment /
    .circuit / .placement) and write <name>_blockstructure.json into
    ``out_dir`` (default: cwd).  ``contact_variant`` enumerates the
    seam-side choices where a patch is adjacent to more than one corridor
    tile (mixed-radix; variant 0 = the greedy default)."""
    out_base = pathlib.Path(out_dir) if out_dir else HERE
    _cv = [contact_variant]
    scan_radix, scan_need = [], {}      # -> LAST_SCAN at the end
    exp = cp.experiment
    steps = exp.ppm_sequence
    batches = getattr(exp, "_step_batches", None) or [[i] for i in range(len(steps))]
    batch_of = {i: b for b, batch in enumerate(batches) for i in batch}
    n_layers = len(batches) + 2            # init + one per batch + readout
    top = n_layers - 1

    from circls.metrics.experiment_stats import experiment_stats
    stats = experiment_stats(exp, cp.circuit)
    rounds_init = getattr(exp, "rounds_init", 1)
    lifetimes = getattr(exp, "lifetimes", {})   # step-index (first, last)

    cubes, pipes, pins, y_bottoms = set(), set(), {}, set()
    contact_fallbacks = []          # (step, patch) where no legal face was adjacent

    # corridor occupancy per merge layer (z = 1 + batch): a late-born
    # patch's init cube is placed one layer below its first merge, a
    # retired patch's readout cube one layer above its last merge; when a
    # corridor of the neighbouring batch runs through that cell (measured
    # 2026-09-06: cnot_network_subroutine after the register-wise
    # reordering, ancilla (1, 3) at layer 24) the two would pin opposite
    # wall letters on one cube.  In that case the patch is born INTO its
    # first merge layer / retired INTO its last one instead (the init
    # rounds ride the merge window's cube, the way a zero-standalone-round
    # patch is already drawn).
    corridor_pins = {}
    for i, st in enumerate(steps):
        r = exp._routes[i]
        letters = {P for _, P in st.interaction_type}
        wall = {"Z": "X", "X": "Z"}[next(iter(letters))] if len(letters) == 1 else None
        for c in ([tuple(c) for c in r.tree] if (r is not None and r.tree) else []):
            corridor_pins[(c[0], c[1], 1 + batch_of[i])] = wall

    def _clashes(v, letter):
        # only a genuinely contradictory letter moves the cube (an equal or
        # unpinned corridor wall keeps the previous drawing unchanged)
        return v in corridor_pins and corridor_pins[v] is not None and corridor_pins[v] != letter

    def pin(v, letter):
        if pins.setdefault(v, letter) != letter:
            raise SystemExit(f"pin conflict at {v}: {pins[v]} vs {letter} "
                             f"-- not expressible in ZXCube vocabulary")

    # patch columns
    col_range = {}
    max_hi = max((h for _, h in stats.patch_live.values()), default=0)
    for nm, tile in cp.placement.items():
        first, last = lifetimes.get(nm, (None, None))
        if nm not in stats.patch_live and first is None:
            continue                    # never materialized
        # a late-born patch is initialized in the rounds BEFORE its first
        # merge window, so its init cube sits one layer below that merge;
        # symmetrically a retired patch's readout cube sits one layer above
        # its last merge (the temporal faces must not collide with the
        # corridor's own init/readout walls in the merge layer)
        if nm in stats.patch_live:
            lo_r, hi_r = stats.patch_live[nm]
            born = 0 if lo_r <= rounds_init or first is None \
                else batch_of[first]
            died = top if hi_r >= max_hi or last is None \
                else min(top, 2 + batch_of[last])
        else:
            # born into its first merge and retired right after its last one
            # (zero standalone rounds -- absent from patch_live)
            born = batch_of[first]
            died = min(top, 2 + batch_of[last])
        x, y = tile
        init = exp.initial_states.get(nm)
        if first is not None and born == batch_of[first] and init != "Y" and _clashes((x, y, born), init):
            born = 1 + batch_of[first]
        if last is not None and died == 2 + batch_of[last] and _clashes((x, y, died), exp.final_measure_states.get(nm)):
            died = 1 + batch_of[last]
        col_range[nm] = (born, died)
        for z in range(born, died + 1):
            cubes.add((x, y, z))
            if z > born:
                pipes.add(((x, y, z - 1), (x, y, z)))
        init = exp.initial_states.get(nm)
        if init == "Y":
            y_bottoms.add((x, y, born))
        else:
            pin((x, y, born), init)
        pin((x, y, died), exp.final_measure_states.get(nm))

    # merge windows: corridor cubes + semantically-routed pipes
    for i, st in enumerate(steps):
        r = exp._routes[i]
        tree = [tuple(c) for c in r.tree] if (r is not None and r.tree) else []
        z = 1 + batch_of[i]
        letters = {P for _, P in st.interaction_type}
        wall = {"Z": "X", "X": "Z"}[letters.pop()] if len(letters) == 1 else None
        for t in tree:
            cubes.add((t[0], t[1], z))
            if wall:
                pin((t[0], t[1], z), wall)
        # corridor-corridor pipes along the tree
        ts = set(tree)
        for t in tree:
            for dx, dy in ((1, 0), (0, 1)):
                u = (t[0] + dx, t[1] + dy)
                if u in ts:
                    pipes.add(((t[0], t[1], z), (u[0], u[1], z)))
        # corridor-patch pipes: only the PPM's own targets, and exactly ONE
        # contact pipe per patch -- an L-shaped tree can wrap a patch corner
        # and touch it on two sides, but a second pipe would impose
        # contradictory wall letters on the column (the physical seam is one
        # side); assign scarcest-candidates first, prefer unused tiles
        # candidate contact tiles: the tree tiles adjacent to the patch
        # THROUGH A PARALLEL-LAW-LEGAL FACE -- the same rule the compiler's
        # router uses to build its attach groups
        # (multi_patch_coupler._legal_attach_groups), so the pipe lands on
        # the side the compiled circuit really seams.  Before 2026-09-11 any
        # adjacent tree tile was a candidate and the wall-rule search could
        # pick a face the circuit never uses (toffoli_n3 step 1: q1/q2
        # piped from (3, 2) while the seams sit at (4, 3)/(4, 1)), leaving
        # the true contact tiles as dead-end cubes.
        cand = {}
        orient_now = _orient_for_step(exp, i)
        for nm, _P in st.interaction_type:
            px, py = cp.placement[nm]
            adjacent = [t for t in tree
                        if abs(t[0] - px) + abs(t[1] - py) == 1]
            legal = [t for t in adjacent
                     if _seam_legal(orient_now.get(nm), _P, t, (px, py))]
            if legal:
                cand[nm] = legal
            else:
                cand[nm] = adjacent
                if adjacent:
                    contact_fallbacks.append((i, nm))
                    print(f"warning: step {i} target {nm}: no parallel-law-"
                          f"legal contact tile among {adjacent}; falling "
                          f"back to any adjacent tile")
        contacts = {t: 0 for t in tree}
        pmap = dict(st.interaction_type)
        for nm in sorted(cand, key=lambda n: len(cand[n])):
            if not cand[nm]:
                print(f"warning: step {i} target {nm} not adjacent to its "
                      f"corridor tree -- geometry incomplete")
                continue
            ordered = sorted(cand[nm], key=lambda t: (contacts[t], t))
            t = ordered[_cv[0] % len(ordered)]
            _cv[0] //= len(ordered)
            scan_radix.append(len(ordered))
            contacts[t] += 1
            px, py = cp.placement[nm]
            # a P-type merge opens the patch wall whose axis letter is the
            # dual of P; two different letters demanded on one axis = no
            # ZXCube assignment can exist (recorded for the search order)
            need = {"Z": "X", "X": "Z"}.get(pmap[nm])
            if need:
                axis = "x" if t[0] != px else "y"
                scan_need.setdefault((nm, axis), set()).add(need)
            a, b = sorted([(t[0], t[1], z), (px, py, z)])
            pipes.add((a, b))

    det = cp.circuit.compile_detector_sampler(seed=0)
    _, obs = det.sample(512, separate_observables=True)
    n_det_obs = sum(1 for k in range(obs.shape[1])
                    if bool((obs[:, k] == obs[0, k]).all()))
    data = {
        "name": name, "distance": distance, "n_layers": n_layers,
        "cubes": sorted([list(v) for v in cubes]),
        "pipes": sorted([[list(a), list(b)] for a, b in pipes]),
        "pins": sorted([[list(v), s] for v, s in pins.items()]),
        "y_bottoms": sorted([list(v) for v in y_bottoms]),
        "deterministic_observables": n_det_obs,
        "num_observables": int(obs.shape[1]),
        "contact_policy": "parallel-law faces (compiler seam sides)",
        "contact_fallbacks": [[int(i), nm] for i, nm in contact_fallbacks],
    }
    global LAST_SCAN
    LAST_SCAN = {"radix": scan_radix,
                 "conflicts": sorted({nm for (nm, ax), ls in scan_need.items()
                                      if len(ls) > 1})}
    out = out_base / f"{name}_blockstructure.json"
    out.write_text(json.dumps(data, indent=1))
    print(f"wrote {out.name}: {len(data['cubes'])} cubes, "
          f"{len(data['pipes'])} pipes, {len(data['pins'])} pinned walls, "
          f"{len(data['y_bottoms'])} Y half-cube(s), {data['n_layers']} layers")
    return data


def auto_export(cp, name, blockgraph_dir, distance=3, tqec_python=None,
                max_variants=256, time_budget_s=900, enum_cap=4096):
    """Phase A + phase B with a search over seam-side embedding variants.

    Where a patch is adjacent to more than one corridor tile, the seam
    can sit on either side; the choice changes which wall letters the
    merge pins, so some schedules embed only under a non-default choice
    (expressibility depends on the embedding geometry, not the circuit).

    The search first enumerates the WHOLE embedding space cheaply (no
    tqec): every mixed-radix contact variant up to ``enum_cap``,
    deduplicated by geometry.  Variants where some patch would have to
    open both an X and a Z wall on the same axis (see LAST_SCAN) cannot
    satisfy the wall rules, so conflict-free variants are checked first;
    the conflicted ones are kept as a fallback, not discarded.  Then a
    tqec-equipped python checks up to ``max_variants`` candidates in
    that order, stopping at the first that embeds.  Compact summary on
    stdout, full per-variant log in blockgraph/export_log.txt.  Returns
    the winning variant number, or None."""
    import contextlib
    import hashlib
    import io
    import math
    import os
    import subprocess
    import time
    bg = pathlib.Path(blockgraph_dir)
    bg.mkdir(exist_ok=True)
    tq = tqec_python or os.environ.get("TQEC_PYTHON")
    t0 = time.time()

    # phase 0: enumerate distinct geometries and their conflict flags
    seen, variants, space = set(), [], None
    v = 0
    while space is None or v < min(space, enum_cap):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            data = extract_from_cp(cp, name, distance, out_dir=bg,
                                   contact_variant=v)
        if v == 0:
            print(buf.getvalue().strip())
            space = math.prod(LAST_SCAN["radix"]) if LAST_SCAN["radix"] else 1
        h = hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            variants.append((v, bool(LAST_SCAN["conflicts"])))
        v += 1
        if time.time() - t0 > time_budget_s / 3:
            break                       # keep budget for the tqec checks
    clean = [w for w, c in variants if not c]
    order = clean + [w for w, c in variants if c]
    enum_note = (f"seam-side space: {space} embedding(s), {len(variants)} "
                 f"distinct geometries, {len(clean)} pass the wall-letter "
                 f"pre-filter")
    if v < space:
        enum_note += f" (enumeration stopped at variant {v})"
    print(enum_note)
    if tq is None:
        print(f"phase B needs tqec: run  <python-with-tqec> "
              f"tools/export_blockgraph_auto.py export --name {name}  "
              f"(from blockgraph/; committed output shown below)")
        return None

    logs, last, checked = [enum_note], "", 0
    for w in order:
        if checked >= max_variants or time.time() - t0 > time_budget_s:
            break
        with contextlib.redirect_stdout(io.StringIO()):
            extract_from_cp(cp, name, distance, out_dir=bg, contact_variant=w)
        r = subprocess.run([tq, str(pathlib.Path(__file__).resolve()),
                            "export", "--name", name],
                           cwd=bg, capture_output=True, text=True, timeout=900)
        checked += 1
        tag = " [pre-filter pass]" if w in clean else " [conflicted]"
        logs.append(f"--- contact variant {w}{tag}\n{r.stdout}{r.stderr}")
        last = (r.stdout.strip().splitlines() or ["(no output)"])[-1]
        if "exported" in r.stdout:
            (bg / "export_log.txt").write_text("\n".join(logs))
            note = "" if w == 0 else f"  (seam-side variant {w})"
            print(f"{last}{note}")
            print("(full search log: blockgraph/export_log.txt)")
            return w
        if "Y cube is not implemented" in r.stdout:
            break    # the Y half-cubes are the same in every embedding
    (bg / "export_log.txt").write_text("\n".join(logs))
    with contextlib.redirect_stdout(io.StringIO()):
        extract_from_cp(cp, name, distance, out_dir=bg)   # canonical geometry
    print(f"NOT EXPORTED -- {checked} of {len(variants)} distinct "
          f"embedding(s) checked (space {space}); last: {last}")
    print("(full search log: blockgraph/export_log.txt)")
    return None


def extract(name, distance, qasm_file, compile_json=None, contact_variant=0):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "experiments"))
    sys.path.insert(0, str(HERE))
    import contextlib, io
    from circls.pipeline import compile_qasm

    qasm = pathlib.Path(qasm_file).read_text()
    kwargs = dict(distance=distance, assignment="optimized",
                  liveness=True, keep_patches=set())
    if compile_json:
        kwargs.update(json.loads(compile_json))
    with contextlib.redirect_stdout(io.StringIO()):
        cp = compile_qasm(qasm, **kwargs)
    extract_from_cp(cp, name, distance, contact_variant=contact_variant)


def export(name):
    from tqec.computation.block_graph import BlockGraph
    from tqec.utils.position import Position3D

    data = json.loads((HERE / f"{name}_blockstructure.json").read_text())
    cubes = [tuple(v) for v in data["cubes"]]
    pipes = [(tuple(a), tuple(b)) for a, b in data["pipes"]]
    pins = {tuple(v): s for v, s in data["pins"]}
    y_bottoms = {tuple(v) for v in data["y_bottoms"]}

    # a patch column that joins nothing and whose init and readout bases
    # differ is a non-deterministic memory: its outcome enters no
    # deterministic parity, and tqec's correlation machinery rejects any
    # graph containing such a component -- omit it (the voxel view still
    # shows the full floor plan)
    piped = set()
    for a, b in pipes:
        if a[0] != b[0] or a[1] != b[1]:
            piped.add((a[0], a[1])); piped.add((b[0], b[1]))
    drop_tiles = set()
    by_tile = {}
    for v in cubes:
        by_tile.setdefault((v[0], v[1]), []).append(v)
    for t, vs in by_tile.items():
        if t in piped:
            continue
        lo, hi = min(vs), max(vs)
        if pins.get(lo) != pins.get(hi) and lo not in y_bottoms:
            drop_tiles.add(t)
    if drop_tiles:
        print(f"note: omitting {len(drop_tiles)} unjoined column(s) with "
              f"mixed init/readout bases (non-deterministic memories): "
              f"{sorted(drop_tiles)}")
        cubes = [v for v in cubes if (v[0], v[1]) not in drop_tiles]
        pipes = [(a, b) for a, b in pipes if (a[0], a[1]) not in drop_tiles]

    KINDS = ["ZXZ", "XZZ", "ZXX", "XZX", "XXZ", "ZZX"]
    adjacency = {}
    for a, b in pipes:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
    order = sorted(cubes)
    pipe_set = {frozenset(p) for p in pipes}
    # a cube whose both temporal faces are piped away AND that carries no
    # spatial pipe has a semantically irrelevant z letter -- canonicalize
    # it (k2 = k0) so the search space is only the real protocol freedom
    interior = {v for v in cubes
                if frozenset((v, (v[0], v[1], v[2] - 1))) in pipe_set
                and frozenset((v, (v[0], v[1], v[2] + 1))) in pipe_set
                and all(w[2] == v[2] - 1 or w[2] == v[2] + 1
                        for w in adjacency.get(v, []))}

    def pipe_axis(a, b):
        return [i for i in range(3) if a[i] != b[i]][0]

    def ok_pair(k1, k2, ax):
        if k1 == "Y" or k2 == "Y":
            return ax == 2          # Y half-cube: temporal pipes only
        o = [i for i in range(3) if i != ax]
        return all(k1[i] == k2[i] for i in o) and k1[o[0]] != k1[o[1]]

    sols, truncated = [], [False]

    def rec(i, assign):
        if len(sols) > 500:
            truncated[0] = True
            return
        if i == len(order):
            sols.append(dict(assign))
            return
        v = order[i]
        if v in y_bottoms:
            cands = ["Y"]
        elif v in pins:
            cands = [k for k in KINDS if k[2] == pins[v]]
        elif v in interior:
            cands = [k for k in KINDS if k[2] == k[0]]
        else:
            cands = KINDS
        for k in cands:
            good = True
            for w in adjacency.get(v, []):
                if w in assign and not ok_pair(assign[w], k, pipe_axis(v, w)):
                    good = False
                    break
            if good:
                assign[v] = k
                rec(i + 1, assign)
                del assign[v]

    rec(0, {})
    print(f"legal kind assignments (cap 500): {len(sols)}")
    if truncated[0]:
        print("note: search truncated at the cap -- the selection below is "
              "among the first 500 assignments only")
    if not sols:
        print("NOT EXPORTED: no ZXCube assignment satisfies the wall rules "
              "with the protocol's pinned faces -- expressibility gap.")
        return

    want = data["deterministic_observables"]
    chosen, counts, tqec_errors = None, {}, {}
    for assign in sols:
        g = BlockGraph(data["name"])
        for v in order:
            g.add_cube(Position3D(*v), assign[v])
        for a, b in pipes:
            g.add_pipe(Position3D(*a), Position3D(*b))
        try:
            g.validate()
        except Exception as e:
            tqec_errors[type(e).__name__] = tqec_errors.get(type(e).__name__, 0) + 1
            continue
        try:
            n = len(g.find_correlation_surfaces())
        except Exception as e:
            if "deterministic" in str(e):
                n = 0    # valid graph, but zero deterministic parities
            else:
                tqec_errors[type(e).__name__] = \
                    tqec_errors.get(type(e).__name__, 0) + 1
                continue
        counts[n] = counts.get(n, 0) + 1
        # the protocol-correct wall assignment is the one with the most
        # deterministic parities (wrong orientations destroy determinism,
        # they never create it)
        if n > 0 and (chosen is None or n > chosen[2]):
            chosen = (assign, g, n)
    print("correlation-surface counts among valid assignments:", counts)
    if tqec_errors:
        print(f"note: {sum(tqec_errors.values())} assignment(s) raised during "
              f"tqec evaluation (not semantic verdicts): {tqec_errors}")
    print(f"(CircLS circuit has {want} deterministic observables; TQEC's "
          f"surface count convention differs, so this is context, not a filter)")
    if chosen is None:
        print("NOT EXPORTED: every valid assignment has zero deterministic "
              "parities -- the searched geometry does not express this protocol.")
        return
    assign, g, n = chosen
    # a structurally valid graph can still be beyond tqec's circuit
    # compiler (e.g. it has no Y cube yet) -- gate the export on an
    # actual compile so "exported" always means tqec can consume it
    try:
        from tqec import compile_block_graph
        compile_block_graph(g, observables="auto")
    except Exception as e:
        print(f"NOT EXPORTED: tqec cannot compile this graph -- "
              f"{type(e).__name__}: {e}")
        return
    g.to_dae_file(str(HERE / f"{name}_blockgraph.dae"))
    g.view_as_html(write_html_filepath=str(HERE / f"{name}_blockgraph.html"))
    print(f"exported {name}_blockgraph.dae/.html  "
          f"({g.num_cubes} cubes, {g.num_pipes} pipes, {n} surfaces)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["extract", "export"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--distance", type=int, default=3)
    ap.add_argument("--qasm-file", default=None,
                    help="QASM file to compile (required for extract)")
    ap.add_argument("--compile-json", default=None,
                    help='extra compile_qasm kwargs as JSON, e.g. \'{"parallel_steps": false}\'')
    ap.add_argument("--contact-variant", type=int, default=0)
    a = ap.parse_args()
    if a.mode == "extract":
        if a.qasm_file is None:
            ap.error("extract requires --qasm-file")
        extract(a.name, a.distance, a.qasm_file, a.compile_json, a.contact_variant)
    else:
        export(a.name)
