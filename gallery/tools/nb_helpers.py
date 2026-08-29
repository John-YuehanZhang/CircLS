"""Notebook display helpers: Pauli streams in the notation of Litinski,
"A Game of Surface Codes" (rotations as P_phi with the angle as a
subscript, e.g. (X (x) Z (x) 1 (x) 1)_{pi/4}; measurements as
"P measurement"), plus compact routing summaries."""
import json
import pathlib

_ANGLE = {"t": r"\pi/8", "s": r"\pi/4", "z": r"\pi/2"}


def _product(op, n):
    letters = [op.paulis.get(q, "I") for q in range(n)]
    return r" \otimes ".join(r"\mathbb{1}" if s == "I" else s for s in letters)


def op_latex(op, n):
    """One NWQEC op in Game-of-Surface-Codes notation (LaTeX)."""
    prod = _product(op, n)
    if op.kind == "m":
        sign = "-" if op.sign < 0 else ""
        return rf"${sign}{prod}$ measurement"
    angle = _ANGLE[op.kind]
    if op.sign < 0:
        angle = "-" + angle
    return rf"$({prod})_{{{angle}}}$"


def show_ops(circ, limit=None, tail=0):
    """Render an NWQEC op stream as GoSC-style math lines; with ``tail``
    the last ``tail`` lines (e.g. the terminal measurements) stay visible
    after the ellipsis."""
    from IPython.display import display, Markdown
    lines = [op_latex(op, circ.num_qubits) for op in circ.ops]
    if limit is None or len(lines) <= limit + tail:
        md = "  \n".join(lines)
    else:
        md = "  \n".join(lines[:limit])
        md += f"  \n... ({len(lines)} ops total, full list in data/)"
        if tail:
            md += "  \n" + "  \n".join(lines[-tail:])
    display(Markdown(md))


def fmt_ops_plain(circ):
    """ASCII version for the saved data files."""
    out = []
    for op in circ.ops:
        s = "".join(op.paulis.get(q, "I") for q in range(circ.num_qubits))
        angle = {"t": "pi/8", "s": "pi/4", "z": "pi/2"}.get(op.kind)
        head = f"({'-' if op.sign < 0 else '+'}{s})"
        out.append(f"{head}_{angle} rotation" if angle else f"{head} measurement")
    return out


def save_ops(circ, path):
    pathlib.Path(path).write_text("\n".join(fmt_ops_plain(circ)) + "\n")


def ppm_step_lines(steps):
    return [f"PPM {i}: " + "  ".join(f"{nm}.{P}" for nm, P in st.interaction_type)
            for i, st in enumerate(steps)]


def save_routing(cp, steps, path):
    """Persist placement + per-step corridor routing; return a one-line
    summary string."""
    exp = cp.experiment
    routing = []
    for i, st in enumerate(steps):
        r = exp._routes[i]
        tree = [list(c) for c in r.tree] if (r is not None and r.tree) else []
        routing.append({"step": i,
                        "targets": [[nm, P] for nm, P in st.interaction_type],
                        "corridor_tiles": tree})
    info = {"placement": {nm: list(c) for nm, c in cp.placement.items()},
            "routing": routing}
    pathlib.Path(path).write_text(json.dumps(info, indent=1))
    joint = sum(1 for r in routing if len(r["targets"]) > 1)
    return (f"{len(info['placement'])} patches placed, {len(routing)} PPM steps "
            f"({joint} joint) routed; full detail in {pathlib.Path(path).name}")


def save_ppm_sequence(raw, steps, path):
    """Raw terminal measurements + executed PPM sequence, one text file."""
    pathlib.Path(path).write_text(
        "# raw terminal measurements (NWQEC keep_cx=False; signs via stim):\n"
        + "\n".join(fmt_ops_plain(raw))
        + "\n\n# executed PPM sequence after re-selection / reordering:\n"
        + "\n".join(ppm_step_lines(steps)) + "\n")


def save_ler_points(points, path):
    """points: {(d, p): (ler, errors, shots)} -> json."""
    pathlib.Path(path).write_text(json.dumps(
        {f"d{d}_p{p:g}": {"ler": v[0], "errors": v[1], "shots": v[2]}
         for (d, p), v in points.items()}, indent=1))


def plot_ler(points, ps, title, png_path):
    """Log-log LER figure, one curve per distance, zero-error points skipped."""
    import matplotlib
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    marks = {3: "o-", 5: "s-", 7: "^-"}
    for d in sorted({d for d, _ in points}):
        xs = [p for p in ps if points[(d, p)][1] > 0]
        ys = [points[(d, p)][0] for p in xs]
        es = [points[(d, p)][0] / max(points[(d, p)][1], 1) ** 0.5 for p in xs]
        ax.errorbar(xs, ys, yerr=es, fmt=marks.get(d, "d-"), capsize=3,
                    label=f"d = {d}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("physical error rate p"); ax.set_ylabel("logical error rate")
    ax.set_title(title)
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
