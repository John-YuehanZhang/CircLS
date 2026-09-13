"""LER notebook helpers (copied from the CircLS gallery's nb_helpers.py)."""
import json
import pathlib


def save_ler_points(points, path, shas=None):
    """points: {(d, p): (ler, errors, shots)} -> json; ``shas`` maps a distance to the sha256 prefix of
    the circuit the points were sampled from, so a stale cache can be told apart from a valid one."""
    shas = shas or {}
    pathlib.Path(path).write_text(json.dumps(
        {f"d{d}_p{p:g}": {"ler": v[0], "errors": v[1], "shots": v[2],
                          "circuit_sha256_prefix": shas.get(d)}
         for (d, p), v in points.items()}, indent=1))


def ler_figure(points, ps, title):
    """Log-log LER figure (one curve per distance, zero-error points skipped); returns the figure."""
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
    return fig


def plot_ler(points, ps, title, png_path):
    """ler_figure saved to a PNG."""
    import matplotlib.pyplot as plt
    fig = ler_figure(points, ps, title)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
