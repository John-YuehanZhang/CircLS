"""Small matplotlib renderer for a block graph given as cubes + pipes (the .bgraph
text format topologiq writes, or a tqec BlockGraph).  Faces follow the ZXCube letter
convention: the two faces perpendicular to axis i are drawn in the colour of letter i
(X red, Z blue); ports are grey; Hadamard pipes carry a yellow band."""
import itertools

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

COL = {"X": "#d7a4a1", "Z": "#b9cdff", "O": "#c9c9c9"}


def _box(lo, hi, letters, alpha=1.0, edge="k"):
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    V = np.array(list(itertools.product((x0, x1), (y0, y1), (z0, z1))))
    faces = {  # axis -> the two faces perpendicular to it (vertex indices of V)
        0: ([0, 1, 3, 2], [4, 5, 7, 6]),
        1: ([0, 1, 5, 4], [2, 3, 7, 6]),
        2: ([0, 2, 6, 4], [1, 3, 7, 5]),
    }
    polys, cols = [], []
    for ax, (f1, f2) in faces.items():
        for f in (f1, f2):
            polys.append(V[f])
            cols.append(COL.get(letters[ax], "#c9c9c9"))
    pc = Poly3DCollection(polys, facecolors=cols, edgecolors=edge, linewidths=0.4, alpha=alpha)
    return pc


def parse_bgraph_text(text):
    cubes, pipes, sec = {}, [], None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("CUBES:"):
            sec = "c"; continue
        if s.startswith("PIPES:"):
            sec = "p"; continue
        if not s or sec is None:
            continue
        f = [x.strip() for x in s.split(";")]
        if sec == "c":
            idx, x, y, z, kind, label = f[:6]
            if "None" in (x, y, z):
                continue
            cubes[int(idx)] = ((int(x), int(y), int(z)), kind, label)
        else:
            u, v, kind = f[:3]
            if int(u) in cubes and int(v) in cubes:
                pipes.append((cubes[int(u)][0], cubes[int(v)][0], kind))
    return [(p, k, lb) for p, k, lb in cubes.values()], pipes


def from_tqec(g):
    cubes = [((c.position.x, c.position.y, c.position.z), "OOO" if c.is_port else str(c.kind), c.label)
             for c in g.cubes]
    pipes = [((p.u.position.x, p.u.position.y, p.u.position.z),
              (p.v.position.x, p.v.position.y, p.v.position.z), str(p.kind)) for p in g.pipes]
    return cubes, pipes


def plot(cubes, pipes, png_path, title="", size=0.6, elev=22, azim=-55, label_ports=True):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d")
    h = size / 2
    for (x, y, z), kind, label in cubes:
        letters = kind.upper()[:3] if kind != "OOO" else "OOO"
        ax.add_collection3d(_box((x - h, y - h, z - h), (x + h, y + h, z + h), letters,
                                 alpha=0.55 if kind == "OOO" else 1.0))
        if label_ports and label:
            ax.text(x, y, z + h + 0.15, label, fontsize=6, ha="center")
    r = 0.18
    for (ax0, ay0, az0), (bx, by, bz), kind in pipes:
        axis = kind.upper().index("O")
        lo = [min(ax0, bx), min(ay0, by), min(az0, bz)]
        hi = [max(ax0, bx), max(ay0, by), max(az0, bz)]
        lo2 = [lo[i] - r if i != axis else lo[i] + h for i in range(3)]
        hi2 = [hi[i] + r if i != axis else hi[i] - h for i in range(3)]
        ax.add_collection3d(_box(lo2, hi2, kind.upper()[:3], edge="0.3"))
        if kind.upper().endswith("H"):
            mid = [(lo[i] + hi[i]) / 2 for i in range(3)]
            b0 = [mid[i] - (r + 0.04 if i != axis else 0.08) for i in range(3)]
            b1 = [mid[i] + (r + 0.04 if i != axis else 0.08) for i in range(3)]
            pc = Poly3DCollection([np.array(list(itertools.product(*zip(b0, b1))))[f]
                                   for f in ([0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5])],
                                  facecolors="#ffe066", edgecolors="k", linewidths=0.3)
            ax.add_collection3d(pc)
    xs = [c[0][0] for c in cubes]; ys = [c[0][1] for c in cubes]; zs = [c[0][2] for c in cubes]
    span = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)) + 1.5
    cx, cy, cz = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2, (max(zs) + min(zs)) / 2
    ax.set_xlim(cx - span / 2, cx + span / 2); ax.set_ylim(cy - span / 2, cy + span / 2); ax.set_zlim(cz - span / 2, cz + span / 2)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z (time)")
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 1))
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(png_path, dpi=170)
    plt.close(fig)
    return png_path
