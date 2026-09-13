"""Compatibility rewrite: detach Hadamard pipes from spatial-junction cubes.

A spatial H pipe touching a spatial cube (XXZ/ZZX) is unimplemented in
both tqec conventions.  Fix: stretch the graph by one unit along the
pipe axis at the junction side, insert a regular cube in the gap, and
leave the H on the segment whose both ends are regular (the domain
wall slides freely along the tube, so semantics are unchanged).
Every other pipe crossing the cut plane is likewise extended through
an inserted cube of matching colors.

``stretch_and_fix`` is the whole interface; ``topologiq_bridge.build``
calls it in a loop on the cubes and pipes it parsed from the .bgraph.
"""
from tqec.utils.position import Position3D

AXIS = {0: "x", 1: "y", 2: "z"}


def is_spatial_cube(kind):
    return kind not in ("PORT",) and len(kind) == 3 and kind[0] == kind[1]


def pipe_axis(kind):
    return kind.upper().index("O")


def valid_kinds(x=None, y=None, z=None):
    out = []
    for a in "XZ":
        for b in "XZ":
            for c in "XZ":
                k = a + b + c
                if len(set(k)) == 1:
                    continue
                if (x and a != x) or (y and b != y) or (z and c != z):
                    continue
                out.append(k)
    return out


def stretch_and_fix(cubes, pipes):
    """One rewrite round: fix the first offending H pipe; return None if clean."""
    for i, (u, v, kind) in enumerate(pipes):
        if not kind.endswith("H") or pipe_axis(kind) == 2:
            continue                      # temporal H is fine
        su = is_spatial_cube(cubes[u][0])
        sv = is_spatial_cube(cubes[v][0])
        if not (su or sv):
            continue
        ax = pipe_axis(kind)
        # cut just below the higher endpoint along the pipe axis
        hi = u if u.as_tuple()[ax] > v.as_tuple()[ax] else v
        cut = hi.as_tuple()[ax]

        def shift(pos):
            t = list(pos.as_tuple())
            if t[ax] >= cut:
                t[ax] += 1
            return Position3D(*t)

        new_cubes = {shift(p): kv for p, kv in cubes.items()}
        new_pipes = []
        for (a, b, k) in pipes:
            a2, b2 = shift(a), shift(b)
            gap = abs(a2.as_tuple()[pipe_axis(k)] - b2.as_tuple()[pipe_axis(k)])
            if gap == 1:
                new_pipes.append((a2, b2, k))
                continue
            # pipe crosses the cut: insert a cube in the gap
            lo, hi2 = (a2, b2) if a2.as_tuple()[pipe_axis(k)] < b2.as_tuple()[pipe_axis(k)] else (b2, a2)
            t = list(lo.as_tuple()); t[pipe_axis(k)] += 1
            mid = Position3D(*t)
            plain = k[:3]
            flip = "".join({"X": "Z", "Z": "X", "O": "O"}[c] for c in plain)

            def mid_kind(letters):
                fixed = {AXIS[d]: letters[d] for d in range(3)
                         if d != pipe_axis(k)}
                cands = valid_kinds(**fixed)
                cands.sort(key=lambda kk: kk[0] == kk[1])   # prefer a non-junction kind
                return cands[0]

            if k.endswith("H"):
                # keep H on the segment away from any spatial-cube endpoint;
                # the mid cube and far segment carry the flipped colors
                lo_sp = is_spatial_cube(new_cubes[lo][0])
                if lo_sp:
                    # H on mid--hi2: mid keeps unflipped colors
                    new_cubes[mid] = (mid_kind(plain), "")
                    new_pipes.append((lo, mid, plain))
                    new_pipes.append((mid, hi2, plain + "H"))
                else:
                    # H on lo--mid: mid takes flipped colors
                    new_cubes[mid] = (mid_kind(flip), "")
                    new_pipes.append((lo, mid, plain + "H"))
                    new_pipes.append((mid, hi2, flip))
            else:
                new_cubes[mid] = (mid_kind(plain), "")
                new_pipes.append((lo, mid, plain))
                new_pipes.append((mid, hi2, plain))
        return new_cubes, new_pipes
    return None
