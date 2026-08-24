"""Result provenance: every official result
file starts with one provenance record answering "which code, which config,
which environment produced these numbers".

Dirty-check is the point: a dirty working tree REFUSES to stamp official
results (the sh_chain->twisted_ghz episode ran five sweeps against an
uncommitted redesign, 2026-08-05).  ``allow_dirty=True`` is for interactive
exploration only and is recorded loudly in the output.

No hostname / username / absolute paths in the record (they leak
identity) — CPU model and library versions only.
"""
from __future__ import annotations

import datetime
import platform
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(_ROOT), *args],
                          capture_output=True, text=True,
                          check=True).stdout.strip()


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _scrub_argv(tok: str) -> str:
    """Reduce any absolute or home-anchored path in an argv token to
    <path>/basename, including --flag=/abs/value forms."""
    if tok.startswith("-") and "=" in tok:
        k, v = tok.split("=", 1)
        return f"{k}={_scrub_argv(v)}"
    if tok.startswith(("/", "~")):
        return "<path>/" + Path(tok.rstrip("/")).name
    return tok


def provenance(allow_dirty: bool = False) -> dict:
    sha = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    if dirty and not allow_dirty:
        raise RuntimeError(
            "working tree is dirty — commit first, or pass --allow-dirty for "
            "throwaway runs (official results must map to one commit)")
    import numpy
    import stim
    versions = {"python": platform.python_version(),
                "stim": stim.__version__,
                "numpy": numpy.__version__}
    try:
        import pymatching
        versions["pymatching"] = pymatching.__version__
    except ImportError:
        pass
    try:
        import networkx
        versions["networkx"] = networkx.__version__
    except ImportError:
        pass
    return {"record": "provenance",
            "circls_sha": sha,
            "dirty": dirty,
            "versions": versions,
            "cpu": _cpu_model(),
            # path-like argv values are reduced to their basename so a
            # committed provenance row never records machine-local layout
            # (covers bare values, --flag=/abs forms, and ~ expansions)
            "argv": [Path(sys.argv[0]).name,
                     *(_scrub_argv(a) for a in sys.argv[1:])],
            "timestamp_utc": datetime.datetime.now(
                datetime.timezone.utc).isoformat(timespec="seconds")}
