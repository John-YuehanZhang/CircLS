"""Collect what a section wrote and print it at the end of the section's cell, as paths
relative to the entry folder."""
import pathlib

_ENTRIES, _INLINE = [], []


def record_saved(entries, figures_inline=()):
    """entries: [(path, note), ...]; figures_inline: names of figures embedded in the notebook only."""
    _ENTRIES.extend(entries)
    _INLINE.extend(figures_inline)


def print_saved(here):
    """Print (and clear) everything recorded since the last call."""
    here = pathlib.Path(here).resolve()
    rels = [(str(pathlib.Path(p).resolve().relative_to(here)), note) for p, note in _ENTRIES]
    width = max((len(r) for r, _ in rels), default=0)
    print("saved:")
    for rel, note in rels:
        print(f"  {rel:{width}s}  {note}")
    for name in _INLINE:
        print(f"  ({name}: shown inline above, embedded in this notebook)")
    _ENTRIES.clear()
    _INLINE.clear()


def clear_saved():
    """Drop anything recorded but not yet printed (e.g. after a failed cell)."""
    _ENTRIES.clear()
    _INLINE.clear()
