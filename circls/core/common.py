"""Shared PPM-driver constants (imported by the driver and by the
compiler-side mixins; lives outside both to keep module loading
acyclic)."""
_CONJ = {'X': 'Z', 'Z': 'X'}
_FLIP_O = {'X_horizontal': 'X_vertical', 'X_vertical': 'X_horizontal'}
#: auto_rotate candidate search cap: steps with more targets than this only
#: try the no-rotation candidates (the per-target product is exponential)
_ROTATION_SEARCH_CAP = 6


