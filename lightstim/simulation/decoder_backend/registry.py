"""Decoder registry: name -> backend -> Decoder class."""
from __future__ import annotations

from typing import Any, Dict, Type

# Registry: decoder_name -> { backend -> decoder_class }
_decoder_registry: Dict[str, Dict[str, type]] = {}

# Names registered as aliases (excluded from list_decoders output)
_ALIASES: set = set()


def register_decoder(
    name: str,
    decoder_class: Type,
    aliases: list[str] | None = None,
    backend: str = "cpu",
) -> None:
    """Register a decoder class under a name, backend, and optional aliases."""
    name = name.lower()
    _decoder_registry.setdefault(name, {})[backend] = decoder_class
    if aliases:
        for alias in aliases:
            a = alias.lower()
            _decoder_registry.setdefault(a, {})[backend] = decoder_class
            _ALIASES.add(a)


def get_decoder(
    name: str,
    backend: str = "cpu",
    **params,
) -> Any:
    """
    Get a decoder instance by name and backend.

    Args:
        name: Decoder name (e.g. 'pymatching', 'bposd').
        backend: 'cpu' or 'gpu'.
        **params: Decoder-specific parameters.

    Returns:
        Decoder instance implementing sinter.Decoder interface.
    """
    from . import decoders  # noqa: F401 — registration happens on import

    name = name.lower()

    if name in _decoder_registry:
        by_backend = _decoder_registry[name]
        cls = by_backend.get(backend)
        if cls is None and backend != "cpu":
            hint = (
                " Install cudaq_qec for GPU support: pip install cudaq_qec"
                if backend == "gpu"
                else ""
            )
            raise ImportError(
                f"Decoder '{name}' has no '{backend}' backend registered.{hint}"
            )
        cls = cls or by_backend.get("cpu")
        if cls is not None:
            return cls(**params) if params else cls()
        # name IS registered, but not for this backend (e.g. bposd
        # registered gpu-only via cudaqx while stimbposd is absent)
        hint = (" Install stimbposd for the CPU backend: "
                "pip install -e '.[decoders]'" if backend == "cpu" else "")
        raise ImportError(
            f"Decoder '{name}' has no '{backend}' backend registered."
            f"{hint}")

    available = list_decoders()
    raise ValueError(f"Unknown decoder '{name}'. Available: {available}")


def has_decoder(name: str, backend: str = "cpu") -> bool:
    """True iff ``name`` is registered WITH the given backend.

    ``name in list_decoders()`` is not enough: a decoder can be
    registered for one backend only (bposd arrives cpu-only via
    stimbposd, gpu-only via cudaqx)."""
    from . import decoders  # noqa: F401 — registration happens on import
    by_backend = _decoder_registry.get(name.lower())
    return bool(by_backend) and (backend in by_backend)


def list_decoders() -> list[str]:
    """Return sorted list of registered decoder primary names (aliases excluded)."""
    from . import decoders  # noqa: F401 — ensure decoders are loaded
    return sorted(n for n in _decoder_registry if n not in _ALIASES)


# Keep _unique_decoder_names as a private alias for back-compat
_unique_decoder_names = list_decoders
