"""Head registry — discover heads by name, load only the one requested (SPEC §2.2).

The registry maps a `--domain` name to the module path of its head, and imports
that module lazily. This is what keeps `uv sync --group molecular` from dragging
in `neuralop`: importing the spine, or loading the molecular head, never touches
another head's module (and so never imports its deps).

A head registers by name → "module:attribute" string. `load(name)` imports just
that module and returns the head instance. `names()` lists known domains without
importing any of them.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from heads.base import Head

# name → "module_path:attribute". The value is a string so listing names imports
# nothing. Heads are added here as they land (molecular first; weather/genomics/
# llm in v0.4+). The spine never imports these modules eagerly.
_REGISTRY: dict[str, str] = {
    "molecular": "heads.molecular.head:HEAD",
    "weather": "heads.weather.head:HEAD",
}


def names() -> list[str]:
    """Known domain names, without importing any head module."""
    return sorted(_REGISTRY)


def load(name: str) -> Head:
    """Import only the requested head's module and return its instance.

    Raises KeyError with the available names if `name` is unknown.
    """
    try:
        target = _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown domain {name!r}; available: {names() or '(none registered)'}"
        ) from None
    module_path, _, attr = target.partition(":")
    module = importlib.import_module(module_path)
    head: Head = getattr(module, attr)
    return head
