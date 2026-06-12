"""Contract-conformance test — the guard rail (SPEC §2.1, CLAUDE.md).

Every registered head must satisfy the `Head` protocol in `heads/base.py`. This
is what stops a new head from quietly special-casing the spine: if a head drops a
contract method, this test fails before the spine ever has to branch on a domain.

The test parametrizes over the registry, so it is meaningful with zero heads
installed (it simply has nothing to check yet) and grows automatically as heads
land. Loading a head imports only that head's module (registry is lazy), so a
head whose deps aren't installed is reported as a skip, not a hard failure —
keeping the contract test runnable in a core-only environment.
"""

from __future__ import annotations

import inspect

import pytest

from heads.base import Head
from spine import registry

# Names declared by the Head protocol that every head must provide.
_REQUIRED_CALLABLES = (
    "prepare_data",
    "fit",
    "predict",
    "sweep_axes",
    "tile_label",
    "metric_name",
    "viewer",
)
_REQUIRED_ATTRS = ("name", "dependency_group")


def _load_or_skip(name: str) -> Head:
    try:
        return registry.load(name)
    except ImportError as exc:  # head deps not installed in this env
        pytest.skip(f"head {name!r} deps not installed: {exc}")


@pytest.mark.parametrize("name", registry.names())
def test_registered_head_satisfies_protocol(name: str) -> None:
    head = _load_or_skip(name)
    # runtime_checkable structural check
    assert isinstance(head, Head), f"{name!r} does not satisfy the Head protocol"


@pytest.mark.parametrize("name", registry.names())
def test_registered_head_has_all_members(name: str) -> None:
    head = _load_or_skip(name)
    for attr in _REQUIRED_ATTRS:
        assert hasattr(head, attr), f"{name!r} missing attribute {attr!r}"
    for fn in _REQUIRED_CALLABLES:
        assert callable(getattr(head, fn, None)), f"{name!r} missing method {fn!r}"


@pytest.mark.parametrize("name", registry.names())
def test_name_attr_matches_registry_key(name: str) -> None:
    head = _load_or_skip(name)
    assert head.name == name, f"head.name {head.name!r} != registry key {name!r}"


@pytest.mark.parametrize("name", registry.names())
def test_fit_signature_takes_run_and_hp(name: str) -> None:
    head = _load_or_skip(name)
    params = list(inspect.signature(head.fit).parameters)
    assert params[:2] == ["run", "hp"], f"{name!r}.fit must be fit(run, hp)"


def test_registry_lists_without_importing_heads() -> None:
    # names() must not raise even with no heads / no head deps installed.
    assert isinstance(registry.names(), list)
