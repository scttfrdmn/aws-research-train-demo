"""Guard rail: head-derived tag VALUES must satisfy SageMaker's charset (SPEC §9).

CreateTrainingJob rejects tag values outside `([\\p{L}\\p{Z}\\p{N}_.:/=+\\-@]*)`
— this is what caught the `·` separator and the `rmse↓` suffix at submit time.
Every head's `tile_label` / `metric_name` feed tags, so they must stay inside it.
"""

from __future__ import annotations

import itertools
import re

import pytest

from heads.base import Head
from spine import registry

# SageMaker tag-value charset. \w covers letters/digits/underscore; the class
# adds space, dot, colon, slash, equals, plus, hyphen, at — and we allow the
# unicode letters/separators \p{L}\p{Z} via re.UNICODE's \w and an explicit
# space. Anything else (arrows, middle-dot) is rejected.
_TAG_VALUE = re.compile(r"^[\w\s.:/=+\-@]*$", re.UNICODE)


def _load_or_skip(name: str) -> Head:
    try:
        return registry.load(name)
    except ImportError as exc:
        pytest.skip(f"head {name!r} deps not installed: {exc}")


@pytest.mark.parametrize("name", registry.names())
def test_metric_name_is_tag_safe(name: str) -> None:
    head = _load_or_skip(name)
    assert _TAG_VALUE.match(head.metric_name()), head.metric_name()


@pytest.mark.parametrize("name", registry.names())
def test_tile_label_is_tag_safe_across_the_grid(name: str) -> None:
    head = _load_or_skip(name)
    axes = head.sweep_axes()
    # exercise every point of the suggested grid
    names = [a.name for a in axes]
    for combo in itertools.product(*(a.values for a in axes)):
        hp = dict(zip(names, combo, strict=True))
        label = head.tile_label(hp)
        assert _TAG_VALUE.match(label), f"{name}: {label!r} not tag-safe"
        assert len(label) <= 256, f"{name}: tile_label too long for a tag"
