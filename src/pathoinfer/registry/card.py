"""What pathoinfer knows about a model's classes beyond the exported metadata."""

from collections.abc import Mapping, Sequence

import matplotlib

RGB = tuple[int, int, int]


def class_colors(
    names: Sequence[str], overrides: Mapping[str, RGB] | None = None, cmap: str = "tab20"
) -> dict[str, RGB]:
    """RGB colour per class name: ``cmap``'s colours in class order, wrapping after
    its last, with ``overrides`` replacing any of them.

    Raises:
        ValueError: ``overrides`` names a class the model does not have.
    """
    overrides = overrides or {}
    unknown = set(overrides) - set(names)
    if unknown:
        raise ValueError(f"colours for unknown classes {sorted(unknown)}; classes: {list(names)}")
    # shortcut: cycling suits qualitative maps (tab10, tab20); a continuous map gives
    # near-identical neighbours, so sample it evenly if users ask for one.
    colormap = matplotlib.colormaps[cmap]
    colors = {}
    for i, name in enumerate(names):
        r, g, b, _ = colormap(i % colormap.N)
        colors[name] = (round(r * 255), round(g * 255), round(b * 255))
    return colors | dict(overrides)
