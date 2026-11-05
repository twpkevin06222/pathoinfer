import pytest

from pathoinfer.registry.card import class_colors


def test_default_colours_cycle_the_colormap() -> None:
    names = [f"c{i}" for i in range(21)]
    colors = class_colors(names, cmap="tab20")
    assert colors["c0"] == (31, 119, 180)  # tab20's first colour
    assert len({colors[n] for n in names[:20]}) == 20
    assert colors["c20"] == colors["c0"]


def test_overrides_replace_default_colours() -> None:
    colors = class_colors(["background", "tumor"], {"tumor": (255, 0, 0)})
    assert colors["tumor"] == (255, 0, 0)
    assert colors["background"] == class_colors(["background"])["background"]


def test_override_for_unknown_class_is_rejected() -> None:
    with pytest.raises(ValueError, match="tumour"):
        class_colors(["background", "tumor"], {"tumour": (255, 0, 0)})
