from pathlib import Path

import pytest

from tests.tiling.conftest import write_pyramid


@pytest.fixture(scope="session")
def slides(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """``slide_name-01.tiff`` (0.25 mpp) and ``slide_name-02.tiff`` (no resolution)."""
    folder = tmp_path_factory.mktemp("slides")
    write_pyramid(folder / "slide_name-01.tiff", 0.25)
    write_pyramid(folder / "slide_name-02.tiff", None)
    return folder
