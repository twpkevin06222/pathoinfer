from pathlib import Path

import pytest

from tests.tiling.conftest import write_pyramid


@pytest.fixture(scope="session")
def slide(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A synthetic pyramidal slide at 0.25 mpp for calibration and parity windows."""
    return write_pyramid(tmp_path_factory.mktemp("slides") / "slide_name-01.tiff", 0.25)
