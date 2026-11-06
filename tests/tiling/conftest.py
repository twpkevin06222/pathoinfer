from pathlib import Path

import pytest

from pathoinfer.tiling.synthetic import HEIGHT, SLIDE_MPP, WIDTH, write_pyramid

__all__ = ["HEIGHT", "SLIDE_MPP", "WIDTH", "write_pyramid"]


@pytest.fixture(scope="session")
def slide_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_pyramid(tmp_path_factory.mktemp("slides") / "synthetic.tiff", SLIDE_MPP)


@pytest.fixture(scope="session")
def slide_without_mpp(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_pyramid(tmp_path_factory.mktemp("slides") / "no_mpp.tiff", None)
