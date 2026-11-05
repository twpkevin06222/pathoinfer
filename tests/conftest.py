import shutil

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if shutil.which("nvidia-smi"):
        return
    skip = pytest.mark.skip(reason="no CUDA GPU (nvidia-smi not found)")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
