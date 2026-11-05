import shutil
from collections.abc import Iterator

import pytest
from loguru import logger

pytest_plugins = ["tests.outputs"]


@pytest.fixture
def caplog(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """Route loguru records into pytest's caplog, as patchology's tests do."""
    handler_id = logger.add(caplog.handler, format="{message}", level=0)
    yield caplog
    logger.remove(handler_id)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if shutil.which("nvidia-smi"):
        return
    skip = pytest.mark.skip(reason="no CUDA GPU (nvidia-smi not found)")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
