"""Lazy imports for optional dependencies, so ``import pathoinfer`` stays light."""

import importlib
from types import ModuleType


def require(module: str, install: str) -> ModuleType:
    """Import ``module`` or raise an ImportError telling the user to run ``install``,
    e.g. ``require("onnxruntime", "pip install 'pathoinfer[onnx]'")``."""
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise ImportError(f"{module} is not installed: {install}") from e


def extra(name: str) -> str:
    """The install command for pathoinfer's ``name`` extra."""
    return f"pip install 'pathoinfer[{name}]'"
