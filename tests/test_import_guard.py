import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
FILES = sorted([*(ROOT / "src").rglob("*.py"), *(ROOT / "tests").rglob("*.py")])
BANNED = {"pandas", "geopandas", "pathotrain", "vllm"}
# The optional interop helper is the only place geopandas may appear.
ALLOWED = {ROOT / "src/pathoinfer/io/geodataframe.py": {"geopandas"}}
# Heavy optional dependencies that only their own extra's code may load, and lazily.
LAZY = [
    "onnxruntime",
    "torch",
    "tensorrt",
    "tritonclient",
    "httpx",
    "fastapi",
    "modelopt",
    "llmcompressor",
    "mlflow",
    "huggingface_hub",
    "geopandas",
]


def imported_modules(path: Path) -> set[str]:
    """Top-level names of every import in ``path``, including lazy ``require("x", ...)`` calls."""
    names = set()
    for node in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
        elif (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", None) in ("require", "import_module")
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            names.add(str(node.args[0].value).split(".")[0])
    return names


# torch is core but heavy: only these modules import it at the top, and nothing else
# imports them, so `import pathoinfer` and the CLI stay light.
TORCH_MODULES = ["pathoinfer.tiling.dataset"]


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_banned_imports(path: Path) -> None:
    banned = imported_modules(path) & (BANNED - ALLOWED.get(path, set()))
    assert not banned, f"{path.relative_to(ROOT)} imports {sorted(banned)}"


def test_importing_every_module_loads_no_optional_dependency() -> None:
    script = f"""
import importlib, json, pkgutil, sys
import pathoinfer
for m in pkgutil.walk_packages(pathoinfer.__path__, "pathoinfer."):
    if m.name not in {TORCH_MODULES!r}:
        importlib.import_module(m.name)
loaded = {LAZY!r} + {TORCH_MODULES!r}
print(json.dumps(sorted(n for n in loaded if n in sys.modules)))
"""
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert json.loads(out.stdout) == []
