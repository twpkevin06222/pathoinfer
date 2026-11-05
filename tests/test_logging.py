"""Log levels run in fresh interpreters: levels are process-wide, set on import."""

import os
import subprocess
import sys
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"

# Log from inside each package's namespace (loguru names records by the calling module).
EMIT = """
def emit(tag):
    for package in ("pathoinfer", "patchology"):
        for lv in ("DEBUG", "INFO"):
            exec("logger.log(lv, msg)", {"logger": logger, "__name__": package + ".m",
                 "lv": lv, "msg": f"{tag}:{package}:{lv}"})
"""


def run(code: str, env: dict[str, str] | None = None, args: list[str] | None = None) -> str:
    clean = {k: v for k, v in os.environ.items() if not k.endswith("_LOG_LEVEL")}
    clean.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [sys.executable, "-c", code, *(args or [])],
        env=clean | (env or {}),
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stderr


def shown(err: str, tag: str, package: str) -> list[str]:
    return [lv for lv in ("DEBUG", "INFO") if f"{tag}:{package}:{lv}" in err]


def test_default_is_info_and_set_log_level_changes_it() -> None:
    err = run(
        "import sys\nfrom loguru import logger\nimport pathoinfer\nfrom pathoinfer import _logging\n"
        + EMIT
        + "print('level', _logging.LOG_LEVEL, file=sys.stderr)\nemit('before')\n"
        "pathoinfer.set_log_level('DEBUG')\n"
        "print('level', _logging.LOG_LEVEL, file=sys.stderr)\nemit('after')\n"
    )
    assert "level INFO" in err and "level DEBUG" in err
    assert shown(err, "before", "pathoinfer") == ["INFO"]
    assert shown(err, "after", "pathoinfer") == ["DEBUG", "INFO"]
    # Only pathoinfer's level changed.
    assert shown(err, "after", "patchology") == ["INFO"]


def test_environment_sets_each_package_level() -> None:
    err = run(
        "import sys\nfrom loguru import logger\nimport pathoinfer\n" + EMIT + "emit('env')\n",
        env={"PATHOINFER_LOG_LEVEL": "DEBUG", "PATCHOLOGY_LOG_LEVEL": "WARNING"},
    )
    assert shown(err, "env", "pathoinfer") == ["DEBUG", "INFO"]
    assert shown(err, "env", "patchology") == []
    assert err.count("env:pathoinfer:INFO") == 1, "printed twice"


def test_log_call_logs_arguments_at_debug_only() -> None:
    code = (
        "import numpy as np\nimport pathoinfer\nfrom pathoinfer._logging import log_call\n"
        "src = '@log_call\\ndef stage(tiles, name=\"x\"): return 1'\n"
        "scope = {'log_call': log_call, '__name__': 'pathoinfer.stages'}\n"
        "exec(src, scope)\n"
        "scope['stage'](np.zeros((2, 3), np.uint8), name='slide')\n"
    )
    assert "stage(" not in run(code)
    err = run(code, env={"PATHOINFER_LOG_LEVEL": "DEBUG"})
    assert "pathoinfer.stages:stage" in err
    assert "stage(ndarray(shape=(2, 3), dtype=uint8), name='slide')" in err


def test_file_sink_captures_pathoinfer_and_patchology(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    run(
        "import sys\nfrom loguru import logger\nimport pathoinfer\n"
        + EMIT
        + f"sink = pathoinfer.add_file_sink({str(log)!r})\nemit('file')\n"
        "logger.info('file:other')\nlogger.remove(sink)\n"
    )
    text = log.read_text()
    assert "file:pathoinfer:DEBUG" in text and "file:patchology:INFO" in text
    assert "file:other" not in text


def test_cli_verbose_sets_debug_for_pathoinfer_and_patchology() -> None:
    cli = "import sys\nfrom pathoinfer.cli import main\nsys.exit(main(sys.argv[1:]))"
    folder = str(FIXTURES / "tiny_seg")
    quiet = run(cli, args=["card", "validate", folder])
    assert "ok:" in quiet and "validate_folder(" not in quiet
    for flag in (["-v"], ["--log-level", "debug"]):
        err = run(cli, args=[*flag, "card", "validate", folder])
        assert "| DEBUG" in err and "validate_folder(" in err, flag
    assert "ok:" not in run(cli, args=["--log-level", "WARNING", "card", "validate", folder])
