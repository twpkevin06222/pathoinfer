import pytest

from pathoinfer._extras import extra, require


def test_missing_dependency_names_the_install_command() -> None:
    with pytest.raises(ImportError, match=r"pip install 'pathoinfer\[onnx\]'"):
        require("pathoinfer_no_such_module", extra("onnx"))


def test_present_dependency_is_returned() -> None:
    assert require("json", extra("onnx")).dumps({}) == "{}"
