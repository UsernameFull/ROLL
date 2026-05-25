import importlib.util
import sys

import pytest

from roll.platforms import current_platform


def _require_module(module_name: str) -> None:
    try:
        module_spec = importlib.util.find_spec(module_name)
    except ValueError:
        module_spec = None

    available = module_spec is not None or module_name in sys.modules
    if not available and not current_platform.is_npu():
        pytest.skip(f"{module_name} is not installed in this environment.")
    assert available, f"{module_name} must be installed for NPU SGLang tests."


def test_sglang_import_available():
    _require_module("sglang")
    import sglang

    assert sglang.__version__
