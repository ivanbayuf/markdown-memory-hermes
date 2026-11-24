"""Make the tests path-portable: resolve the Hermes repo from ``HERMES_REPO``
(environment variable) with a sane default; the plugin under test is the repo
copy itself (this tests/ directory sits inside it)."""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(os.environ.get("HERMES_REPO", Path.home() / ".hermes" / "hermes-agent"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_PLUGIN_DIR = Path(__file__).resolve().parent.parent


def load_engine_module():
    """Load the plugin package via spec (dir name carries a hyphen)."""
    name = "markdown_memory_test"
    spec = importlib.util.spec_from_file_location(
        name, _PLUGIN_DIR / "__init__.py", submodule_search_locations=[str(_PLUGIN_DIR)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def engine_mod():
    return load_engine_module()
