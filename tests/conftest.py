import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path("/home/ubuntu/.hermes/hermes-agent")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_PLUGIN_DIR = Path("/home/ubuntu/.hermes/plugins/markdown-memory")


def load_engine_module():
    """Load the plugin package via spec (dir name carries a hyphen)."""
    name = "markdown_memory_test"
    spec = importlib.util.spec_from_file_location(
        name, _PLUGIN_DIR / "__init__.py", submodule_search_locations=[str(_PLUGIN_DIR)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


import pytest


@pytest.fixture
def engine_mod():
    return load_engine_module()
