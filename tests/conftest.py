import importlib.util
import os
import sys
from pathlib import Path

# Hermes host checkout (needed only by the integration tests and the
# ContextEngine subclass check). Override with HERMES_REPO=/path/to/hermes-agent.
HERMES_REPO = Path(os.environ.get(
    "HERMES_REPO", str(Path.home() / ".hermes" / "hermes-agent")))
REPO_ROOT = HERMES_REPO  # backwards-compatible name
if HERMES_REPO.is_dir() and str(HERMES_REPO) not in sys.path:
    sys.path.insert(0, str(HERMES_REPO))

# Always test the plugin source in THIS checkout, never an installed copy.
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


import pytest


def hermes_available() -> bool:
    return importlib.util.find_spec("agent") is not None


requires_hermes = pytest.mark.skipif(
    not hermes_available(), reason="Hermes host not importable; set HERMES_REPO")


@pytest.fixture
def engine_mod():
    return load_engine_module()
