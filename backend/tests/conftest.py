import sys
from pathlib import Path

import pytest

# Tests import backend modules directly and never need API keys.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pytest_addoption(parser):
    parser.addoption(
        "--run-live-evals", action="store_true", default=False,
        help="Also run evaluation cases against the real LLMs and SerpApi (uses quota).",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "live: needs API keys and consumes LLM/SerpApi quota")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-live-evals"):
        return
    skip = pytest.mark.skip(reason="live eval: pass --run-live-evals to run (uses API quota)")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)
