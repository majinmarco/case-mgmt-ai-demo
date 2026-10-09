import os

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "real_model: calls a real model API; runs only with RUN_REAL_MODEL_TESTS=1"
    )


def pytest_collection_modifyitems(config, items):
    if os.environ.get("RUN_REAL_MODEL_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="real-model test; set RUN_REAL_MODEL_TESTS=1 to run")
    for item in items:
        if "real_model" in item.keywords:
            item.add_marker(skip)
