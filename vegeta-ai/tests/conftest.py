import os

import pytest


HAVE_KEY = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "requires_api_key" in item.keywords and not HAVE_KEY:
            item.add_marker(pytest.mark.skip(reason="no ANTHROPIC_API_KEY"))
