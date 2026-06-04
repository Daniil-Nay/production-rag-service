"""Test bootstrap.

The smoke tests exercise pure logic (chunking, RRF, citation checks, the injection
filter) and never call an LLM or touch the database. But importing the package still
constructs ``Settings()``, which requires the LLM credentials. On a fresh clone there is
no ``.env``, so without this shim ``pytest`` would fail at collection with a wall of
pydantic "field required" errors before a single test runs.

We set throwaway values so configuration loads. Nothing here reaches a real provider.
"""

import os

_DUMMY_ENV = {
    "LLM_BASE_URL": "http://localhost:0/v1",
    "LLM_API_KEY": "test-key-not-used",
    "LLM_MODEL_PRIMARY": "test-model",
    "LLM_MODEL_CHEAP": "test-model",
}

for _key, _value in _DUMMY_ENV.items():
    os.environ.setdefault(_key, _value)
