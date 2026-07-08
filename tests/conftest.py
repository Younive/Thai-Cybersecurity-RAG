"""Shared fixtures. Neutralizes the two import-time obstacles so src modules
import without a real chroma_db/ or live OpenRouter key:

- retrieval.py runs `VectorStoreManager().get_exist_cromadb()` at import.
- api.py + manager build OpenAI clients needing a key present (not valid).

Runs at collection time (before any test module imports src), so the patch and
env vars are in place when `import retrieval` / `import api` first fire.
`pythonpath = ["src"]` (pyproject) makes the bare imports resolve.
"""
import os
from unittest.mock import MagicMock

# Dummy creds: client construction validates presence, never calls the network
# here. setdefault + load_dotenv(override=False) means a real .env.local can't
# leak real keys into the test process either.
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
os.environ.setdefault("OPENROUTER_EMBEDDING_MODEL", "test-embed")
os.environ.setdefault("OPENROUTER_GENERAL_MODEL", "test-chat")

# Replace the store loader before retrieval.py imports it. Returns a fresh
# MagicMock so no test touches a real Chroma dir.
import vectorstore.manage_vectorstore as _mv  # noqa: E402
_mv.VectorStoreManager.get_exist_cromadb = lambda self: MagicMock(name="fake_vectorstore")

import pytest  # noqa: E402
from langchain_core.documents import Document  # noqa: E402


@pytest.fixture
def make_doc():
    """Factory: make_doc("text", source="dataset/x.pdf", page=3) -> Document."""
    def _make(content, **metadata):
        return Document(page_content=content, metadata=metadata)
    return _make
