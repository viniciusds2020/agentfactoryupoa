"""Ensure enterprise routes are registered during tests."""
import os
import sys
from unittest.mock import MagicMock

# Must be set before app.py is imported so the enterprise_router is included.
os.environ.setdefault("SIMPLE_MODE", "false")

# Mock faiss if not installed (prevents ImportError in vectordb.py)
if "faiss" not in sys.modules:
    try:
        import faiss  # noqa: F401
    except ImportError:
        sys.modules["faiss"] = MagicMock()
