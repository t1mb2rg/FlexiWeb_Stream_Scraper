"""Opt-in ASGI entrypoint with persistent conversation-session routes enabled.

Run with, for example:
    uvicorn session_app:app --host 127.0.0.1 --port 8000

The legacy ``python main.py`` entrypoint and ``/api/ask`` behavior remain unchanged.
"""

from main import GenericScraper, app, browser_mgr
from session_api import install_session_api

install_session_api(app, browser_mgr, GenericScraper)
