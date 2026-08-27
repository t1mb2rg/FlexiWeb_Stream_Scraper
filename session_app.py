"""Opt-in ASGI entrypoint with persistent conversation-session routes enabled.

Run with, for example:
    uvicorn session_app:app --host 127.0.0.1 --port 8000

This entrypoint intentionally does not start the legacy interactive CLI loop,
so API callers exclusively own the shared browser page while the service runs.
The legacy ``python main.py`` entrypoint and ``/api/ask`` behavior remain unchanged.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from main import GenericScraper, browser_mgr
from session_api import install_session_api


@asynccontextmanager
async def lifespan(app: FastAPI):
    await browser_mgr.start()
    try:
        yield
    finally:
        await browser_mgr.stop()


app = FastAPI(title="FlexiWeb Session API", lifespan=lifespan)
install_session_api(app, browser_mgr, GenericScraper)


@app.get("/api/health")
async def health() -> dict[str, object]:
    page = getattr(browser_mgr, "page", None)
    page_alive = False
    if page is not None:
        checker = getattr(page, "is_closed", None)
        try:
            page_alive = not checker() if callable(checker) else True
        except Exception:
            page_alive = False

    context = getattr(browser_mgr, "context", None)
    context_alive = context is not None
    page_count = 0
    if context is not None:
        try:
            page_count = len(context.pages)
        except Exception:
            context_alive = False

    return {
        "status": "ok" if page_alive and context_alive else "degraded",
        "browser_context_alive": context_alive,
        "page_alive": page_alive,
        "page_count": page_count,
    }
