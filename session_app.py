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
async def health() -> dict[str, str]:
    return {"status": "ok"}
