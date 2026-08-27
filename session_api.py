from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class SessionAskRequest(BaseModel):
    site: str = "chatgpt"
    prompt: str = Field(min_length=1)
    request_id: str | None = None
    session_id: str | None = None
    conversation_url: str | None = None


class SessionAskResponse(BaseModel):
    status: str
    site: str
    output: str
    request_id: str | None = None
    session_id: str | None = None
    conversation_url: str | None = None


class SessionStore:
    """Small persistent mapping from caller-owned session names to web conversation URLs."""

    def __init__(self, path: Path):
        self.path = path

    def load(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(raw, dict):
            return {}
        return {
            str(key): str(value)
            for key, value in raw.items()
            if isinstance(key, str) and isinstance(value, str) and value.strip()
        }

    def get(self, session_id: str | None) -> str | None:
        if not session_id:
            return None
        return self.load().get(session_id)

    def set(self, session_id: str | None, conversation_url: str | None) -> None:
        if not session_id or not conversation_url:
            return
        data = self.load()
        data[session_id] = conversation_url
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)


def install_session_api(
    app: FastAPI,
    browser_mgr: Any,
    scraper_factory: Callable[[str, Any], Any],
    *,
    session_store_path: Path | None = None,
) -> None:
    """Install a synchronous, session-aware API without changing the legacy /api/ask route.

    Requests are serialized because the current FlexiWeb runtime owns one shared browser page.
    A named session remembers the page URL after each successful turn, allowing callers to
    keep using the same ChatGPT/Gemini/etc. conversation across process requests.
    """

    store = SessionStore(
        session_store_path
        or Path(getattr(browser_mgr, "user_data_dir", "browser_user_data")) / "flexiweb_sessions.json"
    )
    browser_lock = asyncio.Lock()

    @app.post("/api/ask/sync", response_model=SessionAskResponse)
    async def ask_ai_sync(request: SessionAskRequest) -> SessionAskResponse:
        async with browser_lock:
            try:
                scraper = scraper_factory(request.site, browser_mgr.page)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            target_url = request.conversation_url or store.get(request.session_id)
            if target_url:
                try:
                    await browser_mgr.page.goto(target_url)
                except Exception as exc:
                    raise HTTPException(
                        status_code=502,
                        detail=f"failed to open requested conversation: {exc}",
                    ) from exc

            try:
                await scraper.execute(request.prompt)
            except Exception as exc:
                raise HTTPException(status_code=502, detail=f"scraper execution failed: {exc}") from exc

            output = str(getattr(scraper.logger, "full_output", "")).strip()
            current_url = str(getattr(browser_mgr.page, "url", "") or "")
            if request.session_id and current_url:
                store.set(request.session_id, current_url)

            if not output:
                raise HTTPException(status_code=502, detail="scraper completed without a final output")

            return SessionAskResponse(
                status="completed",
                site=request.site,
                output=output,
                request_id=request.request_id,
                session_id=request.session_id,
                conversation_url=current_url or target_url,
            )

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str) -> dict[str, str | None]:
        return {"session_id": session_id, "conversation_url": store.get(session_id)}
