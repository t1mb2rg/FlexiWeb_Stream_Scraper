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


def _page_is_alive(page: Any) -> bool:
    if page is None:
        return False
    checker = getattr(page, "is_closed", None)
    if not callable(checker):
        return True
    try:
        return not bool(checker())
    except Exception:
        return False


async def ensure_live_page(browser_mgr: Any) -> Any:
    """Return a usable page, recovering from a closed tab when the context is still alive."""

    current = getattr(browser_mgr, "page", None)
    if _page_is_alive(current):
        return current

    context = getattr(browser_mgr, "context", None)
    if context is None:
        raise RuntimeError("browser context is unavailable")

    try:
        pages = list(context.pages)
    except Exception as exc:
        raise RuntimeError(f"browser context is unavailable: {exc}") from exc

    for candidate in reversed(pages):
        if _page_is_alive(candidate):
            browser_mgr.page = candidate
            return candidate

    try:
        page = await context.new_page()
    except Exception as exc:
        raise RuntimeError(f"browser context is unavailable: {exc}") from exc

    browser_mgr.page = page
    return page


async def _run_session_turn(scraper: Any, prompt: str) -> None:
    """Run one turn with an atomic input fill when the scraper exposes browser primitives.

    The legacy scraper types text with ``press_sequentially``. On contenteditable chat
    composers, embedded newlines can be interpreted as Enter presses and split one API
    request into multiple messages. The session API instead fills the complete prompt in
    one operation and submits exactly once, while retaining the legacy implementation as
    a compatibility fallback for custom scraper implementations.
    """

    page = getattr(scraper, "page", None)
    selectors = getattr(scraper, "selectors", None)
    config = getattr(scraper, "config", None)
    logger = getattr(scraper, "logger", None)

    required_page_methods = (
        "wait_for_selector",
        "locator",
        "query_selector_all",
    )
    if (
        page is None
        or not isinstance(selectors, dict)
        or not isinstance(config, dict)
        or logger is None
        or not all(hasattr(page, method) for method in required_page_methods)
        or not hasattr(scraper, "track_stream")
    ):
        await scraper.execute(prompt)
        return

    logger.init_turn(prompt)

    current_url = str(getattr(page, "url", "") or "")
    base_url = str(config.get("base_url", "") or "")
    if base_url and base_url not in current_url:
        await page.goto(base_url)

    input_selector = selectors["input_box"]
    await page.wait_for_selector(input_selector)
    input_locator = page.locator(input_selector)
    await input_locator.click()
    await input_locator.focus()
    await input_locator.fill(prompt)
    await asyncio.sleep(0.1)

    ai_container_selector = selectors.get("ai_answer_container", "")
    pre_send_count = 0
    if ai_container_selector:
        try:
            pre_send_count = len(await page.query_selector_all(ai_container_selector))
        except Exception:
            pre_send_count = 0

    if selectors.get("enter_to_submit", True):
        await page.keyboard.press("Enter")
    else:
        await page.click(selectors["submit_button"])

    if ai_container_selector:
        start_time = asyncio.get_running_loop().time()
        while (asyncio.get_running_loop().time() - start_time) < 30.0:
            try:
                current_count = len(await page.query_selector_all(ai_container_selector))
            except Exception:
                current_count = 0
            if current_count > pre_send_count:
                break
            await asyncio.sleep(0.1)
        else:
            raise TimeoutError("timed out waiting for a new assistant response container")

    await scraper.track_stream(pre_send_count)


async def _extract_latest_visible_output(scraper: Any) -> str:
    """Fallback to visible text from the latest assistant wrapper.

    Site-specific markdown selectors remain the preferred extraction path. This fallback
    keeps the synchronous API usable when a site's internal markdown DOM changes while
    the outer assistant-turn selector still resolves correctly.
    """

    page = getattr(scraper, "page", None)
    selectors = getattr(scraper, "selectors", None)
    if page is None or not isinstance(selectors, dict):
        return ""

    container_selector = selectors.get("ai_answer_container")
    if not container_selector:
        return ""

    try:
        wrappers = await page.query_selector_all(container_selector)
    except Exception:
        return ""
    if not wrappers:
        return ""

    wrapper = wrappers[-1]
    exclude_selectors = selectors.get("exclude_selectors", [])

    if exclude_selectors and hasattr(wrapper, "evaluate"):
        try:
            cleaned = await wrapper.evaluate(
                """
                (node, selectors) => {
                    const clone = node.cloneNode(true);
                    for (const selector of selectors) {
                        try {
                            clone.querySelectorAll(selector).forEach((element) => element.remove());
                        } catch (_) {
                            // Ignore a stale site-specific selector and keep extracting.
                        }
                    }
                    return (clone.innerText || clone.textContent || '').trim();
                }
                """,
                exclude_selectors,
            )
            if cleaned:
                return str(cleaned).strip()
        except Exception:
            pass

    try:
        return str(await wrapper.inner_text()).strip()
    except Exception:
        return ""


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
                page = await ensure_live_page(browser_mgr)
            except RuntimeError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc

            target_url = request.conversation_url or store.get(request.session_id)
            if target_url:
                try:
                    await page.goto(target_url)
                except Exception as exc:
                    # The tab can disappear between the liveness check and navigation.
                    # Reacquire once before declaring the browser unavailable.
                    browser_mgr.page = None
                    try:
                        page = await ensure_live_page(browser_mgr)
                        await page.goto(target_url)
                    except Exception as retry_exc:
                        raise HTTPException(
                            status_code=502,
                            detail=f"failed to open requested conversation: {retry_exc}",
                        ) from retry_exc

            try:
                scraper = scraper_factory(request.site, page)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            try:
                await _run_session_turn(scraper, request.prompt)
            except Exception as exc:
                raise HTTPException(status_code=502, detail=f"scraper execution failed: {exc}") from exc

            output = str(getattr(scraper.logger, "full_output", "")).strip()
            if not output:
                output = await _extract_latest_visible_output(scraper)
                if output and hasattr(scraper.logger, "append_chunk"):
                    scraper.logger.append_chunk("final_output", output)

            current_url = str(getattr(page, "url", "") or "")
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
