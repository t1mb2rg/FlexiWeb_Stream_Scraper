from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from session_api import SessionStore, install_session_api


def test_session_store_round_trip(tmp_path: Path):
    store = SessionStore(tmp_path / "sessions.json")
    assert store.get("forge") is None

    store.set("forge", "https://chatgpt.com/c/example")

    assert store.get("forge") == "https://chatgpt.com/c/example"


def test_session_store_ignores_empty_values(tmp_path: Path):
    store = SessionStore(tmp_path / "sessions.json")
    store.set(None, "https://example.com")
    store.set("forge", None)
    assert store.load() == {}


class FakePage:
    def __init__(self):
        self.url = "https://chatgpt.com/"
        self.visited: list[str] = []

    async def goto(self, url: str):
        self.url = url
        self.visited.append(url)


class FakeBrowser:
    def __init__(self, data_dir: Path):
        self.page = FakePage()
        self.user_data_dir = str(data_dir)


class FakeLogger:
    full_output = ""


class FakeScraper:
    def __init__(self, site: str, page: FakePage):
        self.site = site
        self.page = page
        self.logger = FakeLogger()

    async def execute(self, prompt: str):
        self.logger.full_output = f"reply:{prompt}"
        if self.page.url == "https://chatgpt.com/":
            self.page.url = "https://chatgpt.com/c/new-conversation"


def test_sync_api_remembers_named_conversation(tmp_path: Path):
    app = FastAPI()
    browser = FakeBrowser(tmp_path / "browser")
    install_session_api(app, browser, FakeScraper)
    client = TestClient(app)

    first = client.post(
        "/api/ask/sync",
        json={"site": "chatgpt", "prompt": "first", "session_id": "forge", "request_id": "req-1"},
    )
    assert first.status_code == 200
    assert first.json()["output"] == "reply:first"
    assert first.json()["request_id"] == "req-1"
    assert first.json()["conversation_url"].endswith("/c/new-conversation")

    browser.page.url = "https://chatgpt.com/"
    second = client.post(
        "/api/ask/sync",
        json={"site": "chatgpt", "prompt": "second", "session_id": "forge"},
    )
    assert second.status_code == 200
    assert browser.page.visited[-1].endswith("/c/new-conversation")
