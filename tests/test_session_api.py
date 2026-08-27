import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    def __init__(self, data_dir: Path, page=None):
        self.page = page or FakePage()
        self.user_data_dir = str(data_dir)


class FakeLogger:
    def __init__(self):
        self.full_output = ""
        self.prompt_text = ""

    def init_turn(self, prompt: str):
        self.prompt_text = prompt
        self.full_output = ""

    def append_chunk(self, node_type: str, chunk: str):
        if node_type == "final_output":
            self.full_output += chunk


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


class AtomicFakeLocator:
    def __init__(self):
        self.filled: list[str] = []

    async def click(self):
        return None

    async def focus(self):
        return None

    async def fill(self, text: str):
        self.filled.append(text)


class AtomicFakeWrapper:
    def __init__(self, text: str):
        self.text = text

    async def evaluate(self, expression: str, selectors):
        return self.text

    async def inner_text(self):
        return self.text


class AtomicFakeKeyboard:
    def __init__(self, page):
        self.page = page
        self.presses: list[str] = []

    async def press(self, key: str):
        self.presses.append(key)
        if key == "Enter":
            self.page.wrappers.append(
                AtomicFakeWrapper('{"decision":"continue","guidance":"bridge-ok"}')
            )
            self.page.url = "https://chatgpt.com/c/current-conversation"


class AtomicFakePage:
    def __init__(self):
        self.url = "https://chatgpt.com/c/current-conversation"
        self.visited: list[str] = []
        self.input = AtomicFakeLocator()
        self.wrappers: list[AtomicFakeWrapper] = []
        self.keyboard = AtomicFakeKeyboard(self)

    async def goto(self, url: str):
        self.url = url
        self.visited.append(url)

    async def wait_for_selector(self, selector: str):
        return None

    def locator(self, selector: str):
        return self.input

    async def query_selector_all(self, selector: str):
        return list(self.wrappers)

    async def click(self, selector: str):
        return None


class AtomicFakeScraper:
    def __init__(self, site: str, page: AtomicFakePage):
        self.site = site
        self.page = page
        self.logger = FakeLogger()
        self.config = {"base_url": "https://chatgpt.com"}
        self.selectors = {
            "input_box": "#prompt-textarea",
            "enter_to_submit": True,
            "ai_answer_container": "assistant-turn",
            "exclude_selectors": ["button"],
        }

    async def track_stream(self, pre_send_count: int = 0):
        # Simulate the site-specific markdown parser missing a newly changed DOM.
        # The session API must still recover the visible assistant output.
        return None


def test_sync_api_preserves_multiline_prompt_and_falls_back_to_visible_output(tmp_path: Path):
    app = FastAPI()
    page = AtomicFakePage()
    browser = FakeBrowser(tmp_path / "browser", page=page)
    install_session_api(app, browser, AtomicFakeScraper)
    client = TestClient(app)

    prompt = "[FORGE BRIDGE SMOKE TEST]\nline two\nline three"
    response = client.post(
        "/api/ask/sync",
        json={
            "site": "chatgpt",
            "prompt": prompt,
            "session_id": "forge-supervisor",
            "request_id": "bridge-smoke-001",
            "conversation_url": "https://chatgpt.com/c/current-conversation",
        },
    )

    assert response.status_code == 200
    assert page.input.filled == [prompt]
    assert page.keyboard.presses == ["Enter"]
    assert response.json()["output"] == '{"decision":"continue","guidance":"bridge-ok"}'
    assert response.json()["request_id"] == "bridge-smoke-001"
