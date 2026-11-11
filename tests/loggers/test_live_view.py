"""The live view of a running API evaluation (lm_eval/loggers/live_view.py)."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from lm_eval.loggers import live_view
from lm_eval.models.api_models import JsonChatStr
from lm_eval.models.openai_completions import LocalChatCompletion, LocalCompletionsAPI


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    live_view._state.clear()
    monkeypatch.setattr(live_view, "_warned", False)
    monkeypatch.delenv(live_view.ENV, raising=False)


def _lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_nothing_is_written_unless_switched_on(tmp_path):
    live_view.record("p", "a")
    assert list(tmp_path.iterdir()) == []


def test_the_default_path_is_beside_the_results():
    assert live_view.default_path(None) is None
    assert live_view.default_path("out/run").endswith("out/run/live_view.jsonl")
    assert live_view.default_path("out/results.json") == "out/live_view.jsonl"


def test_the_first_requests_are_kept_in_full_and_all_are_counted(tmp_path, monkeypatch):
    path = tmp_path / "run" / "live_view.jsonl"
    monkeypatch.setenv(live_view.ENV, str(path))
    for number in range(live_view.KEEP + 5):
        live_view.record(f"prompt {number}", f"answer {number}", "thinking", 12, 9)
    live_view.finish()
    lines = _lines(path)
    assert len(lines) == live_view.KEEP
    assert lines[0] == {
        "n": 1,
        "prompt": "prompt 0",
        "answer": "answer 0",
        "reasoning": "thinking",
        "output_tokens": 12,
        "reasoning_tokens": 9,
    }
    progress = json.loads((path.parent / "live_view_progress.json").read_text())
    assert progress["done"] == live_view.KEEP + 5


def test_long_text_is_capped(tmp_path, monkeypatch):
    path = tmp_path / "live_view.jsonl"
    monkeypatch.setenv(live_view.ENV, str(path))
    live_view.record("p" * (live_view.TEXT_CAP + 7), "a")
    assert _lines(path)[0]["prompt"].endswith("[... 7 more characters]")


def test_a_failure_to_write_never_reaches_the_evaluation(tmp_path, monkeypatch):
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    monkeypatch.setenv(live_view.ENV, str(blocker / "live_view.jsonl"))
    live_view.record("p", "a")  # must not raise
    live_view.finish()


def _session(response_json):
    response = AsyncMock()
    response.ok = True
    response.json = AsyncMock(return_value=response_json)
    response.raise_for_status = lambda: None

    class Context:
        async def __aenter__(self):
            return response

        async def __aexit__(self, *exc):
            pass

    session = MagicMock()
    session.post.return_value = Context()
    return session


def test_a_completions_generation_reaches_the_live_view(tmp_path, monkeypatch):
    path = tmp_path / "live_view.jsonl"
    monkeypatch.setenv(live_view.ENV, str(path))
    model = LocalCompletionsAPI(base_url="http://x", tokenizer_backend=None, model="m")
    session = _session({"choices": [{"index": 0, "text": "42"}]})
    asyncio.run(
        model.amodel_call(session, asyncio.Semaphore(1), ["6*7="], gen_kwargs={})
    )
    assert _lines(path)[0]["prompt"] == "6*7="
    assert _lines(path)[0]["answer"] == "42"


def test_a_chat_generation_carries_its_reasoning_and_tokens(tmp_path, monkeypatch):
    path = tmp_path / "live_view.jsonl"
    monkeypatch.setenv(live_view.ENV, str(path))
    model = LocalChatCompletion(base_url="http://x", tokenizer_backend=None, model="m")
    chat = JsonChatStr(json.dumps([{"role": "user", "content": "6*7="}]))
    session = _session(
        {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "content": "42",
                        "reasoning_content": "six times seven",
                    },
                }
            ],
            "usage": {
                "completion_tokens": 30,
                "completion_tokens_details": {"reasoning_tokens": 25},
            },
        }
    )
    asyncio.run(model.amodel_call(session, asyncio.Semaphore(1), [chat], gen_kwargs={}))
    line = _lines(path)[0]
    assert line["prompt"] == "user: 6*7="
    assert line["answer"] == "42"
    assert line["reasoning"] == "six times seven"
    assert (line["output_tokens"], line["reasoning_tokens"]) == (30, 25)


def test_the_request_body_is_kept_with_long_values_cut(tmp_path, monkeypatch):
    path = tmp_path / "live_view.jsonl"
    monkeypatch.setenv(live_view.ENV, str(path))
    request = {
        "model": "m",
        "temperature": 0.6,
        "messages": [{"role": "user", "content": "x" * (live_view.TEXT_CAP + 3)}],
        "prompt": list(range(live_view.REQUEST_LIST_CAP + 4)),
    }
    live_view.record("p", "a", request=request)
    kept = _lines(path)[0]["request"]
    assert kept["model"] == "m" and kept["temperature"] == 0.6
    assert kept["messages"][0]["content"].endswith("[... 3 more characters]")
    assert len(kept["prompt"]) == live_view.REQUEST_LIST_CAP + 1
    assert kept["prompt"][-1] == "[... 4 more items]"
    # Without one, the line has no request key at all.
    live_view.record("p2", "a2")
    assert "request" not in _lines(path)[1]


def test_a_chat_generation_records_the_body_it_sent(tmp_path, monkeypatch):
    path = tmp_path / "live_view.jsonl"
    monkeypatch.setenv(live_view.ENV, str(path))
    model = LocalChatCompletion(base_url="http://x", tokenizer_backend=None, model="m")
    chat = JsonChatStr(json.dumps([{"role": "user", "content": "6*7="}]))
    session = _session({"choices": [{"index": 0, "message": {"content": "42"}}]})
    asyncio.run(model.amodel_call(session, asyncio.Semaphore(1), [chat], gen_kwargs={}))
    body = _lines(path)[0]["request"]
    assert body["model"] == "m"
    assert body["messages"] == [{"role": "user", "content": "6*7="}]
    assert "api_key" not in json.dumps(body).lower()
