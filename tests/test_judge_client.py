"""Unit tests for the shared judge client (lm_eval.api.judge)."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from lm_eval.api import judge


_JUDGE_ENV_VARS = (
    "LM_EVAL_JUDGE_BACKEND",
    "LM_EVAL_JUDGE_MODEL",
    "LM_EVAL_JUDGE_MAX_WORKERS",
    "LM_EVAL_JUDGE_MAX_RETRIES",
    "LM_EVAL_JUDGE_TIMEOUT",
    "LM_EVAL_JUDGE_STRIP_THINK",
    "LM_EVAL_JUDGE_LOG_PATH",
    "CSCS_SERVING_API",
    "OPENAI_API_KEY",
)


@pytest.fixture(autouse=True)
def _clear_judge_env(monkeypatch: pytest.MonkeyPatch):
    """Strip every env var the judge module reads, before each test, so
    nothing leaks across tests or from the developer's shell.
    """
    for key in _JUDGE_ENV_VARS:
        monkeypatch.delenv(key, raising=False)
    yield


def _cfg(
    *, strip_think: bool = True, max_workers: int = 4, log_path: str | None = None
):
    return judge.JudgeConfig(
        backend="cscs",
        model="test-judge",
        base_url="https://example.invalid/v1",
        api_key="x",
        max_workers=max_workers,
        max_retries=0,
        timeout=10.0,
        strip_think=strip_think,
        log_path=log_path,
    )


# ---------------------------------------------------------------------------
# judge_config_from_env: backend / model / API key resolution
# ---------------------------------------------------------------------------


def test_default_backend_is_cscs(monkeypatch):
    monkeypatch.setenv("CSCS_SERVING_API", "secret-cscs")

    cfg = judge.judge_config_from_env()

    assert cfg.backend == "cscs"
    assert cfg.base_url == "https://api.swissai.svc.cscs.ch/v1"
    assert cfg.api_key == "secret-cscs"
    assert cfg.model == "Qwen/Qwen3.5-27B"
    assert cfg.max_workers == 32
    assert cfg.max_retries == 8
    assert cfg.timeout == 800.0
    assert cfg.strip_think is True


def test_openai_backend_via_env(monkeypatch):
    monkeypatch.setenv("LM_EVAL_JUDGE_BACKEND", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")

    cfg = judge.judge_config_from_env()

    assert cfg.backend == "openai"
    assert cfg.base_url == "https://api.openai.com/v1"
    assert cfg.api_key == "sk-secret"
    assert cfg.model == "gpt-4o"


def test_backend_arg_is_lowercased(monkeypatch):
    monkeypatch.setenv("CSCS_SERVING_API", "x")
    cfg = judge.judge_config_from_env(backend="CSCS")
    assert cfg.backend == "cscs"


def test_model_env_overrides_backend_default(monkeypatch):
    monkeypatch.setenv("CSCS_SERVING_API", "secret-cscs")
    monkeypatch.setenv("LM_EVAL_JUDGE_MODEL", "Qwen/Qwen3.5-32B")

    cfg = judge.judge_config_from_env()

    assert cfg.model == "Qwen/Qwen3.5-32B"


def test_explicit_model_arg_overrides_env(monkeypatch):
    """Per-task pin (e.g. FollowBench wanting gpt-4-1106-preview) wins."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.setenv("LM_EVAL_JUDGE_MODEL", "gpt-4o")

    cfg = judge.judge_config_from_env(backend="openai", model="gpt-4-1106-preview")

    assert cfg.backend == "openai"
    assert cfg.model == "gpt-4-1106-preview"


def test_unknown_backend_raises(monkeypatch):
    monkeypatch.setenv("LM_EVAL_JUDGE_BACKEND", "bedrock")
    with pytest.raises(ValueError, match="Unknown LM_EVAL_JUDGE_BACKEND"):
        judge.judge_config_from_env()


def test_missing_api_key_raises():
    with pytest.raises(RuntimeError, match="CSCS_SERVING_API"):
        judge.judge_config_from_env()


def test_missing_openai_key_names_correct_var(monkeypatch):
    monkeypatch.setenv("LM_EVAL_JUDGE_BACKEND", "openai")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        judge.judge_config_from_env()


@pytest.mark.parametrize(
    "raw,expected",
    [("8", 8), ("0", 0), ("32", 32)],
)
def test_max_workers_from_env(monkeypatch, raw, expected):
    monkeypatch.setenv("CSCS_SERVING_API", "secret")
    monkeypatch.setenv("LM_EVAL_JUDGE_MAX_WORKERS", raw)
    cfg = judge.judge_config_from_env()
    assert cfg.max_workers == expected


def test_max_workers_bad_value_falls_back(monkeypatch):
    monkeypatch.setenv("CSCS_SERVING_API", "secret")
    monkeypatch.setenv("LM_EVAL_JUDGE_MAX_WORKERS", "not-an-int")
    cfg = judge.judge_config_from_env()
    assert cfg.max_workers == 32


def test_max_retries_and_timeout_from_env(monkeypatch):
    monkeypatch.setenv("CSCS_SERVING_API", "secret")
    monkeypatch.setenv("LM_EVAL_JUDGE_MAX_RETRIES", "3")
    monkeypatch.setenv("LM_EVAL_JUDGE_TIMEOUT", "120.5")
    cfg = judge.judge_config_from_env()
    assert cfg.max_retries == 3
    assert cfg.timeout == 120.5


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("0", False),
        ("false", False),
        ("False", False),
        ("no", False),
        ("off", False),
        ("1", True),
        ("true", True),
        ("yes", True),
    ],
)
def test_strip_think_from_env(monkeypatch, raw, expected):
    monkeypatch.setenv("CSCS_SERVING_API", "secret")
    monkeypatch.setenv("LM_EVAL_JUDGE_STRIP_THINK", raw)
    cfg = judge.judge_config_from_env()
    assert cfg.strip_think is expected


def test_strip_think_kwarg_overrides_env(monkeypatch):
    monkeypatch.setenv("CSCS_SERVING_API", "secret")
    monkeypatch.setenv("LM_EVAL_JUDGE_STRIP_THINK", "0")
    cfg = judge.judge_config_from_env(strip_think=True)
    assert cfg.strip_think is True


def test_log_path_picked_up(monkeypatch, tmp_path):
    monkeypatch.setenv("CSCS_SERVING_API", "secret")
    log = tmp_path / "judge.jsonl"
    monkeypatch.setenv("LM_EVAL_JUDGE_LOG_PATH", str(log))
    cfg = judge.judge_config_from_env()
    assert cfg.log_path == str(log)


# ---------------------------------------------------------------------------
# strip_think_tags
# ---------------------------------------------------------------------------


def test_strip_think_removes_block():
    text = "<think>internal\nreasoning</think>\nFinal verdict: A"
    assert judge.strip_think_tags(text) == "Final verdict: A"


def test_strip_think_handles_multiple_blocks_and_case():
    text = "<THINK>a</THINK>x<think>b</think>y"
    assert judge.strip_think_tags(text) == "xy"


def test_strip_think_handles_orphaned_close_tag():
    """Chat template ate the opener — we still want the trailing verdict."""
    text = "stuff stuff </think>Final verdict: B"
    assert judge.strip_think_tags(text) == "Final verdict: B"


def test_strip_think_no_op_when_absent():
    assert judge.strip_think_tags("plain output") == "plain output"


def test_strip_think_handles_empty():
    assert judge.strip_think_tags("") == ""
    assert judge.strip_think_tags(None) is None


def test_strip_think_unterminated_open_left_untouched():
    """Unterminated <think> with no closing tag is left alone — better
    than nuking the entire output.
    """
    text = "<think>unterminated reasoning that never closes"
    assert judge.strip_think_tags(text) == text


# ---------------------------------------------------------------------------
# run_judge_calls: order preservation, concurrency, failure handling
# ---------------------------------------------------------------------------


def _fake_client_with_responses(responses):
    """Build a mock OpenAI client whose chat.completions.create returns or
    raises the configured per-call values in dispatch order.
    """
    state = {"i": 0}

    def _create(**kwargs):
        idx = state["i"]
        state["i"] += 1
        resp = responses[idx % len(responses)]
        if isinstance(resp, Exception):
            raise resp
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=resp))]
        )

    client = MagicMock()
    client.chat.completions.create.side_effect = _create
    return client


def test_run_judge_calls_preserves_order():
    """The contract is index-stable: outputs[i] always matches prompts[i]
    regardless of which thread completed first. Drive the mock with a
    deterministic transform of the prompt so thread-scheduling cannot
    perturb the assertion.
    """

    def _create(**kwargs):
        # Echo the last user message back as the "verdict" — order-stable
        # by construction (each prompt has a unique content).
        echo = kwargs["messages"][-1]["content"]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=f"v:{echo}"))]
        )

    client = MagicMock()
    client.chat.completions.create.side_effect = _create

    prompts = [{"messages": [{"role": "user", "content": f"p{i}"}]} for i in range(8)]
    with patch.object(judge, "get_judge_client", return_value=client):
        out = judge.run_judge_calls(prompts, config=_cfg(max_workers=4))

    assert out == [f"v:p{i}" for i in range(8)]
    assert client.chat.completions.create.call_count == 8


def test_run_judge_calls_strips_think():
    client = _fake_client_with_responses(["<think>scratch</think>\nA wins"])
    with patch.object(judge, "get_judge_client", return_value=client):
        out = judge.run_judge_calls(
            [{"messages": [{"role": "user", "content": "x"}]}],
            config=_cfg(strip_think=True),
        )
    assert out == ["A wins"]


def test_run_judge_calls_keeps_think_when_disabled():
    client = _fake_client_with_responses(["<think>scratch</think>\nA wins"])
    with patch.object(judge, "get_judge_client", return_value=client):
        out = judge.run_judge_calls(
            [{"messages": [{"role": "user", "content": "x"}]}],
            config=_cfg(strip_think=False),
        )
    assert out == ["<think>scratch</think>\nA wins"]


def test_run_judge_calls_failed_call_returns_none():
    client = _fake_client_with_responses([RuntimeError("kaboom")])
    with patch.object(judge, "get_judge_client", return_value=client):
        out = judge.run_judge_calls(
            [{"messages": [{"role": "user", "content": "x"}]}],
            config=_cfg(),
        )
    assert out == [None]


def test_run_judge_calls_mixed_success_and_failure(tmp_path):
    """Even when half the calls fail, the index-stable contract holds and
    the JSONL log records one entry per call (success or failure).
    """
    log_path = tmp_path / "judge.jsonl"

    def _create(**kwargs):
        msg = kwargs["messages"][-1]["content"]
        if msg.startswith("fail"):
            raise RuntimeError(f"boom-{msg}")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=f"v:{msg}"))]
        )

    client = MagicMock()
    client.chat.completions.create.side_effect = _create

    prompts = [
        {"messages": [{"role": "user", "content": "ok-0"}], "tag": 0},
        {"messages": [{"role": "user", "content": "fail-1"}], "tag": 1},
        {"messages": [{"role": "user", "content": "ok-2"}], "tag": 2},
        {"messages": [{"role": "user", "content": "fail-3"}], "tag": 3},
    ]
    with patch.object(judge, "get_judge_client", return_value=client):
        out = judge.run_judge_calls(prompts, config=_cfg(log_path=str(log_path)))

    assert out == ["v:ok-0", None, "v:ok-2", None]

    recs = [json.loads(line) for line in log_path.read_text().splitlines()]
    # Logged in input order; one record per call.
    assert [r["tag"] for r in recs] == [0, 1, 2, 3]
    assert recs[0]["output"] == "v:ok-0" and recs[0]["error"] is None
    assert recs[1]["output"] is None and "boom-fail-1" in recs[1]["error"]
    assert recs[2]["output"] == "v:ok-2" and recs[2]["error"] is None
    assert recs[3]["output"] is None and "boom-fail-3" in recs[3]["error"]


def test_run_judge_calls_empty_input_is_noop():
    with patch.object(judge, "get_judge_client") as fake:
        out = judge.run_judge_calls([], config=_cfg())
    assert out == []
    fake.assert_not_called()


def test_run_judge_calls_writes_jsonl_log_in_input_order(tmp_path):
    log_path = tmp_path / "judge.jsonl"
    client = _fake_client_with_responses(["v0", "v1"])

    with patch.object(judge, "get_judge_client", return_value=client):
        judge.run_judge_calls(
            [
                {"messages": [{"role": "user", "content": "a"}], "tag": "row-0"},
                {"messages": [{"role": "user", "content": "b"}], "tag": "row-1"},
            ],
            config=_cfg(log_path=str(log_path)),
        )

    recs = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [r["tag"] for r in recs] == ["row-0", "row-1"]
    for rec in recs:
        assert rec["backend"] == "cscs"
        assert rec["model"] == "test-judge"
        assert rec["error"] is None


def test_run_judge_calls_log_write_failure_is_swallowed(tmp_path, caplog):
    """A bad log path should not break the eval — log a warning and move on."""
    bad_path = tmp_path / "does" / "not" / "exist.jsonl"
    client = _fake_client_with_responses(["v0"])

    with (
        caplog.at_level(logging.WARNING, logger="lm_eval.api.judge"),
        patch.object(judge, "get_judge_client", return_value=client),
    ):
        out = judge.run_judge_calls(
            [{"messages": [{"role": "user", "content": "x"}], "tag": 0}],
            config=_cfg(log_path=str(bad_path)),
        )
    assert out == ["v0"]
    assert any("failed to write judge log" in r.message for r in caplog.records)


def test_run_judge_calls_passes_extra_body():
    captured: dict = {}

    def _create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )

    client = MagicMock()
    client.chat.completions.create.side_effect = _create

    with patch.object(judge, "get_judge_client", return_value=client):
        judge.run_judge_calls(
            [{"messages": [{"role": "user", "content": "x"}]}],
            config=_cfg(),
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            max_tokens=2048,
        )

    assert captured["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert captured["max_tokens"] == 2048
    assert captured["temperature"] == 0.0
    assert captured["model"] == "test-judge"


def test_run_judge_calls_invokes_on_result_callback():
    client = _fake_client_with_responses(["a", "b", "c"])
    seen = []
    with patch.object(judge, "get_judge_client", return_value=client):
        judge.run_judge_calls(
            [
                {"messages": [{"role": "user", "content": "x"}], "tag": i}
                for i in range(3)
            ],
            config=_cfg(max_workers=1),
            on_result=lambda idx, out: seen.append((idx, out)),
        )
    assert sorted(seen) == [(0, "a"), (1, "b"), (2, "c")]
