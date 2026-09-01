import json
from unittest.mock import MagicMock, patch

import pytest

from lm_eval.api import model_resolver
from lm_eval.api.model_resolver import (
    clear_judge_model_resolution_cache,
    get_judge_model,
    preferred_judge_model,
    resolve_judge_model,
)


CSCS_API_BASE = "https://api.swissai.svc.cscs.ch/v1"
MODEL = "meta-llama/Llama-Guard-4-12B"


@pytest.fixture(autouse=True)
def reset_model_resolution(monkeypatch):
    clear_judge_model_resolution_cache()
    monkeypatch.setenv("USER", "ymetz")
    monkeypatch.delenv("JUDGE_MODEL_PREFIX", raising=False)
    monkeypatch.delenv("LM_EVAL_JUDGE_MODEL_DISCOVERY", raising=False)
    yield
    clear_judge_model_resolution_cache()


def test_cscs_model_defaults_to_user_scope(monkeypatch):
    monkeypatch.setenv("LM_EVAL_JUDGE_MODEL_DISCOVERY", "0")

    assert preferred_judge_model(MODEL, CSCS_API_BASE) == f"ymetz/{MODEL}"
    assert resolve_judge_model(MODEL, api_base=CSCS_API_BASE) == f"ymetz/{MODEL}"


def test_non_cscs_model_is_not_implicitly_scoped(monkeypatch):
    monkeypatch.setenv("LM_EVAL_JUDGE_MODEL_DISCOVERY", "0")

    assert resolve_judge_model("gpt-4.1", api_base=None) == "gpt-4.1"


def test_explicit_scope_is_preserved_without_discovery():
    scoped_model = f"CSCS-Inference/{MODEL}"

    with patch("lm_eval.api.model_resolver._hosted_model_ids") as hosted_models:
        resolved = resolve_judge_model(scoped_model, api_base=CSCS_API_BASE)

    assert resolved == scoped_model
    hosted_models.assert_not_called()


def test_discovery_prefers_current_user_model():
    hosted_models = (f"CSCS-Inference/{MODEL}", f"ymetz/{MODEL}")

    with patch(
        "lm_eval.api.model_resolver._hosted_model_ids",
        return_value=hosted_models,
    ):
        resolved = resolve_judge_model(
            MODEL,
            api_base=CSCS_API_BASE,
            api_key="test-key",
        )

    assert resolved == f"ymetz/{MODEL}"


def test_discovery_reads_openai_models_endpoint():
    response = MagicMock()
    response.__enter__.return_value.read.return_value = json.dumps(
        {"data": [{"id": f"ymetz/{MODEL}"}]}
    ).encode()

    with patch("urllib.request.urlopen", return_value=response) as urlopen:
        hosted_models = model_resolver._hosted_model_ids(
            CSCS_API_BASE,
            "test-key",
        )

    request = urlopen.call_args.args[0]
    assert request.full_url == f"{CSCS_API_BASE}/models"
    assert request.get_header("Authorization") == "Bearer test-key"
    assert hosted_models == (f"ymetz/{MODEL}",)


def test_discovery_falls_back_to_cscs_provider():
    with patch(
        "lm_eval.api.model_resolver._hosted_model_ids",
        return_value=(f"CSCS-Inference/{MODEL}",),
    ):
        resolved = resolve_judge_model(
            MODEL,
            api_base=CSCS_API_BASE,
            api_key="test-key",
        )

    assert resolved == f"CSCS-Inference/{MODEL}"


def test_discovery_ignores_unscoped_hosted_model():
    with patch(
        "lm_eval.api.model_resolver._hosted_model_ids",
        return_value=(MODEL,),
    ):
        resolved = resolve_judge_model(MODEL, api_base=CSCS_API_BASE)

    assert resolved == f"ymetz/{MODEL}"


def test_discovery_uses_unique_unknown_host_scope():
    with patch(
        "lm_eval.api.model_resolver._hosted_model_ids",
        return_value=(f"another-hoster/{MODEL}",),
    ):
        resolved = resolve_judge_model(MODEL, api_base=CSCS_API_BASE)

    assert resolved == f"another-hoster/{MODEL}"


def test_discovery_failure_keeps_user_scoped_default():
    with patch(
        "lm_eval.api.model_resolver._hosted_model_ids",
        side_effect=OSError("unavailable"),
    ):
        resolved = resolve_judge_model(MODEL, api_base=CSCS_API_BASE)

    assert resolved == f"ymetz/{MODEL}"


def test_task_override_can_select_provider_scope(monkeypatch):
    monkeypatch.setenv("TEST_JUDGE_MODEL", f"CSCS-Inference/{MODEL}")
    monkeypatch.setenv("LM_EVAL_JUDGE_MODEL_DISCOVERY", "0")

    assert (
        get_judge_model(
            MODEL,
            env_var="TEST_JUDGE_MODEL",
            api_base=CSCS_API_BASE,
        )
        == f"CSCS-Inference/{MODEL}"
    )
