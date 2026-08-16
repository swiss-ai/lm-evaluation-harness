"""Unit tests for lm-eval-managed vLLM data-parallel replicas."""

from __future__ import annotations

import importlib
import importlib.metadata
import sys
from types import ModuleType
from unittest.mock import patch

from lm_eval.api.registry import model_registry


def _load_vllm_adapter(monkeypatch):
    reservations: list[int] = []
    engine_calls: list[tuple[dict, dict]] = []
    shutdowns: list[bool] = []

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeTokensPrompt:
        def __init__(self, prompt_token_ids):
            self.prompt_token_ids = prompt_token_ids

    class FakeLLM:
        def __init__(self, **model_args):
            dp_env = {
                key: value
                for key, value in __import__("os").environ.items()
                if key.startswith("VLLM_DP_")
            }
            engine_calls.append((model_args, dp_env))

        def generate(self, prompts, **_kwargs):
            return [prompt.prompt_token_ids for prompt in prompts]

    vllm = ModuleType("vllm")
    vllm.__path__ = []
    vllm.LLM = FakeLLM
    vllm.SamplingParams = FakeSamplingParams
    vllm.TokensPrompt = FakeTokensPrompt
    vllm_lora = ModuleType("vllm.lora")
    vllm_lora.__path__ = []
    vllm_lora_request = ModuleType("vllm.lora.request")
    vllm_lora_request.LoRARequest = object
    vllm_tokenizers = ModuleType("vllm.tokenizers")
    vllm_tokenizers.get_tokenizer = lambda *_args, **_kwargs: None

    ray = ModuleType("ray")

    def remote(*, num_gpus):
        reservations.append(num_gpus)

        def decorate(function):
            class RemoteFunction:
                @staticmethod
                def remote(*args):
                    return function(*args)

            return RemoteFunction

        return decorate

    ray.remote = remote
    ray.get = lambda refs: refs
    ray.shutdown = lambda: shutdowns.append(True)

    for name, module in {
        "vllm": vllm,
        "vllm.lora": vllm_lora,
        "vllm.lora.request": vllm_lora_request,
        "vllm.tokenizers": vllm_tokenizers,
        "ray": ray,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    # Other tests may already have materialized the lazy registry entry. Re-importing
    # this module creates a distinct VLLM class object, which duplicate-alias protection
    # correctly rejects. Restore the lazy placeholder for this isolated import; the
    # monkeypatch fixture puts the previous registry value back after the test.
    monkeypatch.setitem(
        model_registry._objs,
        "vllm",
        "lm_eval.models.vllm_causallms:VLLM",
    )
    sys.modules.pop("lm_eval.models.vllm_causallms", None)
    with patch.object(importlib.metadata, "version", return_value="0.22.1"):
        adapter = importlib.import_module("lm_eval.models.vllm_causallms")
    return adapter, reservations, engine_calls, shutdowns


def test_dense_dp_uses_independent_gpu_reserved_ray_replicas(monkeypatch):
    adapter, reservations, engine_calls, shutdowns = _load_vllm_adapter(monkeypatch)
    for key in (
        "VLLM_DP_RANK",
        "VLLM_DP_RANK_LOCAL",
        "VLLM_DP_SIZE",
        "VLLM_DP_MASTER_IP",
        "VLLM_DP_MASTER_PORT",
    ):
        monkeypatch.setenv(key, "leaked")

    model = object.__new__(adapter.VLLM)
    model.data_parallel_size = 2
    model.tensor_parallel_size = 1
    model.model_args = {
        "model": "test-model",
        "distributed_executor_backend": "ray",
    }
    model.lora_request = None

    requests = [[1], [2], [3], [4]]
    sampling = [adapter.SamplingParams(max_tokens=1) for _ in requests]
    outputs = adapter.VLLM._model_generate(
        model, requests, generate=True, sampling_params=sampling
    )

    assert outputs == requests
    assert reservations == [1]
    assert len(engine_calls) == 2
    assert all("distributed_executor_backend" not in args for args, _ in engine_calls)
    assert all(not dp_env for _, dp_env in engine_calls)
    assert shutdowns == [True]
