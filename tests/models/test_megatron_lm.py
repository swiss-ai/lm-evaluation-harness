import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.models import megatron_lm
from lm_eval.models.megatron_lm import (
    MegatronLMEval,
    _get_experimental_attention_variant_spec,
    _override_attention_mask_type,
)


class _FakeTokenizer:
    eod = 0
    bos = 1

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(char) - ord("a") + 2 for char in text]

    def decode(self, tokens, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(token - 2 + ord("a")) for token in tokens if token >= 2)


def _bare_model():
    model = MegatronLMEval.__new__(MegatronLMEval)
    model._tp_size = 1
    model._pp_size = 1
    model._ep_size = 1
    model._args = SimpleNamespace(sequence_parallel=False)
    return model


def _likelihood_model(*, batch_size=1, tp_size=1, sequence_parallel=False):
    model = _bare_model()
    LM.__init__(model)
    model.tokenizer = SimpleNamespace(eod=0)
    model._device = torch.device("cpu")
    model._global_rank = 0
    model._max_length = 16
    model._batch_size = batch_size
    model._tp_size = tp_size
    model._args = SimpleNamespace(sequence_parallel=sequence_parallel)
    return model


def test_parallelism_validation_and_dp_rank_mapping():
    cases = [
        (1, 1, 1, 1, "single"),
        (4, 1, 1, 1, "data_parallel"),
        (4, 4, 1, 1, "tensor_parallel"),
        (4, 2, 1, 1, "tensor_data_parallel"),
        (4, 2, 1, 2, "tensor_data_parallel"),
        (2, 1, 2, 1, "pipeline_parallel"),
        (8, 2, 2, 1, "pipeline_parallel"),
    ]
    for devices, tp, pp, ep, expected in cases:
        model = _bare_model()
        model._validate_parallelism_config(devices, tp, pp, ep)
        assert model._parallelism_mode == expected

    with pytest.raises(ValueError, match="divisible by EP"):
        _bare_model()._validate_parallelism_config(6, 1, 1, 4)
    with pytest.raises(ValueError, match="divisible by TP"):
        _bare_model()._validate_parallelism_config(5, 2, 1, 1)
    with pytest.raises(ValueError, match=r"divisible by TP \* PP"):
        _bare_model()._validate_parallelism_config(6, 2, 2, 1)

    model = _bare_model()
    model._tp_size = 2
    model._global_rank = 3
    model._dp_rank = 1
    model._dp_world_size = 2
    model._set_parallelism(devices=4)
    assert (model.rank, model.world_size, model.cache_rank) == (1, 2, 3)
    assert not model.is_main_process


def test_loglikelihood_batches_requests():
    model = _likelihood_model(batch_size=2)
    model.tokenizer = _FakeTokenizer()
    model._max_length = 16
    batch_sizes = []

    def forward(input_ids, attention_mask=None):
        del attention_mask
        batch_sizes.append(input_ids.shape[0])
        logits = torch.zeros((*input_ids.shape, 32))
        for batch, row in enumerate(input_ids):
            for position in range(len(row) - 1):
                logits[batch, position, row[position + 1]] = 10
        return logits

    model._model_forward = forward
    requests = [Instance("loglikelihood", {}, ("a", "b"), idx) for idx in range(5)]
    results = model.loglikelihood(requests)
    assert len(results) == 5
    assert all(greedy for _, greedy in results)
    assert batch_sizes == [2, 2, 1]


def test_attention_mask_override_skips_specs_without_mask_parameter():
    def layer(params):
        return SimpleNamespace(
            submodules=SimpleNamespace(self_attention=SimpleNamespace(params=params))
        )

    kda = layer({})
    full_attention = layer({"attn_mask_type": "causal"})
    block = SimpleNamespace(layer_specs=[kda, full_attention])
    assert _override_attention_mask_type(block, "arbitrary") == 1
    assert kda.submodules.self_attention.params == {}
    assert full_attention.submodules.self_attention.params == {
        "attn_mask_type": "arbitrary"
    }


_EXPERIMENTAL_SPEC_MODULE = (
    "megatron.core.models.gpt.experimental_attention_variant_module_specs"
)


def test_ordinary_moe_does_not_require_experimental_attention_support(monkeypatch):
    monkeypatch.setitem(sys.modules, _EXPERIMENTAL_SPEC_MODULE, None)
    args = SimpleNamespace(experimental_attention_variant=None, num_experts=8)
    assert _get_experimental_attention_variant_spec(args, object()) is None


def test_experimental_attention_uses_supported_builder(monkeypatch):
    config = object()
    expected = object()
    module = ModuleType(_EXPERIMENTAL_SPEC_MODULE)

    def get_spec(received):
        assert received is config
        return expected

    module.get_transformer_block_with_experimental_attention_variant_spec = get_spec
    monkeypatch.setitem(sys.modules, _EXPERIMENTAL_SPEC_MODULE, module)
    args = SimpleNamespace(experimental_attention_variant="kda", num_experts=8)
    assert _get_experimental_attention_variant_spec(args, config) is expected


def test_experimental_attention_reports_unsupported_megatron(monkeypatch):
    monkeypatch.setitem(sys.modules, _EXPERIMENTAL_SPEC_MODULE, None)
    args = SimpleNamespace(experimental_attention_variant="kda", num_experts=8)
    with pytest.raises(ImportError, match=r"does not provide.*get_transformer"):
        _get_experimental_attention_variant_spec(args, object())


class _MetricModel(torch.nn.Module):
    pg_collection = object()

    def forward(self, input_ids, position_ids, attention_mask):
        del position_ids, attention_mask
        return torch.zeros((*input_ids.shape, 8))


def test_model_forward_collects_backend_metrics():
    model = _bare_model()
    LM.__init__(model)
    model.model = _MetricModel()
    model._global_rank = 0
    model._ep_size = 1
    model._args = SimpleNamespace()
    calls = []

    def collect(received, pg_collection=None):
        calls.append((received, pg_collection))
        return {"router_max": 0.5}

    model._inference_metric_collectors = [collect]
    output = model._model_forward(torch.tensor([[1, 2]]))
    assert output.shape == (1, 2, 8)
    assert calls == [(model.model, model.model.pg_collection)]
    assert model.get_model_metrics() == [{"router_max": 0.5}]
    assert model.requires_uniform_request_groups


def _position_sensitive_forward(captured=None):
    def forward(input_ids, attention_mask=None):
        if captured is not None:
            captured.append((input_ids.clone(), attention_mask.clone()))
        logits = torch.zeros((*input_ids.shape, 8))
        logits[:, :, 3] = torch.arange(1, input_ids.shape[1] + 1)
        return logits

    return forward


def test_loglikelihood_right_padding_preserves_scores_and_mask():
    requests = [(None, [2], [3]), (None, [2, 2, 2], [3])]
    reference = _likelihood_model(batch_size=1)
    reference._model_forward = _position_sensitive_forward()
    expected = reference._loglikelihood_tokens(requests, disable_tqdm=True)

    captured = []
    batched = _likelihood_model(batch_size=2)
    batched._model_forward = _position_sensitive_forward(captured)
    actual = batched._loglikelihood_tokens(requests, disable_tqdm=True)
    assert torch.equal(captured[0][0], torch.tensor([[2, 2, 2, 3], [2, 3, 0, 0]]))
    assert torch.equal(captured[0][1], torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]))
    for got, want in zip(actual, expected, strict=True):
        assert got[0] == pytest.approx(want[0])
        assert got[1] is want[1]


def test_sequence_parallel_likelihood_right_pads_to_tp():
    captured = []
    model = _likelihood_model(batch_size=2, tp_size=2, sequence_parallel=True)
    model._model_forward = _position_sensitive_forward(captured)
    results = model._loglikelihood_tokens(
        [(None, [2], [3]), (None, [2, 2], [3])], disable_tqdm=True
    )
    assert all(greedy for _, greedy in results)
    assert torch.equal(captured[0][0], torch.tensor([[2, 2, 3, 0], [2, 3, 0, 0]]))
    assert torch.equal(captured[0][1], torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]))


def test_sequence_parallel_generation_padding_stays_left():
    model = _likelihood_model(batch_size=1, tp_size=2, sequence_parallel=True)
    ids, mask = model._pad_for_sequence_parallel(
        torch.tensor([[2, 3, 4]]), torch.ones((1, 3), dtype=torch.long)
    )
    assert torch.equal(ids, torch.tensor([[0, 2, 3, 4]]))
    assert torch.equal(mask, torch.tensor([[0, 1, 1, 1]]))


class _SamplingParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _NativeTokenizer:
    bos = 1
    eod = 0

    def tokenize(self, text):
        return {"prompt": [1, 2], "truncated": [1, 3]}[text]

    def detokenize(self, tokens):
        return {(4, 5, 6): "answer<stop>ignored", (1, 3): "truncated"}[tuple(tokens)]


class _FakeEngine:
    def __init__(self, result, max_sequence_length=8):
        self.result = result
        self.context = SimpleNamespace(max_sequence_length=max_sequence_length)
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return [SimpleNamespace(merge=lambda: self.result) for _ in kwargs["prompts"]]


def _install_sampling_params(monkeypatch):
    module = ModuleType("megatron.core.inference.sampling_params")
    module.SamplingParams = _SamplingParams
    monkeypatch.setitem(sys.modules, module.__name__, module)


def _native_model(result):
    model = MegatronLMEval.__new__(MegatronLMEval)
    LM.__init__(model)
    model.tokenizer = _NativeTokenizer()
    model._native_generation_engine_type = "dynamic"
    model._native_generation_engine = _FakeEngine(result)
    return model


def test_native_generation_maps_greedy_sampling_and_stop_words(monkeypatch):
    _install_sampling_params(monkeypatch)
    model = _native_model(SimpleNamespace(generated_tokens=[4, 5, 6]))
    model._pp_size = 2
    model._tp_size = 1
    model._global_rank = 0
    model._max_length = 8
    model._max_gen_toks = 3
    model._batch_size = 1
    model._args = SimpleNamespace(sequence_parallel=False)
    model._model_forward = lambda *args, **kwargs: pytest.fail("eager forward")
    request = Instance(
        "generate_until",
        {},
        ("prompt", {"until": ["<stop>"], "max_gen_toks": 3}),
        0,
    )
    assert model.generate_until([request]) == ["answer"]
    params = model._native_generation_engine.calls[0]["sampling_params"]
    assert (params.temperature, params.top_k, params.top_p) == (1.0, 1, 0.0)
    assert params.stop_words == ["<stop>"]


def test_native_generation_rejects_negative_temperature(monkeypatch):
    _install_sampling_params(monkeypatch)
    model = _native_model(SimpleNamespace(generated_tokens=[]))
    with pytest.raises(ValueError, match="non-negative"):
        model._native_generate([[1, 2]], [], 1, -1.0, 1.0, 0)


def test_native_likelihood_scores_selected_prompt_continuation(monkeypatch):
    _install_sampling_params(monkeypatch)
    result = SimpleNamespace(
        prompt_log_probs=[-0.1, -0.2, -0.3],
        prompt_top_n_logprobs=[{"a": -0.1}, {"b": -0.2}, {"c": -0.1}],
    )
    model = _native_model(result)
    assert model._native_loglikelihood([[1, 2, 3, 4]], [2], [2]) == [
        pytest.approx((-0.5, False))
    ]
    params = model._native_generation_engine.calls[0]["sampling_params"]
    assert (params.num_tokens_to_generate, params.return_log_probs) == (1, True)
    assert not params.skip_prompt_log_probs
    assert params.top_n_logprobs == 1


def test_loglikelihood_uses_native_engine_without_eager_forward(monkeypatch):
    _install_sampling_params(monkeypatch)
    model = _native_model(None)
    model._use_inference_engine_for_likelihood = True
    # The model context is shorter than the engine context: the complete
    # four-token prompt must still be passed to prefill scoring.
    model._max_length = 4
    model._batch_size = 2
    model._global_rank = 0
    model._tp_size = 1
    model._pp_size = 2
    model._args = SimpleNamespace(sequence_parallel=False)
    model._model_forward = lambda *args, **kwargs: pytest.fail("eager forward")
    results = {
        (1, 2, 3, 4): SimpleNamespace(
            prompt_log_probs=[-0.1, -0.2, -0.3],
            prompt_top_n_logprobs=[{"a": -0.1}, {"b": -0.2}, {"c": -0.3}],
        ),
        (1, 2): SimpleNamespace(
            prompt_log_probs=[-0.7], prompt_top_n_logprobs=[{"a": -0.7}]
        ),
    }

    def generate(**kwargs):
        model._native_generation_engine.calls.append(kwargs)
        return [
            SimpleNamespace(merge=lambda result=results[tuple(prompt)]: result)
            for prompt in kwargs["prompts"]
        ]

    model._native_generation_engine.generate = generate
    got = model._loglikelihood_tokens(
        [(None, [1], [2]), (None, [1, 2], [3, 4])], disable_tqdm=True
    )
    assert got[0] == pytest.approx((-0.7, True))
    assert got[1] == pytest.approx((-0.5, True))
    assert model._native_generation_engine.calls[0]["prompts"] == [
        [1, 2, 3, 4],
        [1, 2],
    ]


def test_native_likelihood_reserves_decode_slot(monkeypatch):
    _install_sampling_params(monkeypatch)
    result = SimpleNamespace(
        prompt_log_probs=[-0.1, -0.2, -0.3],
        prompt_top_n_logprobs=[{"a": -0.1}, {"b": -0.2}, {"c": -0.3}],
    )
    model = _native_model(result)
    model._use_inference_engine_for_likelihood = True
    model._max_length = 8
    model._batch_size = 1
    model._global_rank = 1
    model._tp_size = 1
    model._args = SimpleNamespace(sequence_parallel=False)
    model._native_generation_engine.context.max_sequence_length = 5
    assert model._loglikelihood_tokens(
        [(None, [0, 1, 2, 3], [4, 5])], disable_tqdm=True
    ) == [pytest.approx((-0.5, True))]
    assert model._native_generation_engine.calls[0]["prompts"] == [[2, 3, 4, 5]]


def test_native_engine_rejects_unsupported_options():
    model = MegatronLMEval.__new__(MegatronLMEval)
    model._args = SimpleNamespace(
        inference_dynamic_batching=False, use_legacy_static_engine=True
    )
    with pytest.raises(NotImplementedError, match=r"only.*dynamic"):
        model._initialize_native_generation_engine()

    model._args = SimpleNamespace(
        inference_dynamic_batching=False, use_legacy_static_engine=False
    )
    model._use_inference_engine_for_likelihood = True
    with pytest.raises(ValueError, match="requires --inference-dynamic-batching"):
        model._initialize_native_generation_engine()


def test_pipeline_parallelism_requires_native_engine():
    model = _bare_model()
    model._pp_size = 2
    model._args = SimpleNamespace(
        inference_dynamic_batching=False, use_legacy_static_engine=False
    )
    model._use_inference_engine_for_likelihood = False
    with pytest.raises(NotImplementedError, match="Pipeline parallelism requires"):
        model._initialize_native_generation_engine()

    with pytest.raises(NotImplementedError, match="Eager forward does not support"):
        model._model_forward(torch.tensor([[1, 2]]))
    with pytest.raises(NotImplementedError, match="likelihood requires"):
        model._loglikelihood_tokens([], disable_tqdm=True)


def test_native_engine_rejects_unverified_parallelism():
    for sequence_parallel, context_parallel_size in ((True, 1), (False, 2)):
        model = MegatronLMEval.__new__(MegatronLMEval)
        model._args = SimpleNamespace(
            inference_dynamic_batching=True,
            use_legacy_static_engine=False,
            sequence_parallel=sequence_parallel,
            context_parallel_size=context_parallel_size,
        )
        with pytest.raises(NotImplementedError, match="has not been verified"):
            model._initialize_native_generation_engine()


def test_native_engine_rejects_metric_collectors():
    model = MegatronLMEval.__new__(MegatronLMEval)
    model._args = SimpleNamespace(
        inference_dynamic_batching=True,
        use_legacy_static_engine=False,
        sequence_parallel=False,
        context_parallel_size=1,
    )
    model._inference_metric_collectors = [object()]
    with pytest.raises(ValueError, match="cannot be combined"):
        model._initialize_native_generation_engine()


class _GatherCounts:
    def __init__(self, counts, remote_example=None):
        self.counts = counts
        self.remote_example = remote_example

    def gather(self, local_count):
        assert local_count.item() == self.counts[0]
        return torch.tensor(self.counts, device=local_count.device)

    def gather_object(self, local_example):
        return [local_example, self.remote_example]


class _RollingModel(MegatronLMEval):
    @property
    def accelerator(self):
        return self._test_accelerator


def _rolling_model(counts, remote_example=None):
    model = _RollingModel.__new__(_RollingModel)
    LM.__init__(model)
    model._global_rank = model._rank = 0
    model._world_size = len(counts)
    model._tp_size = 1
    model._args = SimpleNamespace(sequence_parallel=False)
    model._device = torch.device("cpu")
    model._max_length = 4
    model._test_accelerator = _GatherCounts(counts, remote_example)
    model.tokenizer = SimpleNamespace(eod=0)
    model.tok_encode = lambda text: list(range(len(text)))
    return model


def test_rolling_likelihood_flattens_and_pads_windows(monkeypatch):
    model = _rolling_model([3, 5])
    calls, cached = [], []
    model.cache_hook.add_partial = lambda *args: cached.append(args)
    monkeypatch.setattr(
        megatron_lm,
        "get_rolling_token_windows",
        lambda token_list, **kwargs: [([0], [1])] * len(token_list),
    )
    monkeypatch.setattr(megatron_lm, "make_disjoint_window", lambda window: window)
    model._loglikelihood_tokens = lambda windows, disable_tqdm=False: (
        calls.append((windows, disable_tqdm)) or [(1.0, False)] * len(windows)
    )
    requests = [
        Instance("loglikelihood_rolling", {}, ("ab",), 0),
        Instance("loglikelihood_rolling", {}, ("c",), 1),
    ]
    assert model.loglikelihood_rolling(requests) == [2.0, 1.0]
    assert len(calls) == 1 and len(calls[0][0]) == 5
    assert [entry[1] for entry in cached] == [("ab",), ("c",)]


def test_rolling_likelihood_pads_empty_local_shard(monkeypatch):
    remote = (None, [9], [10])
    model = _rolling_model([0, 2], remote)
    scored = []
    model._loglikelihood_tokens = lambda windows, disable_tqdm=False: (
        scored.extend(windows) or [(0.0, False)] * len(windows)
    )
    monkeypatch.setattr(
        megatron_lm, "get_rolling_token_windows", lambda token_list, **kwargs: []
    )
    assert model.loglikelihood_rolling([]) == []
    assert scored == [remote, remote]
