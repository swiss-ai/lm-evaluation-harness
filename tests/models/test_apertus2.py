"""apertus2-completions: client-side Apertus 2 rendering and parsing.

A synthetic tokenizer (controls at IDs 0-5, BOS 8) keeps the tests offline and
proves that no control ID is hard-coded; set APERTUS2_TOKENIZER to the real
Apertus_2_instruct directory to run the real-artifact checks too.
"""

import json
import os
import sys
from unittest.mock import patch

import pytest


if sys.version_info < (3, 12):
    pytest.skip("apertus-common needs Python >= 3.12", allow_module_level=True)
lib = pytest.importorskip("apertus_common")

from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from lm_eval.api.instance import Instance
from lm_eval.api.task import ConfigurableTask
from lm_eval.evaluator import evaluate
from lm_eval.models.apertus2 import _REQUEST as REQUEST, Apertus2Completions
from lm_eval.models.api_models import JsonChatStr


PREFIX = "The best answer is"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]
CALL_HISTORY = [
    {"role": "user", "content": "Weather in Bern?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "apertus_0",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city":"Bern"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "apertus_0", "content": "sunny"},
]
QUESTION = [{"role": "user", "content": "Q?"}]
PREFILLED = [*QUESTION, {"role": "assistant", "content": PREFIX}]


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    path = tmp_path_factory.mktemp("apertus2_tokenizer")
    controls = ["<|in|>", "<|/in|>", "<|out|>", "<|/out|>", "<|hdr|>", "<|wait|>"]
    specials = [*controls, "<|pad|>", "<|other|>", "<s>"]
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {token: index for index, token in enumerate([*specials, *alphabet])}
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=False
    )
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(
        [AddedToken(token, normalized=False, special=True) for token in specials]
    )
    roles = {
        "input_start_token": "<|in|>",
        "input_end_token": "<|/in|>",
        "output_start_token": "<|out|>",
        "output_end_token": "<|/out|>",
        "header_end_token": "<|hdr|>",
        "wait_token": "<|wait|>",
    }
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        extra_special_tokens=roles,
        pad_token="<|pad|>",  # noqa: S106
        bos_token="<s>",  # noqa: S106
        clean_up_tokenization_spaces=False,
    ).save_pretrained(path)
    return str(path)


@pytest.fixture(params=["synthetic", "real"])
def artifact(request, synthetic):
    if request.param == "synthetic":
        return synthetic
    real = os.environ.get("APERTUS2_TOKENIZER")
    if not real:
        pytest.skip("set APERTUS2_TOKENIZER for real-artifact checks")
    return real


def make(artifact, **kwargs):
    kwargs.setdefault("tokenizer", artifact)
    return Apertus2Completions(
        base_url="http://test/v1/completions", model="m", **kwargs
    )


def chat_request(messages, **gen_kwargs):
    return Instance(
        request_type="generate_until",
        doc={},
        arguments=(JsonChatStr(json.dumps(messages)), {"until": [], **gen_kwargs}),
        idx=0,
    )


def native(enc, *items):
    return enc.encode_items(list(items))


def completion(enc, ids, *, stop=None, length=False, with_ids=True, prompt_ids=None):
    """A vLLM completion: the stop token is in token_ids, not in text."""
    choice = {
        "index": 0,
        "text": enc.decode_text(list(ids)),
        "finish_reason": "length" if length else "stop",
        "stop_reason": stop,
    }
    if with_ids:
        choice["token_ids"] = list(ids) + ([stop] if stop is not None else [])
    if prompt_ids is not None:
        choice["prompt_token_ids"] = prompt_ids
    return {"choices": [choice]}


def answer(enc, *items, **kwargs):
    """A finished turn ending at the wait token."""
    return completion(enc, native(enc, *items), stop=enc.controls.wait, **kwargs)


def phase_one(enc, think="B fits.", with_ids=True):
    """After-think phase one, stopped at the think's close."""
    ids = native(enc, lib.Think(payload=think))[:-1]
    return completion(enc, ids, stop=enc.controls.output_end, with_ids=with_ids)


def server_len(artifact, prompt):
    server = PreTrainedTokenizerFast.from_pretrained(artifact)
    return len(server(prompt, add_special_tokens=False).input_ids)


def run(model, instances, responses):
    """Generate with a fake server answering each POST from ``responses``."""
    sent = []
    replies = iter(responses)

    def post(payload):
        sent.append(payload)
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    with patch.object(model, "_post", side_effect=post):
        result = model.generate_until(instances)
    return result, sent


# Setup


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"tokenizer": None}, "needs tokenizer"),
        ({"tokenizer_backend": "huggingface"}, "tokenizer_backend"),
        ({"think_end_token": "</think>"}, "think_start_token"),
        ({"prefill": "drop"}, "prefill"),
        ({"chat_template_args": {"enable_thinking": False}}, "cannot be disabled"),
        ({"chat_template_args": {"foo": 1}}, "unsupported chat_template_args"),
        ({"reasoning_effort": "extreme"}, "extreme"),
    ],
)
def test_setup_rejects_invalid_configuration(synthetic, kwargs, match):
    with pytest.raises(ValueError, match=match):
        make(synthetic, **kwargs)


# Request path


def test_prompt_matches_native_encoding(artifact):
    model = make(
        artifact,
        mask_special_tokens=True,
        reasoning_effort="low",
        chat_template_args={"reasoning_effort": "high"},
    )
    messages = [
        {"role": "system", "content": "Answer the math question."},
        {"role": "user", "content": "Question: 2+2?\nAnswer:"},
        {"role": "assistant", "content": "The answer is 4."},
        {"role": "user", "content": "Question: 3+3?\nAnswer:"},
    ]
    payload = model._create_payload(
        messages, gen_kwargs={"until": ["Question:"], "max_gen_toks": 64, "top_p": 0.9}
    )
    request = payload.pop(REQUEST)
    expected = lib.from_openai(
        messages,
        reasoning_effort="low",
        policy=lib.MappingPolicy(merge_system_messages=True),
    ).value
    enc = model.encoding
    assert payload["prompt"] == model._bos_text + lib.render_str(expected)
    assert request.prompt_ids == enc.encode_conversation(expected)
    assert payload["max_tokens"] == 64 and payload["top_p"] == 0.9
    assert payload["stop_token_ids"] == [enc.controls.wait]
    assert payload["skip_special_tokens"] is False
    assert payload["add_special_tokens"] is False
    assert payload["return_token_ids"] is True
    assert set(payload["logit_bias"]) == {
        str(i) for i in enc.generation_constraints.forbidden_token_ids
    }
    assert "until" not in payload
    assert request.until == ["Question:"]


def test_server_tokenizes_the_prompt_to_native_ids():
    real = os.environ.get("APERTUS2_TOKENIZER")
    if not real:
        pytest.skip("set APERTUS2_TOKENIZER for real-artifact checks")
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Café ☕ 東京 \n  spaced  "},
        {"role": "assistant", "content": "Fine."},
        {"role": "user", "content": "Q?"},
        {"role": "assistant", "content": PREFIX},
    ]
    for prefill in ("after_think", "empty_think"):
        payload = make(real, prefill=prefill)._create_payload(messages, gen_kwargs={})
        server = PreTrainedTokenizerFast.from_pretrained(real)
        server_ids = server(payload["prompt"], add_special_tokens=False).input_ids
        assert server_ids == payload[REQUEST].prompt_ids


@pytest.mark.parametrize("num_concurrent", [1, 2])
def test_unmappable_requests_are_skipped_per_sample(synthetic, num_concurrent):
    model = make(synthetic, num_concurrent=num_concurrent)
    bad_tool = chat_request([{"role": "tool", "tool_call_id": "x", "content": "x"}])
    bad_prefix = chat_request(
        [*QUESTION, {"role": "assistant", "content": [{"type": "image"}]}]
    )
    with (
        patch.object(model, "_post") as post,
        patch.object(model, "_apost") as apost,
    ):
        result = model.generate_until([bad_tool, bad_prefix])
    assert result == ["", ""]
    assert not post.called and not apost.called
    for instance in (bad_tool, bad_prefix):
        assert instance.length_info[0]["generation_request_failed"] == 1
        assert instance.length_info[0]["generation_parse_failed"] == 0
        assert instance.length_info[0]["apertus2_prompt"] == ""


def test_vllm_style_history_is_normalized(synthetic):
    call = {**CALL_HISTORY[1]["tool_calls"][0], "id": "apertus_3"}
    history = [
        {"role": "user", "content": "Weather?", "name": "alice"},
        {
            "role": "assistant",
            "content": "",
            "reasoning": "Need the tool.",
            "refusal": None,
            "tool_calls": [call],
        },
        {**CALL_HISTORY[2], "tool_call_id": "apertus_3"},
    ]
    payload = make(synthetic)._create_payload(
        history, gen_kwargs={"tools": json.dumps(TOOLS)}
    )
    prompt = payload["prompt"]
    assert "<|out|>think<|hdr|>Need the tool.<|/out|>" in prompt
    assert '<|out|>call{"name":"get_weather","counter":3}' in prompt
    assert '<|in|>result{"name":"get_weather","counter":3}' in prompt
    assert payload[REQUEST].info["generation_request_failed"] == 0


def test_budget_is_clamped_or_skipped(synthetic):
    model = make(synthetic, max_model_len=80)
    clamped = chat_request(QUESTION, max_gen_toks=1000)
    _, sent = run(model, [clamped], [answer(model.encoding, lib.Reply(payload="A"))])
    assert server_len(synthetic, sent[0]["prompt"]) + sent[0]["max_tokens"] == 80
    assert clamped.length_info[0]["generation_budget_clamped"] == 1
    skipped = chat_request(QUESTION)
    result, sent = run(make(synthetic, max_model_len=5), [skipped], [])
    assert result == [""] and sent == []
    assert skipped.length_info[0]["generation_prompt_too_long"] == 1


def test_configuration_errors_fail_before_requests(synthetic):
    with pytest.raises(TypeError, match="apply_chat_template"):
        make(synthetic).generate_until(
            [Instance("generate_until", {}, ("plain text", {"until": []}), 0)]
        )
    with pytest.raises(ValueError, match="prefill=reject"):
        make(synthetic, prefill="reject").generate_until([chat_request(PREFILLED)])
    literal = chat_request([{"role": "user", "content": "say <|wait|>"}])
    with pytest.raises(ValueError, match="literal_special_tokens"):
        make(synthetic, literal_special_tokens="reject").generate_until([literal])


def test_literal_special_tokens_are_counted(synthetic):
    model = make(synthetic, literal_special_tokens="reject")
    enc = model.encoding
    # Assistant turns may hold model output, so reject only counts them.
    history = [
        {"role": "user", "content": "Q?"},
        {"role": "assistant", "content": "I wrote <|wait|>"},
        {"role": "user", "content": "Again?"},
    ]
    instance = chat_request(history)
    run(model, [instance], [answer(enc, lib.Reply(payload="ok"))])
    assert instance.length_info[0]["generation_literal_special_tokens"] == 1


# Response path


def test_reply_after_thinking_with_until_and_metrics(synthetic):
    model = make(synthetic)
    enc = model.encoding
    think = "48/2 = 24. Question: unrelated"
    instance = chat_request(QUESTION, until=["Question:"])
    reply = answer(
        enc,
        lib.Think(payload=think),
        lib.Reply(payload="The answer is 72.\n\nQuestion: next"),
    )
    result, sent = run(model, [instance], [reply])
    assert result == ["The answer is 72.\n\n"]
    assert REQUEST not in sent[0]
    json.dumps(sent[0])  # the sent payload is plain JSON
    info = instance.length_info[0]
    assert info["thinking_format_correct"] == 1
    assert info["generation_truncated"] == 0
    assert info["thinking_length_chars"] == len(think)
    assert info["response_length_words"] == 5 + 6
    assert info["apertus2_output"].startswith("<|out|>think<|hdr|>48/2")
    assert len(info["apertus2_prompt"]) == 16
    assert all(info[key] == 0 for key in info if key.startswith("generation_"))


def test_prompt_mismatch_reveals_a_wrong_server_tokenizer(synthetic):
    model = make(synthetic)
    enc = model.encoding
    good, bad = chat_request(QUESTION), chat_request(QUESTION)
    ids = model._create_payload(QUESTION, gen_kwargs={})[REQUEST].prompt_ids
    responses = [
        answer(enc, lib.Reply(payload="A"), prompt_ids=ids),
        answer(enc, lib.Reply(payload="A"), prompt_ids=ids[:-1]),
    ]
    unchecked = chat_request(QUESTION)
    responses.append(answer(enc, lib.Reply(payload="A")))
    run(model, [good, bad, unchecked], responses)
    assert good.length_info[0]["generation_prompt_mismatch"] == 0
    assert bad.length_info[0]["generation_prompt_mismatch"] == 1
    assert "generation_prompt_mismatch" not in unchecked.length_info[0]


def test_output_cut_inside_thinking_scores_empty(synthetic):
    model = make(synthetic)
    enc = model.encoding
    instance = chat_request(QUESTION)
    ids = native(enc, lib.Think(payload="still thinking", open=True))
    result, _ = run(model, [instance], [completion(enc, ids, length=True)])
    assert result == [""]
    info = instance.length_info[0]
    assert info["generation_think_truncated"] == 1
    assert info["generation_truncated"] == 1
    assert info["thinking_format_has_close"] == 0


def test_output_cut_inside_reply_mid_utf8_character(artifact):
    model = make(artifact)
    enc = model.encoding
    ids = native(
        enc, lib.Think(payload="t"), lib.Reply(payload="The answer is 72", open=True)
    )
    ids += enc.encode_text(" 🚀")[:-1]
    instance = chat_request(QUESTION)
    result, _ = run(model, [instance], [completion(enc, ids, length=True)])
    assert result[0].rstrip() == "The answer is 72"
    assert instance.length_info[0]["generation_parse_failed"] == 0


def test_trailing_eos_and_bad_responses(synthetic):
    model = make(synthetic)
    enc = model.encoding
    other = enc.token_to_id("<|other|>")
    good = completion(enc, native(enc, lib.Reply(payload="A")), stop=other)
    responses = [
        good,
        {"choices": [{"index": 0, "text": "<|out|>reply<|out|>"}]},
        {"choices": [{"index": 0, "token_ids": [10**9], "finish_reason": "stop"}]},
        {"choices": [{"index": 0, "token_ids": [[1]], "finish_reason": "stop"}]},
        {"choices": [{"index": 0, "finish_reason": "stop"}]},
        {"object": "error"},
        None,
    ]
    instances = [chat_request(QUESTION) for _ in responses]
    result, _ = run(model, instances, responses)
    assert result == ["A", "", "", "", "", "", ""]
    failed = [i.length_info[0]["generation_parse_failed"] for i in instances]
    assert failed == [0, 1, 1, 1, 1, 1, 1]


def test_thinking_lengths_need_well_formed_output(synthetic):
    model = make(synthetic)
    enc = model.encoding
    instance = chat_request(QUESTION)
    ids = native(enc, lib.Think(payload="t"), lib.Reply(payload="A"))
    ids += [enc.controls.output_start, enc.controls.output_start]
    result, _ = run(model, [instance], [completion(enc, ids, stop=enc.controls.wait)])
    assert result == ["A"]  # messages completed before the error are kept
    info = instance.length_info[0]
    assert info["generation_parse_failed"] == 1
    assert info["thinking_format_correct"] == 0
    assert "thinking_length_chars" not in info


def test_tool_calls_follow_the_marker_for_bfcl(synthetic):
    from lm_eval.tasks.bfcl_v3.task import _fc_calls, _split_tool_calls

    model = make(synthetic)
    enc = model.encoding
    call = lib.Call(name="get_weather", counter=0, payload='{"city":"Bern"}')
    with_tools = chat_request(QUESTION, tools=json.dumps(TOOLS))
    without = chat_request(QUESTION)
    reply = answer(enc, lib.Think(payload="t"), call)
    result, _ = run(model, [with_tools, without], [reply, reply])
    content, calls = _split_tool_calls(result[0])
    assert content == "" and calls[0]["id"] == "apertus_0"
    functions = [{"name": "get_weather"}]
    assert _fc_calls(calls, functions) == [{"get_weather": {"city": "Bern"}}]
    assert result[1] == ""
    assert without.length_info[0]["generation_unexpected_call"] == 1


def test_open_reused_and_unknown_items_are_dropped(synthetic):
    model = make(synthetic)
    enc = model.encoding
    tools = json.dumps(TOOLS)
    reused, open_call = (chat_request(CALL_HISTORY, tools=tools) for _ in range(2))
    odd = chat_request(QUESTION)
    responses = [
        answer(enc, lib.Call(name="get_weather", counter=0, payload="{}")),
        completion(
            enc,
            native(
                enc,
                lib.Reply(payload="ok"),
                lib.Call(name="get_weather", counter=1, payload='{"ci', open=True),
            ),
            length=True,
        ),
        answer(
            enc,
            lib.Claim(payload="4"),
            lib.UnknownOutput(output_type="memo", payload="x"),
            lib.Reply(payload="4"),
        ),
    ]
    result, _ = run(model, [reused, open_call, odd], responses)
    assert result == ["", "ok", "4"]
    assert reused.length_info[0]["generation_call_counter_reused"] == 1
    assert odd.length_info[0]["generation_has_claim"] == 1
    assert odd.length_info[0]["generation_unknown_output"] == 1


# Prefill


@pytest.mark.parametrize("with_ids", [True, False])
def test_after_think_prefill_continues_the_reply(synthetic, with_ids):
    model = make(synthetic, return_token_ids=with_ids)
    enc = model.encoding
    instance = chat_request(PREFILLED, until=["."], max_gen_toks=200)
    first = phase_one(enc, with_ids=with_ids)
    first["usage"] = {"completion_tokens": 150}
    second = enc.encode_text(" B.") + [enc.controls.output_end]
    responses = [
        first,
        completion(enc, second, stop=enc.controls.wait, with_ids=with_ids),
    ]
    result, sent = run(model, [instance], responses)
    assert result == [" B"]
    assert sent[0]["prompt"].endswith("<|in|>user<|hdr|>Q?<|/in|>")
    assert sent[0]["stop_token_ids"] == [enc.controls.output_end, enc.controls.wait]
    assert sent[1]["stop_token_ids"] == [enc.controls.wait]
    assert sent[1]["prompt"].endswith(
        f"<|out|>think<|hdr|>B fits.<|/out|><|out|>reply<|hdr|>{PREFIX}"
    )
    if with_ids:  # phase two stays within phase one's prompt + budget
        total = server_len(synthetic, sent[0]["prompt"]) + 200
        assert server_len(synthetic, sent[1]["prompt"]) + sent[1]["max_tokens"] == total
    else:  # without token IDs, the reported usage counts phase one's tokens
        assert sent[1]["max_tokens"] == 200 - 150
    info = instance.length_info[0]
    assert info["generation_prefill_skipped"] == 0
    assert info["thinking_format_correct"] == 1
    assert info["response_length_chars"] == len("B fits.") + len(" B.")


@pytest.mark.parametrize("with_ids", [True, False])
def test_after_think_prefill_skipped_cut_or_out_of_budget(synthetic, with_ids):
    model = make(synthetic)
    enc = model.encoding
    skipped, cut, malformed = (chat_request(PREFILLED) for _ in range(3))
    used = len(phase_one(enc)["choices"][0]["token_ids"])
    exhausted = chat_request(PREFILLED, max_gen_toks=used)
    responses = [
        completion(
            enc,
            native(enc, lib.Reply(payload="D"))[:-1],
            stop=enc.controls.output_end,
            with_ids=with_ids,
        ),
        completion(
            enc,
            native(enc, lib.Think(payload="t", open=True)),
            length=True,
            with_ids=with_ids,
        ),
        phase_one(enc, with_ids=with_ids),
        {},
        phase_one(enc, with_ids=with_ids),
    ]
    result, sent = run(model, [skipped, cut, malformed, exhausted], responses)
    assert result == ["D", "", "", ""] and len(sent) == 5
    assert skipped.length_info[0]["generation_prefill_skipped"] == 1
    assert malformed.length_info[0]["generation_parse_failed"] == 1
    assert cut.length_info[0]["generation_think_truncated"] == 1
    # The budget ran out after a closed think: truncated, but not in thinking.
    info = exhausted.length_info[0]
    assert info["generation_truncated"] == 1
    assert info["generation_think_truncated"] == 0
    assert info["thinking_format_has_close"] == 1


def test_failed_phase_two_retries_the_generation(synthetic):
    model = make(synthetic, max_retries=2)
    enc = model.encoding
    second = completion(
        enc, enc.encode_text(" C") + [enc.controls.output_end], stop=enc.controls.wait
    )
    responses = [phase_one(enc), ConnectionError("lost"), phase_one(enc), second]
    with patch("tenacity.nap.time.sleep"):
        result, sent = run(model, [chat_request(PREFILLED)], responses)
    assert result == [" C"] and len(sent) == 4


def test_empty_think_prefill_keeps_code_indentation(synthetic):
    model = make(synthetic, prefill="empty_think")
    enc = model.encoding
    prompt = "def f(x):\n"
    prefix = f"Here is the completed function:\n```python\n{prompt}\n"
    instance = chat_request(
        [*QUESTION, {"role": "assistant", "content": prefix}], until=["\n```"]
    )
    ids = enc.encode_text("    return x + 1\n```\nDone.") + [enc.controls.output_end]
    result, sent = run(
        model, [instance], [completion(enc, ids, stop=enc.controls.wait)]
    )
    assert len(sent) == 1 and result == ["    return x + 1"]
    assert sent[0]["prompt"].endswith(
        f"<|/in|><|out|>think<|hdr|><|/out|><|out|>reply<|hdr|>{prefix}"
    )
    assert sent[0]["stop_token_ids"] == [enc.controls.wait]
    assert instance.length_info[0]["thinking_format_has_open"] == 0


def test_async_path_runs_both_prefill_phases(synthetic):
    model = make(synthetic, num_concurrent=2)
    enc = model.encoding
    second = completion(
        enc, enc.encode_text(" C") + [enc.controls.output_end], stop=enc.controls.wait
    )
    replies = iter([phase_one(enc), second])
    sent = []

    async def fake_post(session, payload):
        sent.append(payload)
        return next(replies)

    with patch.object(model, "_apost", side_effect=fake_post):
        assert model.generate_until([chat_request(PREFILLED)]) == [" C"]
    assert len(sent) == 2 and all(REQUEST not in p for p in sent)


# End to end with in-memory tasks


class _Docs(list):
    @property
    def features(self):
        return {key: None for key in self[0]}


class _PrefillTask(ConfigurableTask):
    """arc_challenge_chat-like: system description, few-shot, answer prefill."""

    def download(self, *args, **kwargs):
        docs = [
            {"question": "2+2? A. 4 B. 5", "answer": "A"},
            {"question": "3+3? A. 5 B. 6", "answer": "B"},
            {"question": "1+1? A. 2 B. 3", "answer": "A"},
        ]
        self.dataset = {"train": _Docs(docs[:1]), "test": _Docs(docs[1:])}

    def process_results(self, doc, results):
        return {"acc": int(results[0].strip() == doc["answer"])}


def _prefill_task():
    return _PrefillTask(
        config={
            "task": "toy_prefill",
            "output_type": "generate_until",
            "training_split": "train",
            "test_split": "test",
            "fewshot_split": "train",
            "description": "Choose the right letter.",
            "doc_to_text": "{{question}}",
            "doc_to_target": "{{answer}}",
            "gen_prefix": PREFIX,
            "num_fewshot": 1,
            "generation_kwargs": {"until": ["."], "max_gen_toks": 300},
            "metadata": {"version": 1},
        }
    )


def test_evaluate_prefill_task_end_to_end(synthetic):
    model = make(synthetic)
    enc = model.encoding

    def server(payload):
        if enc.controls.output_end in payload["stop_token_ids"]:
            return phase_one(enc)
        letter = "B" if "3+3?" in payload["prompt"] else "A"
        ids = enc.encode_text(f" {letter}.") + [enc.controls.output_end]
        return completion(enc, ids, stop=enc.controls.wait)

    with patch.object(model, "_post", side_effect=server) as post:
        results = evaluate(
            lm=model,
            task_dict={"toy_prefill": _prefill_task()},
            limit=2,
            bootstrap_iters=0,
            log_samples=True,
            apply_chat_template=True,
            fewshot_as_multiturn=True,
        )
    metrics = results["results"]["toy_prefill"]
    assert metrics["acc,none"] == 1.0
    assert metrics["generation_parse_failed,none"] == 0
    assert metrics["thinking_format_correct,none"] == 1
    assert post.call_count == 4
    prompt = post.call_args_list[0].args[0]["prompt"]
    assert '"behavior":"Choose the right letter."' in prompt
    assert f"<|out|>reply<|hdr|>{PREFIX} A<|/out|><|wait|>" in prompt
    sample = results["samples"]["toy_prefill"][0]
    assert sample["length_info"][0][-1]["apertus2_prefill_mode"] == "after_think"
