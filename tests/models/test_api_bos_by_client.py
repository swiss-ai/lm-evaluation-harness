"""bos_by_client: the server is told not to add special tokens, so the BOS the model gets is
exactly the one the client wrote (API models that send text prompts)."""

import pytest

pytest.importorskip("tokenizers")
pytest.importorskip("transformers")

from tokenizers import Tokenizer, models, pre_tokenizers, processors  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

from lm_eval.models.openai_completions import LocalCompletionsAPI  # noqa: E402


WORDS = ["<s>", "<unk>", "The", "capital", "of", "France", "is", "Paris", "[T]", "hi"]


def _tokenizer(tmp_path, name, adds_bos: bool, with_template: bool):
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    if adds_bos:
        tok.post_processor = processors.TemplateProcessing(
            single="<s> $A", special_tokens=[("<s>", 0)]
        )
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<s>", unk_token="<unk>")
    if with_template:
        # a chat template that writes the BOS itself (Apertus, Llama, Gemma 3, Mistral)
        fast.chat_template = "{{ bos_token }}[T] {% for m in messages %}{{ m['content'] }}{% endfor %}"
    path = tmp_path / name
    fast.save_pretrained(path)
    return str(path)


def _model(path, **kwargs):
    return LocalCompletionsAPI(
        model="m",
        base_url="http://x/v1/completions",
        tokenizer_backend="huggingface",
        tokenizer=path,
        tokenized_requests=False,
        **kwargs,
    )


def _server_bos(path, text, add_special_tokens):
    """BOS tokens the server's tokenizer (the same files) leaves in front of the text."""
    fast = PreTrainedTokenizerFast.from_pretrained(path)
    ids = fast(text, add_special_tokens=add_special_tokens).input_ids
    n = 0
    for i in ids:
        if i != fast.bos_token_id:
            break
        n += 1
    return n


def _loglik_text(model, context, continuation):
    context_enc, continuation_enc = model._encode_pair(context, continuation)
    inputs, _, _ = model.batch_loglikelihood_requests([[(None, context_enc, continuation_enc)]])
    return model.decode_batch(inputs)[0]


def test_payloads_tell_the_server_not_to_add_special_tokens(tmp_path):
    path = _tokenizer(tmp_path, "adds", adds_bos=True, with_template=False)
    on, off = _model(path, bos_by_client=True), _model(path)
    assert on._create_payload("p", generate=False)["add_special_tokens"] is False
    assert on._create_payload("p", generate=True, gen_kwargs={})["add_special_tokens"] is False
    assert "add_special_tokens" not in off._create_payload("p", generate=False)
    assert "add_special_tokens" not in off._create_payload("p", generate=True, gen_kwargs={})


def test_a_tokenizer_that_adds_bos_itself_counts_as_add_bos_token(tmp_path):
    adds = _tokenizer(tmp_path, "adds", adds_bos=True, with_template=False)
    never = _tokenizer(tmp_path, "never", adds_bos=False, with_template=False)
    assert _model(adds, bos_by_client=True).add_bos_token is True
    assert _model(never, bos_by_client=True).add_bos_token is False
    assert _model(adds).add_bos_token is False  # unchanged without the option


@pytest.mark.parametrize("adds_bos", [True, False])
def test_no_template_one_bos_reaches_the_server_when_asked_for(tmp_path, adds_bos):
    path = _tokenizer(tmp_path, "t", adds_bos=adds_bos, with_template=False)
    # --bos: forced on, also for a tokenizer that adds BOS itself: no double any more
    model = _model(path, bos_by_client=True, add_bos_token=True)
    loglik = _loglik_text(model, "The capital of France is", " Paris")
    generation = model._with_bos_text("The capital of France is")
    assert _server_bos(path, loglik, add_special_tokens=False) == 1
    assert _server_bos(path, generation, add_special_tokens=False) == 1


def test_no_template_a_tokenizer_that_never_adds_bos_gets_none_unless_asked(tmp_path):
    path = _tokenizer(tmp_path, "never", adds_bos=False, with_template=False)
    model = _model(path, bos_by_client=True)
    assert _server_bos(path, _loglik_text(model, "The capital", " Paris"), add_special_tokens=False) == 0


def test_a_tokenizer_that_adds_bos_still_delivers_one_without_the_server(tmp_path):
    # before: the server added it; now the client writes it
    path = _tokenizer(tmp_path, "adds", adds_bos=True, with_template=False)
    model = _model(path, bos_by_client=True)  # add_bos_token resolves to True
    loglik = _loglik_text(model, "The capital of France is", " Paris")
    generation = model._with_bos_text("The capital of France is")
    assert _server_bos(path, loglik, add_special_tokens=False) == 1
    assert _server_bos(path, generation, add_special_tokens=False) == 1


@pytest.mark.parametrize("adds_bos", [True, False])
def test_template_that_writes_bos_reaches_the_server_once(tmp_path, adds_bos):
    path = _tokenizer(tmp_path, "t", adds_bos=adds_bos, with_template=True)
    model = _model(path, bos_by_client=True)
    rendered = model.apply_chat_template([{"role": "user", "content": "hi"}])
    assert rendered.startswith("<s>")
    loglik = _loglik_text(model, rendered, " Paris")
    generation = model._with_bos_text(rendered)
    assert _server_bos(path, loglik, add_special_tokens=False) == 1
    assert _server_bos(path, generation, add_special_tokens=False) == 1


def test_without_the_option_a_template_that_writes_bos_is_doubled_by_the_server(tmp_path):
    # the behaviour bos_by_client replaces: the template's <s> plus the one the server adds
    path = _tokenizer(tmp_path, "adds", adds_bos=True, with_template=True)
    model = _model(path)
    rendered = model.apply_chat_template([{"role": "user", "content": "hi"}])
    assert _server_bos(path, rendered, add_special_tokens=True) == 2
