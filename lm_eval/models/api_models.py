import abc
import asyncio
import copy
import itertools
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable, Iterable
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NamedTuple,
)


try:
    import requests
    from aiohttp import ClientSession, ClientTimeout, TCPConnector
    from tenacity import RetryError, retry, stop_after_attempt, wait_exponential
    from tqdm import tqdm
except ModuleNotFoundError:
    pass


import base64
from importlib.util import find_spec
from io import BytesIO

from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.api.rate_limiter import get_rate_limiter
from lm_eval.models.utils import (
    Collator,
    attach_length_info,
    build_length_info,
    chunks,
    configure_pad_token,
    detect_open_prefilled,
    postprocess_generated_text,
    resolve_think_tokens,
    resolve_track_thinking_metrics,
)


if TYPE_CHECKING:
    from PIL import Image

# Set a placeholder to use as model output when the returned answer is "None"
# This commonly occurs when the model runs out of tokens during the thinking
# process and returns an empty answer but no error.
# Default value is an empty string, but it can be edited if this is problematic.
LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER = os.environ.get(
    "LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER", ""
)


class Generation(str):
    """A generated text that remembers why the server stopped generating it.

    ``finish_reason`` is the OpenAI-style value ("stop", "length", ...) or None when
    the response didn't carry one. Behaves as a plain ``str`` everywhere else; read
    back in generate_until as the per-response ``generation_truncated`` flag.
    ``generation_info`` holds backend-specific per-response metadata that
    generate_until merges last into the response's ``length_info``.
    """

    finish_reason: str | None = None
    generation_info: dict | None = None


class PrefixedPrompt(str):
    """A rendered chat prompt whose last turn is a task ``gen_prefix`` the model continues.

    Behaves as the plain full prompt (``head`` + ``prefix``). ``head`` is the prompt
    without the prefix, ending in the generation prompt. With a thinking model the
    prefix belongs after the closed reasoning, not before it, so ``generate_until``
    sends ``head`` first and then continues from ``head + reasoning + prefix``.
    """

    head: str
    prefix: str

    def __new__(cls, full: str, head: str, prefix: str):
        obj = super().__new__(cls, full)
        obj.head = head
        obj.prefix = prefix
        return obj


def parse_finish_reasons(outputs: Any) -> list[str | None]:
    """Each choice's ``finish_reason``, in the order ``parse_generations`` returns
    the texts (by choice index, across a batch of responses). Never raises."""
    reasons: list[str | None] = []
    for out in outputs if isinstance(outputs, list) else [outputs]:
        try:
            choices = out["choices"]
            tmp: list[str | None] = [None] * len(choices)
            for pos, choice in enumerate(choices):
                tmp[choice.get("index", pos)] = choice.get("finish_reason")
        except (IndexError, KeyError, TypeError, AttributeError):
            tmp = [None]
        reasons.extend(tmp)
    return reasons


def with_finish_reasons(texts: list[str], outputs: Any) -> list[str]:
    """``texts`` as ``Generation``s carrying their choice's finish_reason; unchanged
    when the response shape doesn't line up one choice per text."""
    reasons = parse_finish_reasons(outputs)
    if len(reasons) != len(texts):
        return texts
    tagged = []
    for text, reason in zip(texts, reasons):
        gen = text if isinstance(text, Generation) else Generation(text)
        gen.finish_reason = reason
        tagged.append(gen)
    return tagged


def truncation_info(text: Any) -> dict:
    """``generation_truncated`` = 1 when the server stopped at the token budget
    (finish_reason "length"), 0 when it stopped on its own; empty when unknown."""
    reason = getattr(text, "finish_reason", None)
    if reason is None:
        return {}
    return {"generation_truncated": int(reason == "length")}


eval_logger = logging.getLogger(__name__)


def _live_prompts(model: Any, messages: list, count: int) -> list[str]:
    """The prompt text of each request in a batch, for the live view: a chat as
    "role: text" lines, a string as it is, token ids decoded."""
    if not messages:
        return []
    if isinstance(messages[0], JsonChatStr):
        chat = json.loads(messages[0].prompt)
        return [
            "\n".join(
                f"{m.get('role', '?')}: {m['content'] if isinstance(m.get('content'), str) else json.dumps(m.get('content'))}"
                for m in chat
            )
        ]
    if isinstance(messages[0], str):
        return [str(m) for m in messages]
    if (
        isinstance(messages[0], list)
        and messages[0]
        and isinstance(messages[0][0], int)
    ):
        return list(model.decode_batch(messages))
    return []


def _live_details(outputs: Any) -> tuple[str, int | None, int | None]:
    """(reasoning text, output tokens, reasoning tokens) of one API response, as far
    as the server reported them."""
    if isinstance(outputs, list):
        outputs = outputs[0] if outputs else {}
    if not isinstance(outputs, dict):
        return "", None, None
    choice = (outputs.get("choices") or [{}])[0] or {}
    message = choice.get("message") or {}
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    usage = outputs.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    return (
        reasoning if isinstance(reasoning, str) else "",
        usage.get("completion_tokens"),
        details.get("reasoning_tokens"),
    )


LogLikelihoodInputs = tuple[tuple[str, str], list[int], list[int]]


class _RequestProgress:
    """Logs how many concurrent API requests have finished, about every 5%: they're
    awaited together, so a long run would otherwise show nothing until all are done."""

    def __init__(self, label: str, total: int, steps: int = 20) -> None:
        self.label, self.total = label, total
        self.finished = self.retries = 0
        self.step = max(1, -(-total // steps))
        self.next_report = self.step
        self.started = time.monotonic()

    def retried(self) -> None:
        self.retries += 1

    def add(self, n: int) -> None:
        self.finished += n
        if self.finished < self.next_report and self.finished < self.total:
            return
        self.next_report = (self.finished // self.step + 1) * self.step
        minutes, seconds = divmod(int(time.monotonic() - self.started), 60)
        eval_logger.info(
            f"{self.label}: {self.finished}/{self.total} requests done "
            f"({minutes}m{seconds:02d}s, {self.retries} retries so far)"
        )


def _special_tokens(tokenizer) -> set[str]:
    """The strings a server's detokenizer treats as special (skip_special_tokens)."""
    tokens: set[str] = set()
    if tokenizer is None:
        return tokens
    tokens |= set(getattr(tokenizer, "all_special_tokens", None) or [])
    for token in (getattr(tokenizer, "added_tokens_decoder", None) or {}).values():
        if getattr(token, "special", False):
            tokens.add(str(token))
    return {t for t in tokens if t}


# utility class to keep track of json encoded chats
def _describe_error(e: BaseException) -> str:
    """``repr(e)`` without the request's headers.

    aiohttp's ``ClientResponseError`` repr includes its ``request_info``, headers
    among them: the ``Authorization`` bearer token would land in the logs.
    """
    request_info = getattr(e, "request_info", None)
    if request_info is None:
        return repr(e)
    return (
        f"{type(e).__name__}(status={getattr(e, 'status', None)}, "
        f"message={getattr(e, 'message', '')!r}, url={request_info.real_url})"
    )


class JsonChatStr(NamedTuple):
    prompt: str

    def encode(self, encoding):
        return self.prompt.encode(encoding)


def create_image_prompt(
    imgs: list["Image.Image"], chat: dict, fmt: str = "PNG"
) -> dict:
    """

    Parameters
    ----------
    img : list[PIL.Image.Image]
        The list of images to encode to base64
    chat : dict
    fmt : str, optional
        Any format Pillow understands (e.g. "PNG", "JPEG").
        Defaults to "PNG".

    Returns
    -------
    dict
    """
    images = []
    for img in imgs:
        buf = BytesIO()
        img.save(buf, format=fmt)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        img_dict = {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}", "detail": "auto"},
        }
        images.append(img_dict)

    # With images, we need "content" to be a list of dicts with
    # "type" and "text"/"image_url".
    # currently we do not support few-shots so only one user message
    # text content also has <image> placeholders, which apparently is not necessary for API class (confirm)

    if isinstance(chat[-1]["content"], list):
        chat[-1]["content"] = images + chat[-1]["content"]
    else:
        text_content = {"type": "text", "text": chat[-1]["content"]}
        chat[-1]["content"] = images + [text_content]
    chat[-1].pop("type", None)
    return chat


def rollout_seed(seed: int, repeat_index: int) -> int:
    """Offset a seed by the repeat index within the signed 64-bit API range."""
    return (seed + repeat_index + 2**63) % 2**64 - 2**63


class TemplateAPI(TemplateLM):
    repeat_seed_support = "provider-dependent"
    supports_repeat_scoring = True
    cache_generation = True

    MULTIMODAL = True

    def __init__(
        self,
        model: str | None = None,
        pretrained: str
        | None = None,  # `model` takes precedence over `pretrained` when passed.
        base_url: str | None = None,
        tokenizer: str | None = None,
        # Loglikelihood tasks require a tokenizer to calculate context lengths,
        # however the requests can be sent as a string if the API doesn't support token inputs.
        # use tokenized_requests=False
        tokenizer_backend: Literal["tiktoken", "huggingface", "remote", "None", "none"]
        | None = "huggingface",
        truncate: bool = False,
        # number of concurrent requests. More useful if not batching
        num_concurrent: int = 1,
        # maximum HTTP request starts per minute; unset disables rate limiting
        requests_per_minute: float | None = None,
        max_retries: int = 3,
        max_gen_toks: int = 256,
        batch_size: str | int = 1,
        seed: int = 1234,
        max_length: int | None = 2048,
        # lower each generation request's max_tokens to what the server's context
        # (max_length) leaves after the prompt; off: the request is sent as it is
        clamp_gen_toks: bool = False,
        add_bos_token: bool = False,
        # The server is told not to add special tokens (add_special_tokens=false, for APIs
        # that tokenize text prompts themselves), so the BOS the model gets is exactly the
        # one in the text sent: written here, once. A tokenizer that adds BOS itself then
        # counts as add_bos_token=True, and a BOS the chat template already wrote is kept
        # as the only one. Without it the server's tokenizer adds its own BOS next to
        # whatever the client wrote: two after a template that starts with <s>.
        bos_by_client: bool = False,
        custom_prefix_token_id: int | None = None,
        # send the requests as tokens or strings
        tokenized_requests: bool = True,
        trust_remote_code: bool = False,
        revision: str | None = "main",
        use_fast_tokenizer: bool = True,
        verify_certificate: bool = True,
        ca_cert_path: str | None = None,
        auth_token: str | None = None,
        eos_string: str | None = None,
        # timeout in seconds
        timeout: int = 300,
        header: dict[str, str] | None = None,
        max_images: int = 1,
        # extra keyword arguments for the chat template, e.g. {"reasoning_effort": "high"};
        # only used where the template is rendered client-side with an HF tokenizer
        chat_template_args: dict[str, Any] | None = None,
        # chat-template switch, folded into chat_template_args as for hf; None keeps the
        # template default
        enable_thinking: bool | None = None,
        # Reasoning handling, as on vllm/sglang: with a close token known, task stop
        # sequences are held back while the model reasons, the reasoning is stripped
        # before scoring and the thinking metrics are recorded. The open token feeds the
        # format metric. Auto-detected from the chat template only with
        # `autodetect_think_tokens` (needs the HF tokenizer).
        think_end_token: str | None = None,
        think_start_token: str | None = None,
        autodetect_think_tokens: bool = False,
        # force the thinking metrics on/off; None = on iff a close token is known
        track_thinking_metrics: bool | None = None,
        # With a reasoning close known, a task's gen_prefix (e.g. "The best answer is")
        # is placed after the closed reasoning (a second request) instead of before it,
        # where it would stop the model thinking. Needs the chat template rendered
        # client-side; False restores the prefix-first behaviour.
        after_think_prefix: bool = True,
        # token budget of that second request (the answer after the reasoning)
        after_think_max_gen_toks: int = 4096,
        **kwargs,
    ) -> None:
        super().__init__()
        missing_packages = [
            pkg
            for pkg in ["aiohttp", "tqdm", "tenacity", "requests"]
            if find_spec(pkg) is None
        ]
        if missing_packages:
            raise ModuleNotFoundError(
                f"Attempted to use an API model, but the required packages {missing_packages} are not installed. "
                'Please install these via `pip install lm-eval[api]` or `pip install -e ."[api]"`'
            )
        self.model = model or pretrained
        self.base_url = base_url
        self.tokenizer = tokenizer
        self._header = header
        if not isinstance(batch_size, int) and "auto" in batch_size:
            eval_logger.warning(
                "Automatic batch size is not supported for API models. Defaulting to batch size 1."
            )
            self._batch_size = 1
        elif int(batch_size) > 1:
            eval_logger.warning(
                "Batch size > 1 detected. Ensure your API supports batched requests with varying total sequence lengths."
            )
            self._batch_size = int(batch_size)
        else:
            self._batch_size = int(batch_size)
        self._truncate = truncate
        self._max_gen_toks = int(max_gen_toks)
        self._seed = int(seed)
        # max_length - 1 as we always have 1 token for generation
        eval_logger.info(f"Using max length {max_length} - 1")
        self.max_length = max_length - 1
        self.clamp_gen_toks = bool(clamp_gen_toks)
        self._context_limit = max_length
        self._n_clamped = 0
        self._clamp_count_failed = False
        if int(num_concurrent) <= 1:
            eval_logger.info(
                "Concurrent requests are disabled. To enable concurrent requests, set `num_concurrent` > 1."
            )
        self._concurrent = int(num_concurrent)
        self._rate_limiter = get_rate_limiter(
            requests_per_minute,
            scope=f"model:{self.base_url}:{self.model}",
        )
        if self._rate_limiter is not None:
            eval_logger.info(
                "Rate limiting API requests to %s requests/minute",
                self._rate_limiter.requests_per_minute,
            )
        self.tokenizer_backend = (
            None if tokenizer_backend in ("None", "none") else tokenizer_backend
        )
        self.add_bos_token = add_bos_token
        self.bos_by_client = bool(bos_by_client)
        self.custom_prefix_token_id = custom_prefix_token_id
        self.tokenized_requests = tokenized_requests
        self.max_retries = int(max_retries)
        self.verify_certificate = verify_certificate
        self.ca_cert_path = ca_cert_path
        self.auth_token = auth_token
        self._eos_string = eos_string
        self.timeout = int(timeout)
        self.max_images = int(max_images)
        self.after_think_prefix = bool(after_think_prefix)
        self.after_think_max_gen_toks = int(after_think_max_gen_toks)
        self.chat_template_args = dict(chat_template_args or {})
        if enable_thinking is not None:
            self.chat_template_args["enable_thinking"] = enable_thinking

        eval_logger.info(f"Using tokenizer {self.tokenizer_backend}")
        if self.tokenizer_backend is None:
            self.tokenizer = None
            self.tokenized_requests = False
        else:
            if self.tokenizer is None:
                if self.tokenizer_backend == "huggingface":
                    import transformers

                    self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                        self.tokenizer or self.model,
                        trust_remote_code=trust_remote_code,
                        revision=revision,
                        use_fast=use_fast_tokenizer,
                    )
                    # Not used as the API will handle padding but to mirror the behavior of the HFLM
                    self.tokenizer = configure_pad_token(self.tokenizer)
                elif self.tokenizer_backend == "tiktoken":
                    try:
                        import tiktoken

                        self.tokenizer = tiktoken.encoding_for_model(self.model)
                    except ModuleNotFoundError as e:
                        raise ModuleNotFoundError(
                            "Attempted to use 'openai' LM type, but the package `tiktoken` is not installed. "
                            "Please install it via `pip install lm-eval[api]` or `pip install -e .[api]`."
                        ) from e
                    if "openai" not in self.base_url:
                        eval_logger.warning(
                            f"Passed `base_url={self.base_url}` but using (OpenAI) Tiktoken tokenizer backend. "
                            "Pass `tokenizer_backend=huggingface` and provide the HF tokenizer name if your model does not use Tiktoken."
                        )
                elif self.tokenizer_backend == "remote":
                    from lm_eval.utils import RemoteTokenizer

                    if not self.base_url:
                        raise ValueError(
                            "base_url is required for remote tokenizer backend"
                        )
                    self.tokenizer = RemoteTokenizer(
                        self.base_url,
                        self.timeout,
                        self.verify_certificate,
                        self.ca_cert_path,
                        self.auth_token,
                    )
                    eval_logger.info(f"Using remote tokenizer from {self.base_url}")
            else:
                import transformers

                assert isinstance(tokenizer, str), "tokenizer must be a string"
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    tokenizer,
                    trust_remote_code=trust_remote_code,
                    revision=revision,
                    use_fast=use_fast_tokenizer,
                )

        hf_tokenizer = (
            self.tokenizer if self.tokenizer_backend == "huggingface" else None
        )
        if (
            self.bos_by_client
            and hf_tokenizer is not None
            and not self.add_bos_token
            and self._tokenizer_adds_bos
        ):
            # the server no longer adds the BOS this tokenizer would have added
            self.add_bos_token = True
        self.think_start_token, self.think_end_token = resolve_think_tokens(
            getattr(hf_tokenizer, "chat_template", None),
            autodetect_think_tokens,
            think_start_token,
            think_end_token,
        )
        self.think_open_prefilled = hf_tokenizer is not None and detect_open_prefilled(
            self.apply_chat_template, self.think_start_token
        )
        self.track_thinking_metrics = resolve_track_thinking_metrics(
            track_thinking_metrics, self.think_end_token
        )
        # Servers drop special tokens from the text they return; reasoning markers made
        # of them (gpt-oss's <|channel|>...<|message|>) would never be found. Ask to keep
        # them then (vLLM and SGLang take skip_special_tokens).
        self.keep_special_tokens = any(
            marker and any(tok in marker for tok in _special_tokens(hf_tokenizer))
            for marker in (self.think_start_token, self.think_end_token)
        )

    @abc.abstractmethod
    def _create_payload(
        self,
        messages: list[list[int]] | list[dict] | list[str] | str,
        *,
        generate: bool = True,
        gen_kwargs: dict | None = None,
        seed: int = 1234,
        eos: str | None = None,
        **kwargs,
    ) -> dict:
        """This method is responsible for creating the json payload that will be sent to the API."""
        raise NotImplementedError

    def create_message(
        self,
        messages: list[list[int]] | list[str] | list[JsonChatStr],
        generate=False,
    ) -> list[list[int]] | list[dict] | list[str] | str:
        """Helper method to transform the prompt into the expected API input format. messages consist of batched requests"""
        if isinstance(messages[0], JsonChatStr):
            # for chat completions we need to decode the json string to list[dict,...]
            assert self._batch_size == 1, (
                "non-tokenized chat requests are only supported with batch_size=1"
            )
            # list[dict["role":..., "content":...],...]
            return json.loads(messages[0].prompt)

        if not self.tokenized_requests:
            # if messages are tokenized:
            if isinstance(messages[0][0], int):
                # assuming decoding is lossless. However, this is only for loglikelihood requests
                # as we need to compute the context length. For generations, we don't need to tokenize.
                messages = self.decode_batch(messages)
            if self._batch_size <= 1:
                # if batch is 1 return str
                return messages[0]
            else:
                # list[str,...]
                return messages

        # list[list[int], ...]
        return messages

    @staticmethod
    @abc.abstractmethod
    def parse_logprobs(
        outputs: Any | list[Any],
        tokens: list[list[int]] | None = None,
        ctxlen: list[int] | None = None,
        **kwargs,
    ) -> list[tuple[float, bool]]:
        """Method used to parse the logprobs from the (batched) API response. This method should return a list of tuples"""
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def parse_generations(outputs: Any | list[Any], **kwargs) -> list[str]:
        """Method used to parse the generations from the (batched) API response. This method should return a list of str"""
        raise NotImplementedError

    @cached_property
    def api_key(self) -> str:
        """Override this property to return the API key for the API request."""
        return ""

    @cached_property
    def header(self) -> dict:
        """Override this property to return the headers for the API request."""
        return self._header or {"Authorization": f"Bearer {self.api_key}"}

    @property
    def tokenizer_name(self) -> str:
        """Must be defined for LM subclasses which implement Chat Templating.
        Should return the name of the tokenizer or chat template used.
        Used only to properly fingerprint caches when requests are being cached with `--cache_requests`, otherwise not used.
        """
        return ""

    def apply_chat_template(
        self,
        chat_history: list[dict[str, str]],
        add_generation_prompt: bool = True,
        **kwargs,
    ) -> str | JsonChatStr | list[dict]:
        """Applies a chat template to a list of chat history between user and model."""
        if self.tokenizer_backend == "huggingface" and self.tokenized_requests:
            return self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
                **self.chat_template_args,
                **kwargs,
            )
        # Past this point the server renders the template, so the arguments would be
        # dropped silently; refuse instead of scoring a different prompt than asked for.
        if self.chat_template_args or kwargs:
            raise ValueError(
                "chat_template_args (including enable_thinking) and task template "
                "arguments (e.g. tools) need the chat template rendered client-side with "
                "a Hugging Face tokenizer, e.g. local-completions with "
                "tokenizer_backend=huggingface. Here the server renders it (a "
                "chat-completions model, or no local tokenizer), so they can't be applied: "
                f"{sorted({**self.chat_template_args, **kwargs})}"
            )
        if self.tokenizer_backend == "remote" and self.tokenized_requests:
            return chat_history
        else:
            # bit of a hack. We'll load back before sending to the API
            return JsonChatStr(json.dumps(chat_history, ensure_ascii=False))

    @cached_property
    def eot_token_id(self) -> int | None:
        if self.tokenizer is None:
            return None
        else:
            if self.tokenizer_backend == "huggingface":
                return self.tokenizer.eos_token_id
            elif self.tokenizer_backend == "tiktoken":
                return self.tokenizer.eot_token
            elif self.tokenizer_backend == "remote":
                return self.tokenizer.eos_token_id

    @cached_property
    def eos_string(self) -> str | None:
        if self._eos_string:
            return self._eos_string
        elif self.tokenizer is not None:
            if self.tokenizer_backend == "huggingface":
                return self.tokenizer.eos_token
            elif self.tokenizer_backend == "tiktoken":
                return self.tokenizer.decode([self.tokenizer.eot_token])
            elif self.tokenizer_backend == "remote":
                return self.tokenizer.eos_token
        else:
            eval_logger.warning(
                "Cannot determine EOS string to pass to stop sequence. Manually set by passing `eos_string` to model_args."
            )
            return None

    @cached_property
    def prefix_token_id(self) -> int | None:
        if self.tokenizer is None:
            return None
        else:
            if self.custom_prefix_token_id is not None:
                return self.custom_prefix_token_id
            if self.tokenizer_backend == "huggingface":
                if self.tokenizer.bos_token_id is not None:
                    return self.tokenizer.bos_token_id
                return self.tokenizer.eos_token_id
            elif self.tokenizer_backend == "remote":
                return self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
            else:
                return self.tokenizer.eot_token

    def tok_encode(
        self,
        string: str,
        left_truncate_len: int | None = None,
        add_special_tokens: bool = False,
        truncation: bool = False,
        **kwargs,
    ) -> list[list[int]] | list[int] | list[str]:
        if self.tokenizer_backend is None:
            return [string]
        elif self.tokenizer_backend == "huggingface":
            # by default for CausalLM - false or self.add_bos_token is set
            if not add_special_tokens:
                add_special_tokens = False or self.add_bos_token
            bos_text = getattr(self.tokenizer, "bos_token", None)
            if (
                self.bos_by_client
                and isinstance(string, str)
                and bos_text
                and string.startswith(bos_text)
            ):
                # written by the chat template: the tokenizer must not add another
                add_special_tokens = False
            encoding: list[list[int]] | list[int] = self.tokenizer(
                string,
                add_special_tokens=add_special_tokens,
                truncation=truncation,
                return_attention_mask=False,
            ).input_ids
            if add_special_tokens and self.add_bos_token:
                if isinstance(string, str):
                    encoding = self._with_bos_ids(encoding)
                else:
                    encoding = [self._with_bos_ids(enc) for enc in encoding]

            # left-truncate the encoded context to be at most `left_truncate_len` tokens long
            if left_truncate_len:
                if not isinstance(string, str):
                    encoding = [enc[-left_truncate_len:] for enc in encoding]
                else:
                    encoding = encoding[-left_truncate_len:]

            return encoding
        elif self.tokenizer_backend == "remote":
            if isinstance(string, str):
                encoding = self.tokenizer.encode(string)
            else:
                encoding = [self.tokenizer.encode(s) for s in string]

            if left_truncate_len:
                if isinstance(string, str):
                    encoding = encoding[-left_truncate_len:]
                else:
                    encoding = [enc[-left_truncate_len:] for enc in encoding]

            return encoding
        else:
            try:
                encoding = self.tokenizer.encode(string)
            except (TypeError, ValueError):
                encoding = self.tokenizer.encode_batch(string)
            return encoding

    def _with_bos_ids(self, ids: list[int]) -> list[int]:
        """``ids`` with the BOS token in front, as ``add_bos_token`` asks.

        Some tokenizers define a BOS token but never add it, whatever
        ``add_special_tokens`` says (Gemma 4: ``add_bos_token`` unset in its
        tokenizer_config, unlike Gemma 3's ``true``), so asking the tokenizer is
        not enough. Already there (the tokenizer added it, or a chat template
        wrote it): unchanged, never doubled."""
        bos = getattr(self.tokenizer, "bos_token_id", None)
        if bos is None or (ids and ids[0] == bos):
            return ids
        return [bos, *ids]

    @cached_property
    def _tokenizer_adds_bos(self) -> bool:
        """Whether the tokenizer puts BOS in front by itself (add_special_tokens).
        The server tokenizes text prompts with the same tokenizer, so for these it
        adds BOS on its own and nothing may be prepended to text."""
        bos = getattr(self.tokenizer, "bos_token_id", None)
        if bos is None:
            return False
        ids = self.tokenizer("a", add_special_tokens=True).input_ids
        return bool(ids) and ids[0] == bos

    def _with_bos_text(self, prompt):
        """A plain-text prompt with the BOS token's text in front, for prompts sent
        as text (``tokenized_requests=False``), which the server tokenizes itself
        -- it reads the BOS text back as the BOS token. Only for tokenizers that
        never add BOS themselves (the server's copy then wouldn't either); others
        are unchanged. Chat-completions messages (JsonChatStr) are left to the
        server's template."""
        bos = getattr(self.tokenizer, "bos_token", None)
        if (
            not bos
            or not isinstance(prompt, str)
            or prompt.startswith(bos)
            or (self._tokenizer_adds_bos and not self.bos_by_client)
        ):
            return prompt
        return bos + prompt

    def decode_batch(self, tokens: list[list[int]]) -> list[str]:
        if self.tokenizer_backend == "huggingface":
            return self.tokenizer.batch_decode(tokens)
        elif self.tokenizer_backend == "tiktoken":
            return self.tokenizer.decode_batch(tokens)
        elif self.tokenizer_backend == "remote":
            return self.tokenizer.batch_decode(tokens)

    def _post(self, payload: dict) -> dict:
        """POST one payload and return the decoded JSON response."""
        if self._rate_limiter is not None:
            self._rate_limiter.acquire()
        response = requests.post(
            self.base_url,
            json=payload,
            headers=self.header,
            verify=self.verify_certificate,
            timeout=self.timeout,
        )
        if not response.ok:
            eval_logger.warning(
                f"API request failed with error message: {response.text}. Retrying..."
            )
        response.raise_for_status()
        return response.json()

    async def _apost(self, session: ClientSession, payload: dict) -> dict:
        """Async counterpart of ``_post``; raising lets the caller retry."""
        if self._rate_limiter is not None:
            await self._rate_limiter.acquire_async()
        async with session.post(
            self.base_url,
            json=payload,
            headers=self.header,
        ) as response:
            if not response.ok:
                error_text = await response.text()
                eval_logger.warning(
                    f"API request failed! Status code: {response.status}, "
                    f"Response text: {error_text}. Retrying..."
                )
            response.raise_for_status()
            return await response.json()

    def _send(self, payload: dict) -> dict:
        """Response for one request payload. Subclasses may issue several requests
        per generation and return one combined response."""
        return self._post(payload)

    async def _asend(self, session: ClientSession, payload: dict) -> dict:
        """Async counterpart of ``_send``."""
        return await self._apost(session, payload)

    def _prompt_token_count(self, message) -> int | None:
        """The prompt's token count by the client tokenizer; None if it can't tell
        (only the huggingface tokenizer is counted with)."""
        tok = self.tokenizer
        if self.tokenizer_backend != "huggingface" or tok is None:
            return None
        try:
            if isinstance(message, list) and message and isinstance(message[0], dict):
                # A chat message list: render it with the model's template, as the
                # server does, then count the text. Without a template, count the
                # contents.
                if getattr(tok, "chat_template", None):
                    message = tok.apply_chat_template(
                        message, add_generation_prompt=True, tokenize=False
                    )
                else:
                    return sum(
                        len(
                            tok.encode(
                                str(m.get("content", "")), add_special_tokens=False
                            )
                        )
                        + 8
                        for m in message
                    )
            if isinstance(message, str):
                return len(tok.encode(message, add_special_tokens=False))
            if isinstance(message, list) and message and isinstance(message[0], str):
                return max(
                    len(tok.encode(m, add_special_tokens=False)) for m in message
                )
        except Exception as e:  # noqa: BLE001 - a failed count must not fail the request
            if not self._clamp_count_failed:
                self._clamp_count_failed = True
                eval_logger.warning(
                    f"clamp_gen_toks: could not count a prompt ({e!r}); "
                    "such requests are sent unclamped."
                )
        return None

    def _clamp_gen_toks(self, message, gen_kwargs: dict | None, generate: bool) -> None:
        """With clamp_gen_toks, lower the request's token budget to what the context
        (max_length) leaves after the prompt, so a long prompt gets a shorter answer
        instead of the server's 400 "maximum context length". Short prompts keep the
        full budget. Without a token count the request is left as it is."""
        if not (self.clamp_gen_toks and generate and gen_kwargs is not None):
            return
        used = self._prompt_token_count(message)
        if used is None:
            return
        key = "max_tokens" if gen_kwargs.get("max_tokens") else "max_gen_toks"
        budget = int(gen_kwargs.get(key) or self._max_gen_toks)
        # slack for tokens the client count misses (special tokens, template wrapper)
        room = self._context_limit - used - 32
        if budget <= room:
            return
        if room < 1:
            # The prompt alone fills the context: sending 1 token would score a
            # silent non-answer, so leave the request to the server's clear 400.
            eval_logger.warning(
                f"clamp_gen_toks: a prompt of ~{used} tokens leaves no room in the "
                f"context of {self._context_limit}; sent unclamped."
            )
            return
        self._n_clamped += 1
        if self._n_clamped <= 3 or self._n_clamped % 100 == 0:
            eval_logger.warning(
                f"clamp_gen_toks: prompt of ~{used} tokens leaves {room} of "
                f"the requested {budget} (context {self._context_limit}); "
                f"{self._n_clamped} requests clamped so far."
            )
        gen_kwargs[key] = room

    def model_call(
        self,
        messages: list[list[int]] | list[str] | list[JsonChatStr],
        *,
        generate: bool = True,
        gen_kwargs: dict | None = None,
        **kwargs,
    ) -> dict | None:
        # !!! Copy: shared dict for each request, need new object !!!
        gen_kwargs = copy.deepcopy(gen_kwargs)
        seed = self._request_seed(gen_kwargs, generate)
        message = self.create_message(messages)
        self._clamp_gen_toks(message, gen_kwargs, generate)
        try:
            return self._send(
                self._create_payload(
                    message,
                    generate=generate,
                    gen_kwargs=gen_kwargs,
                    seed=seed,
                    eos=self.eos_string,
                    **kwargs,
                )
            )
        except RetryError:
            eval_logger.error(
                "API request failed after multiple retries. Please check the API status."
            )
            return None

    def _record_live_view(
        self, messages: list, answers: list, outputs: Any, payload: Any = None
    ) -> None:
        """Hand the finished generation(s) to the live view (loggers/live_view.py),
        which does nothing unless it is switched on."""
        if not os.environ.get(
            "LM_EVAL_LIVE_VIEW"
        ):  # loggers.live_view.ENV, imported lazily
            return
        try:
            from lm_eval.loggers import live_view

            prompts = _live_prompts(self, messages, len(answers))
            reasoning, output_tokens, reasoning_tokens = _live_details(outputs)
            # The request body sent (one per call; a batch shares it) and, for a single
            # generation, the response it got.
            request = payload if isinstance(payload, dict) else None
            response = outputs[0] if isinstance(outputs, list) and outputs else outputs
            if len(answers) != 1 or not isinstance(response, dict):
                response = None
            for index, answer in enumerate(answers):
                live_view.record(
                    prompts[index] if index < len(prompts) else "",
                    str(answer),
                    reasoning if len(answers) == 1 else "",
                    output_tokens if len(answers) == 1 else None,
                    reasoning_tokens if len(answers) == 1 else None,
                    request,
                    response,
                )
        except Exception as exc:  # noqa: BLE001 - never disturb the evaluation
            eval_logger.debug("live view not recorded: %s", exc)

    async def amodel_call(
        self,
        session: ClientSession,
        sem: asyncio.Semaphore,
        messages: list[list[int]] | list[str] | list[JsonChatStr],
        *,
        generate: bool = True,
        cache_keys: list | None = None,
        ctxlens: list[int] | None = None,
        gen_kwargs: dict | None = None,
        **kwargs,
    ) -> list[str] | list[tuple[float, bool]] | None:
        # !!! Copy: shared dict for each request, need new object !!!
        gen_kwargs = copy.deepcopy(gen_kwargs)
        seed = self._request_seed(gen_kwargs, generate)
        message = self.create_message(messages)
        self._clamp_gen_toks(message, gen_kwargs, generate)
        payload = self._create_payload(
            message,
            generate=generate,
            gen_kwargs=gen_kwargs,
            seed=seed,
            **kwargs,
        )
        cache_method = "generate_until" if generate else "loglikelihood"
        acquired = await sem.acquire()
        outputs = None
        try:
            outputs = await self._asend(session, payload)
            tmp_answers = (
                self.parse_generations(
                    outputs=outputs,
                )
                if generate
                else self.parse_logprobs(
                    outputs=outputs,
                    tokens=messages,
                    ctxlens=ctxlens,
                )
            )

            # Convert `None`` values to `LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER` string to maintain consistency
            answers = []
            for a in tmp_answers:
                if a is None:
                    eval_logger.warning(
                        f"API returned null content. Content filled with `LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER = {LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER}`. Check reasoning_content field or generation limits."
                    )
                    answers.append(LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER)
                else:
                    answers.append(a)
            if generate:
                answers = with_finish_reasons(answers, outputs)
                self._record_live_view(messages, answers, outputs, payload)

            # A thinking generation is cached once stripped (generate_until).
            if cache_keys and not (generate and self.think_end_token):
                for res, cache in zip(answers, cache_keys):
                    self.cache_hook.add_partial(cache_method, cache, res)
            return answers
        # If the retries also fail
        except BaseException as e:
            eval_logger.error(
                f"Exception:{_describe_error(e)}, {locals().get('outputs', '(no outputs)')}, retrying."
            )
            raise
        finally:
            if acquired:
                sem.release()

    def batch_loglikelihood_requests(
        self, chunks: Iterable[list[LogLikelihoodInputs]]
    ) -> tuple[list[list[int]], list[int], list[tuple[str, str]]]:
        inputs = []
        ctxlens = []
        cache_keys = []
        for chunk in chunks:
            for cache_key, context_enc, continuation_enc in chunk:
                # max_length - 1 as we always have 1 token for generation
                inp = (context_enc + continuation_enc)[-self.max_length :]
                if len(inp) < len(context_enc + continuation_enc):
                    eval_logger.warning(
                        f"Context length ({len(context_enc)}) + continuation length ({len(continuation_enc)}) > max_length ({self.max_length}). Left truncating context."
                    )
                ctxlen = len(context_enc) - max(
                    0, len(context_enc) + len(continuation_enc) - self.max_length
                )

                inputs.append(inp)
                ctxlens.append(ctxlen)
                cache_keys.append(cache_key)
        return inputs, ctxlens, cache_keys

    async def get_batched_requests(
        self,
        requests: list,
        cache_keys: list,
        *,
        generate: bool = True,
        ctxlens: list[int] | None = None,
        request_seeds: list[int | None] | None = None,
        **kwargs,
    ) -> list[list[str]] | list[list[tuple[float, bool]]]:
        # Distinct seeds require individual HTTP payloads, not batched prompts.
        batch_size = 1 if request_seeds is not None else self._batch_size

        ctxlens = ctxlens or [None] * len(requests)
        conn = TCPConnector(limit=self._concurrent, ssl=self.verify_certificate)
        sem = asyncio.Semaphore(self._concurrent)
        progress = _RequestProgress(
            "generate_until" if generate else "loglikelihood", len(requests)
        )

        def _before_sleep(retry_state) -> None:
            progress.retried()
            eval_logger.info(f"Retry attempt {retry_state.attempt_number}")

        async with ClientSession(
            connector=conn, timeout=ClientTimeout(total=self.timeout)
        ) as session:
            retry_: Callable[..., Awaitable[Any]] = retry(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential(multiplier=0.5, min=1, max=10),
                reraise=True,
                before_sleep=_before_sleep,
            )(self.amodel_call)
            # Create tasks for each batch of request
            tasks = []
            for i, (message, cache_key, ctxlen) in enumerate(
                zip(
                    chunks(requests, n=batch_size),
                    chunks(cache_keys, n=batch_size),
                    chunks(ctxlens, n=batch_size),
                )
            ):
                task_kwargs = kwargs
                if request_seeds is not None:
                    task_kwargs = {
                        **kwargs,
                        "gen_kwargs": {
                            **kwargs["gen_kwargs"],
                            "seed": request_seeds[i],
                        },
                    }
                task = asyncio.create_task(
                    retry_(
                        session=session,
                        sem=sem,
                        messages=message,
                        cache_keys=cache_key,
                        generate=generate,
                        ctxlens=ctxlen,
                        **task_kwargs,
                    )
                )
                size = len(message)
                task.add_done_callback(lambda _, n=size: progress.add(n))
                tasks.append(task)

            # Wait for every task before returning/raising -- gather()
            # propagating on the first failure would exit this `async with`
            # and close the shared session while sibling tasks are still
            # mid-request or asleep in a retry backoff, so they'd then crash
            # with "Session is closed" instead of their own real error.
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException):
                    raise result
            return results

    def _loglikelihood_tokens(self, requests, **kwargs) -> list[tuple[float, bool]]:
        assert self.tokenizer is not None, (
            "Tokenizer is required for loglikelihood tasks to compute context lengths."
        )
        res = []

        def _collate(req: LogLikelihoodInputs):
            """Defines the key for the sorted method"""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = req[1] + req[2]
            return -len(toks), tuple(toks)

        re_ord = Collator(
            requests,
            sort_fn=_collate,
            group_by=None,
        )
        # if concurrent then we'll batch in the async context
        chunked = re_ord.get_batched(n=self._batch_size if self._concurrent <= 1 else 0)
        if self._concurrent <= 1:
            pbar = tqdm(desc="Requesting API", total=len(requests))
            for chunk in chunked:
                inputs, ctxlens, cache_keys = self.batch_loglikelihood_requests([chunk])

                outputs = retry(
                    stop=stop_after_attempt(self.max_retries),
                    wait=wait_exponential(multiplier=0.5, min=1, max=10),
                    reraise=True,
                )(self.model_call)(messages=inputs, generate=False)
                if isinstance(outputs, dict):
                    outputs = [outputs]
                for answer_, cache_key in zip(
                    self.parse_logprobs(
                        outputs=outputs, tokens=inputs, ctxlens=ctxlens
                    ),
                    cache_keys,
                ):
                    if answer_ is not None:
                        res.append(answer_)
                        # cache requests that aren't from a loglikelihood_rolling request
                        if cache_key is not None:
                            self.cache_hook.add_partial(
                                "loglikelihood", cache_key, answer_
                            )
                        pbar.update(1)
        else:
            inputs, ctxlens, cache_keys = self.batch_loglikelihood_requests(chunked)
            res = itertools.chain.from_iterable(
                asyncio.run(
                    self.get_batched_requests(
                        inputs, cache_keys, generate=False, ctxlens=ctxlens
                    )
                )
            )

        return re_ord.get_original(res)

    def _sent_gen_kwargs(self, gen_kwargs: dict) -> dict:
        """The generation kwargs sent to the server. With a reasoning close known, the
        task's stop sequences stay behind: they often occur inside the reasoning (e.g.
        gsm8k's "Question:"), and would cut it off before any answer. They're applied
        after the reasoning is stripped (see generate_until), as on vllm/sglang."""
        gen_kwargs = copy.deepcopy(gen_kwargs)
        if self.think_end_token:
            gen_kwargs["until"] = []
        return gen_kwargs

    def _request_seed(self, gen_kwargs: dict | None, generate: bool) -> int | None:
        """A task's seed (offset per attempt) replaces the model seed; None sends none."""
        return (
            gen_kwargs.pop("seed", self._seed)
            if generate and gen_kwargs
            else self._seed
        )

    def _rollout_seed(self, seed: int, rollout_index: int) -> int:
        return rollout_seed(seed, rollout_index)

    def _prefixed_prompt(self, chat_history, render, full):
        """``full`` as a PrefixedPrompt when its last turn is a gen_prefix to move
        after the reasoning; otherwise ``full`` itself. ``render`` renders messages
        with the model's template (add_generation_prompt, continue_final_message)."""
        if not (
            self.after_think_prefix
            and self.think_end_token
            and chat_history
            and chat_history[-1].get("role") == "assistant"
            and isinstance(chat_history[-1].get("content"), str)
            and chat_history[-1]["content"]
        ):
            return full
        prefix = chat_history[-1]["content"]
        head = render(chat_history[:-1])
        if not (full.startswith(head) and full[len(head) :] == prefix):
            eval_logger.warning(
                "after_think_prefix: the template does not render the prefix as a plain "
                "suffix of the prompt; sending it before the reasoning as before."
            )
            return full
        return PrefixedPrompt(full, head, prefix)

    def _continue_after_think(self, texts, prefixed, gen_kwargs_list):
        """Second pass for requests whose gen_prefix was held back: continue each from
        ``head + reasoning + prefix`` and return ``reasoning + continuation`` (the prefix
        itself is not part of a generation, as when it is sent up front). A generation
        that never closed its reasoning keeps its text: nothing to continue from."""
        todo = []
        for i, (text, held) in enumerate(zip(texts, prefixed, strict=True)):
            if held is None:
                continue
            text = str(text)
            if self.think_end_token in text:
                reasoning = text[
                    : text.index(self.think_end_token) + len(self.think_end_token)
                ]
            elif getattr(texts[i], "finish_reason", None) == "length":
                continue
            else:
                reasoning = ""  # answered without reasoning: the prefix goes up front
            todo.append((i, reasoning, held))
        if not todo:
            return texts
        groups: dict[str, list] = {}
        for i, reasoning, held in todo:
            kw = {k: v for k, v in gen_kwargs_list[i].items() if k != "seed"}
            kw["max_gen_toks"] = self.after_think_max_gen_toks
            groups.setdefault(json.dumps(kw, sort_keys=True, default=str), []).append(
                (i, reasoning, held, kw)
            )
        texts = list(texts)
        for members in groups.values():
            kw = members[0][3]
            contexts = []
            for _, reasoning, held, _ in members:
                head = held.head
                if self.add_bos_token and self.tokenizer_backend == "huggingface":
                    head = self._with_bos_text(head)
                contexts.append(head + reasoning + held.prefix)
            seeds = [gen_kwargs_list[i].get("seed") for i, *_ in members]
            sent = self._sent_gen_kwargs(kw)
            if self._concurrent <= 1:
                answers = []
                for ctx, seed in zip(contexts, seeds, strict=True):
                    outputs = retry(
                        stop=stop_after_attempt(self.max_retries),
                        wait=wait_exponential(multiplier=0.5, min=1, max=10),
                        reraise=True,
                    )(self.model_call)(
                        messages=[ctx],
                        generate=True,
                        gen_kwargs={**sent, "seed": seed} if seed is not None else sent,
                    )
                    generated = self.parse_generations(outputs=outputs)
                    answers.extend(with_finish_reasons(generated, outputs))
            else:
                results = asyncio.run(
                    self.get_batched_requests(
                        contexts,
                        cache_keys=[None] * len(contexts),
                        generate=True,
                        gen_kwargs=sent,
                        request_seeds=seeds if len(set(seeds)) > 1 else None,
                    )
                )
                answers = list(itertools.chain.from_iterable(results))
            for (i, reasoning, _, _), answer in zip(members, answers, strict=True):
                merged = Generation(reasoning + ("" if answer is None else str(answer)))
                merged.finish_reason = getattr(answer, "finish_reason", None)
                merged.generation_info = getattr(texts[i], "generation_info", None)
                texts[i] = merged
        return texts

    def generate_until(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[str]:
        """Generate completions; a sampled attempt's seed is offset by its repeat index.

        A task-level seed replaces the model seed; seed=None leaves seeding to the
        server. The offset is set before batching, so retries keep it.
        """
        res = []
        instances = requests

        def _collate_gen(_requests):
            # sort by the length of the non-tokenized contexts
            return -len(_requests[0])

        # Let the API deal with tokenization
        if len(requests[0].args) > 2:
            assert self.tokenizer is None, (
                "tokenizer is not supported for multimodal requests yet!"
            )
            eval_logger.info(
                f"Using max_images {self.max_images}. Set in the model args."
            )
            requests, all_gen_kwargs, auxiliary_args = zip(
                *(req.args for req in requests)
            )
            requests = tuple(
                JsonChatStr(
                    json.dumps(
                        create_image_prompt(
                            y["visual"][: self.max_images], json.loads(x.prompt)
                        )
                    )
                )
                for x, y in zip(requests, auxiliary_args)
            )
        else:
            requests, all_gen_kwargs = zip(*(req.args for req in requests))

        # A gen_prefix held back until the reasoning is closed: send the prompt without it.
        prefixed = [r if isinstance(r, PrefixedPrompt) else None for r in requests]
        if any(p is not None for p in prefixed):
            requests = tuple(
                str(r.head) if p is not None else r
                for r, p in zip(requests, prefixed, strict=True)
            )

        seeded_gen_kwargs = []
        for instance, gen_kwargs in zip(instances, all_gen_kwargs, strict=True):
            seed = gen_kwargs.get("seed", self._seed)
            # Greedy requests keep their seed: an offset would only split batches.
            if seed is not None and float(gen_kwargs.get("temperature") or 0) > 0:
                seed = self._rollout_seed(seed, instance.repeat_index or 0)
                gen_kwargs = {**gen_kwargs, "seed": seed}
            seeded_gen_kwargs.append(gen_kwargs)
        all_gen_kwargs = seeded_gen_kwargs

        if self.tokenized_requests:
            encodings_list = self.tok_encode(
                requests, add_special_tokens=self.add_bos_token
            )
        else:
            if self.add_bos_token and self.tokenizer_backend == "huggingface":
                requests = tuple(self._with_bos_text(r) for r in requests)
            encodings_list = [None] * len(requests)
        request_args = [
            (a, b, c) for a, b, c in zip(requests, all_gen_kwargs, encodings_list)
        ]

        re_ord = Collator(
            request_args,
            sort_fn=_collate_gen,
            # Concurrent runs send each attempt's seed per request, so attempts that
            # differ only in seed stay in one group instead of running group by group.
            group_fn=lambda req: (
                {k: v for k, v in req[1].items() if k != "seed"}
                if self._concurrent > 1
                else req[1]
            ),
            group_by="gen_kwargs",
        )
        chunked = re_ord.get_batched(
            n=self._batch_size if self._concurrent <= 1 else 0, batch_fn=None
        )
        if not self.tokenized_requests:
            eval_logger.info(
                "Tokenized requests are disabled. Context + generation length is not checked."
            )
        if self._concurrent <= 1:
            pbar = tqdm(desc="Requesting API", total=len(requests))
            for chunk in chunked:
                contexts, all_gen_kwargs, encodings_list = zip(*chunk)
                if self.tokenized_requests:
                    max_gen_toks = all_gen_kwargs[0].get(
                        "max_gen_toks", self._max_gen_toks
                    )
                    max_context_len = self.max_length - max_gen_toks

                    encodings_list = [x[-max_context_len:] for x in encodings_list]

                    if any(
                        len(x) + max_gen_toks > self.max_length for x in encodings_list
                    ):
                        eval_logger.warning(
                            f"Some contexts exceeded (max length: ({self.max_length}) - max_gen_toks: ({max_gen_toks}). They were left truncated."
                        )

                req = encodings_list if self.tokenized_requests else contexts
                outputs = retry(
                    stop=stop_after_attempt(self.max_retries),
                    wait=wait_exponential(multiplier=0.5, min=1, max=10),
                    reraise=True,
                )(self.model_call)(
                    messages=req,
                    generate=True,
                    gen_kwargs=self._sent_gen_kwargs(all_gen_kwargs[0]),
                )
                generated = self.parse_generations(outputs=outputs, contexts=contexts)
                reasons = parse_finish_reasons(outputs)
                if len(reasons) != len(generated):
                    reasons = [None] * len(generated)
                for generated_text, context, reason in zip(
                    generated, contexts, reasons
                ):
                    # Always append to res to maintain the correct number of items
                    # even if generation failed (generated_text is None)
                    if generated_text is None:
                        eval_logger.warning(
                            f"API returned null content. Content filled with `LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER = {LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER}`. Check reasoning_content field or generation limits."
                        )
                        # Patch "None" answer with consistent value (async and sync calls)
                        res.append(Generation(LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER))
                    else:  # an existing Generation keeps its generation_info
                        res.append(
                            generated_text
                            if isinstance(generated_text, Generation)
                            else Generation(generated_text)
                        )
                    res[-1].finish_reason = reason

                    # partial caching only for successful generations; a thinking
                    # generation is cached once stripped (below)
                    if (
                        generated_text is not None
                        and context is not None
                        and not self.think_end_token
                    ):
                        self.cache_hook.add_partial(
                            "generate_until",
                            (context, all_gen_kwargs[0]),
                            generated_text,
                        )
                    pbar.update(1)
        else:
            for chunk in chunked:
                contexts, all_gen_kwargs, encodings_list = zip(*chunk)
                if self.tokenized_requests:
                    max_gen_toks = all_gen_kwargs[0].get(
                        "max_gen_toks", self._max_gen_toks
                    )
                    max_context_len = self.max_length - max_gen_toks

                    encodings_list = [x[-max_context_len:] for x in encodings_list]

                    if any(
                        len(x) + max_gen_toks > self.max_length for x in encodings_list
                    ):
                        eval_logger.warning(
                            f"Some contexts exceeded (max length: ({self.max_length}) - max_gen_toks ({max_gen_toks}). They were left truncated."
                        )

                req = encodings_list if self.tokenized_requests else contexts
                seeds = [kw.get("seed") for kw in all_gen_kwargs]
                results = itertools.chain.from_iterable(
                    asyncio.run(
                        self.get_batched_requests(
                            req,
                            cache_keys=list(zip(contexts, all_gen_kwargs, strict=True)),
                            generate=True,
                            gen_kwargs=self._sent_gen_kwargs(all_gen_kwargs[0]),
                            request_seeds=seeds if len(set(seeds)) > 1 else None,
                        )
                    )
                )
                # Append results to res list
                res.extend(results)

        res = re_ord.get_original(res)
        if any(p is not None for p in prefixed):
            res = self._continue_after_think(res, prefixed, list(all_gen_kwargs))
        # Per-response length and thinking-format info on the raw generation, as on
        # vllm/sglang (thinking_* only with a close token known).
        hf_tokenizer = (
            self.tokenizer if self.tokenizer_backend == "huggingface" else None
        )
        attach_length_info(
            instances,
            [
                build_length_info(
                    text,
                    think_start_token=self.think_start_token,
                    think_end_token=self.think_end_token,
                    tokenizer=hf_tokenizer,
                    open_prefilled=self.think_open_prefilled,
                    track_thinking_metrics=self.track_thinking_metrics,
                )
                | truncation_info(text)
                | (getattr(text, "generation_info", None) or {})
                for text in res
            ],
        )
        if self.think_end_token:
            # Strip the reasoning, then apply the task stops held back from the server.
            res = [
                postprocess_generated_text(
                    text, instance.args[1].get("until"), self.think_end_token
                )
                for text, instance in zip(res, instances, strict=True)
            ]
            for text, instance in zip(res, instances, strict=True):
                self.cache_hook.add_partial(
                    "generate_until", (instance.args[0], instance.args[1]), text
                )
        return res

    def loglikelihood_rolling(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[float]:
        all_windows = []
        window_counts = []

        # Flatten the windows of all documents, so we can evaluate all documents in parallel
        for (string,) in tqdm([req.args for req in requests], disable=disable_tqdm):
            token_list = self.tok_encode(string)
            # The first window is conditioned on prefix_token_id (the BOS token when
            # there is one). The BOS put in front by _with_bos_ids would then be
            # scored as its own first token (BOS given BOS): drop it again. Only
            # where this class added it -- tokenizers that add BOS themselves keep
            # their previous behaviour.
            if (
                self.add_bos_token
                and not self._tokenizer_adds_bos
                and token_list
                and token_list[0] == self.prefix_token_id
            ):
                token_list = token_list[1:]
            rolling_token_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=token_list,
                        prefix_token=self.prefix_token_id,
                        # max_seq_len - (1 for context)
                        max_seq_len=self.max_length - 1,
                        context_len=1,
                    ),
                )
            )

            # TODO: Right now, we pass single EOT token to the Encoder and the full context to the decoder, in seq2seq case
            rolling_token_windows = [(None,) + x for x in rolling_token_windows]
            all_windows.extend(rolling_token_windows)

            window_counts.append(len(rolling_token_windows))

        # Compute the loglikelihoods for all windows in parallel, then sum the scores for each document
        window_scores = (
            self._loglikelihood_tokens(all_windows, disable_tqdm=disable_tqdm)
            if all_windows
            else []
        )

        loglikelihoods = []
        offset = 0
        for request, window_count in zip(requests, window_counts, strict=True):
            (string,) = request.args
            string_nll = sum(
                score for score, _ in window_scores[offset : offset + window_count]
            )
            offset += window_count  # Go to the next document's windows
            loglikelihoods.append(string_nll)

            # cache this loglikelihood_rolling request
            self.cache_hook.add_partial("loglikelihood_rolling", (string,), string_nll)
        return loglikelihoods
