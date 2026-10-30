"""Apertus 2 native chat format over a text ``/v1/completions`` endpoint.

Tasks see a chat model (messages in; reply plus tool calls after
``TOOL_CALLS_MARKER`` out). Prompts are rendered and outputs parsed on the client
with ``apertus-common``; the server only completes text. See docs/apertus2.md.
"""

import hashlib
import inspect
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from typing import Any

from lm_eval.api.registry import register_model
from lm_eval.models.api_models import Generation, JsonChatStr
from lm_eval.models.openai_completions import TOOL_CALLS_MARKER, LocalChatCompletion
from lm_eval.models.utils import truncate_before_stops


eval_logger = logging.getLogger(__name__)

# Payload key of the request state, popped by the send hooks before posting,
# and choice key of the per-sample info in the combined response.
_REQUEST = "_apertus2_request"
_INFO = "_apertus2_info"
_CALL_ID = re.compile(r"apertus_(\d+)")
_PREFILL_MODES = ("after_think", "empty_think", "reject")
_INFO_KEYS = (
    "generation_request_failed",
    "generation_prompt_too_long",
    "generation_budget_clamped",
    "generation_literal_special_tokens",
    "generation_parse_failed",
    "generation_think_truncated",
    "generation_prefill_skipped",
    "generation_unknown_output",
    "generation_has_claim",
    "generation_map_failed",
    "generation_call_counter_reused",
    "generation_unexpected_call",
)
_LIBRARY_HINT = (
    "apertus2-completions needs apertus-common with open output messages "
    "(https://github.com/swiss-ai/apertus-common, commit e0c1581 on "
    "feat/open-output-messages)"
)


def _import_apertus_common():
    """Import apertus-common; it needs Python >= 3.12 and ``allow_open`` parsing."""
    if sys.version_info < (3, 12):
        raise ImportError(f"{_LIBRARY_HINT}, which needs Python >= 3.12")
    try:
        import apertus_common
    except ImportError as error:
        raise ImportError(_LIBRARY_HINT) from error
    if (
        "allow_open"
        not in inspect.signature(apertus_common.parse_output_text).parameters
    ):
        raise ImportError(f"{_LIBRARY_HINT}; the installed version is older")
    return apertus_common


def _prepare_history(messages: list[dict], lib: Any) -> tuple[list[dict], Any]:
    """Normalize chat history for ``from_openai`` and rebuild call associations.

    Ported from the vLLM Apertus 2 integration: null fields, participant names and
    empty reasoning carry no native meaning, and call IDs minted from native
    counters (``apertus_<counter>``) keep those counters across requests.
    """
    prepared = []
    associations = {}
    for raw in messages:
        message = {
            key: value
            for key, value in dict(raw).items()
            if (value is not None or key == "content") and key != "name"
        }
        if message.get("role") == "assistant":
            if "reasoning" in message:
                message.setdefault("reasoning_content", message.pop("reasoning"))
            if message.get("reasoning_content") == "":
                del message["reasoning_content"]
            calls = message.get("tool_calls")
            if not calls:
                message.pop("tool_calls", None)
            elif message.get("content") == "":
                message["content"] = None
            for call in calls if isinstance(calls, list) else ():
                function = call.get("function") if isinstance(call, dict) else None
                if not isinstance(function, dict):
                    continue
                match = _CALL_ID.fullmatch(str(call.get("id")))
                name = function.get("name")
                if match and isinstance(name, str) and name:
                    counter = int(match[1])
                    associations[counter] = lib.ToolAssociation(
                        id=match[0], name=name, counter=counter
                    )
        prepared.append(message)
    context = lib.OpenAIMappingContext(
        associations=tuple(associations.values()),
        next_counter=max(associations, default=-1) + 1,
    )
    return prepared, context


def _join_calls(content: str, calls: list | None) -> str:
    """A task answer: the reply, then any tool calls after ``TOOL_CALLS_MARKER``."""
    if not calls:
        return content
    return content + TOOL_CALLS_MARKER + json.dumps(calls, ensure_ascii=False)


def _history_answer(message: dict) -> str | None:
    """The answer an assistant history message was built from, stripped.

    Tasks store the returned answer verbatim (or stripped), or split at
    ``TOOL_CALLS_MARKER`` into content and parsed tool calls (BFCL).
    """
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        return None
    return _join_calls(content or "", message.get("tool_calls")).strip()


def _with_reasoning(prepared: list[dict], replays: dict[int, list[str]]) -> list[dict]:
    """Prepared messages with replayed thinks before their assistant messages.

    Runs after ``_prepare_history``, which keeps messages 1:1 but deletes empty
    reasoning; replayed thinks may be empty. Each think becomes its own assistant
    message, which the mapping merges into one burst. An empty answer is replayed
    as thinks only, so a cut think gets no empty reply.
    """
    result = []
    for index, message in enumerate(prepared):
        thinks = replays.get(index)
        if thinks:
            result.extend(
                {"role": "assistant", "content": None, "reasoning_content": think}
                for think in thinks[:-1]
            )
            message = {
                **message,
                "content": message.get("content") or None,
                "reasoning_content": thinks[-1],
            }
        result.append(message)
    return result


class _ScopedMessages(list):
    """Decoded chat messages with the conversation scope and repeat index of their
    request."""

    scope: tuple | None = None
    repeat: int = 0


def _first_choice(response: Any) -> dict | None:
    """The response's first choice, or None for a missing or malformed response."""
    try:
        choice = response["choices"][0]
    except (KeyError, IndexError, TypeError):
        return None
    return choice if isinstance(choice, dict) else None


def _used_tokens(choice: dict, response: dict) -> int:
    """Tokens generated by a response; 0 when the server did not report them."""
    ids = choice.get("token_ids")
    if isinstance(ids, list):
        return len(ids)
    usage = response.get("usage")
    tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
    return tokens if isinstance(tokens, int) else 0


@dataclass
class _Request:
    """State of one generation, carried from the payload to the response."""

    info: dict
    until: Any
    has_tools: bool
    max_tokens: int
    conversation: Any = None
    context: Any = None
    prefix: str | None = None
    # The open reply holding the prefix, and the open items the response
    # continues (parsed in front of the generated output).
    open_reply: Any = None
    head: list = field(default_factory=list)
    prompt: str = ""
    prompt_ids: list[int] | None = None
    # After-think: no budget left after a closed think; no second request.
    out_of_budget: bool = False
    # Reasoning replay: the conversation (task, document) and the digest of
    # the request's messages.
    scope: tuple | None = None
    history_digest: str | None = None


@register_model("apertus2-completions")
class Apertus2Completions(LocalChatCompletion):
    """Apertus 2 chat format rendered client-side, sent as a text completion."""

    MULTIMODAL = False

    def __init__(
        self,
        base_url=None,
        tokenizer=None,
        reasoning_effort=None,
        prefill="after_think",
        literal_special_tokens="flag",
        mask_special_tokens=False,
        return_token_ids=True,
        max_model_len=None,
        log_native="output",
        replay_reasoning=True,
        tokenizer_backend=None,
        tokenized_requests=None,
        think_end_token=None,
        think_start_token=None,
        **kwargs,
    ):
        if not tokenizer:
            raise ValueError("apertus2-completions needs tokenizer=<checkpoint dir>")
        if tokenizer_backend is not None or tokenized_requests:
            raise ValueError(
                "apertus2-completions renders and tokenizes with apertus-common; "
                "leave tokenizer_backend and tokenized_requests unset"
            )
        if think_end_token is not None or think_start_token is not None:
            raise ValueError(
                "apertus2-completions separates reasoning natively; "
                "think_start_token/think_end_token do not apply"
            )
        for name, value, allowed in (
            ("prefill", prefill, _PREFILL_MODES),
            ("literal_special_tokens", literal_special_tokens, ("flag", "reject")),
            ("log_native", log_native, ("output", "full")),
        ):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
        lib = _import_apertus_common()
        super().__init__(base_url=base_url, **kwargs)
        self._lib = lib
        self.reasoning_effort = self._resolve_reasoning_effort(reasoning_effort)
        self.prefill = prefill
        self.literal_special_tokens = literal_special_tokens
        self.mask_special_tokens = bool(mask_special_tokens)
        self.return_token_ids = bool(return_token_ids)
        self.max_model_len = int(max_model_len) if max_model_len else None
        self.log_native = log_native
        self.replay_reasoning = bool(replay_reasoning)
        # scope -> history digest -> stripped answer -> think payloads
        self._memory: dict[tuple, dict] = {}
        self._scopes: dict[int, tuple] = {}
        self._repeats: dict[int, int] = {}
        self._artifact = str(tokenizer)
        self._bind_encoding(lib.load_encoding(tokenizer))
        self._policy = lib.MappingPolicy(
            merge_system_messages=True,
            merge_assistant_messages=True,
            omit_claims=True,
        )

    def _resolve_reasoning_effort(self, reasoning_effort: str | None) -> str:
        """Reasoning effort from the argument or ``chat_template_args``."""
        template_args = dict(self.chat_template_args)
        if template_args.pop("enable_thinking", True) is False:
            raise ValueError("Apertus 2 thinking cannot be disabled")
        template_effort = template_args.pop("reasoning_effort", None)
        if template_args:
            raise ValueError(
                f"unsupported chat_template_args for Apertus 2: {sorted(template_args)}"
            )
        effort = reasoning_effort or template_effort or "medium"
        allowed = [level.value for level in self._lib.Thinking]
        if effort not in allowed:
            raise ValueError(
                f"reasoning_effort must be one of {allowed}, got {effort!r}"
            )
        return effort

    def _bind_encoding(self, encoding: Any) -> None:
        """Control IDs, special-token text and masks of the tokenizer."""
        self.encoding = encoding
        self._wait = encoding.controls.wait
        self._output_end = encoding.controls.output_end
        self._bos_text = encoding.id_to_token(encoding.bos_token_id)
        specials = sorted(
            filter(None, map(encoding.id_to_token, encoding.special_token_ids)),
            key=len,
            reverse=True,
        )
        self._literal = re.compile("|".join(map(re.escape, specials)))
        self._logit_bias = {
            str(token): -100.0
            for token in encoding.generation_constraints.forbidden_token_ids
        }
        # vLLM keeps the matched stop/EOS ID in token_ids. Wait and output-end
        # are part of the output framing; any other trailing special ID, such as
        # EOS 2 from generation_config, is dropped before parsing.
        self._droppable = frozenset(encoding.special_token_ids) - {
            self._wait,
            self._output_end,
        }

    @property
    def eos_string(self) -> None:
        """Stops are token IDs; there is no EOS string."""
        return None

    @property
    def tokenizer_name(self) -> str:
        return f"apertus2:{self._artifact}"

    def chat_template(self, chat_template: bool | str = False) -> str:
        """Provenance for the results config; there is no Jinja template."""
        return (
            f"apertus2-native artifact={self._artifact} "
            f"effort={self.reasoning_effort} prefill={self.prefill} "
            f"replay={self.replay_reasoning}"
        )

    def loglikelihood(self, requests, **kwargs):
        raise NotImplementedError(
            "apertus2-completions serves generative tasks only; likelihood and "
            "multiple-choice tasks need a separate Apertus 2 scoring path"
        )

    def loglikelihood_rolling(self, requests, **kwargs):
        raise NotImplementedError(
            "apertus2-completions serves generative tasks only (no rolling likelihood)"
        )

    # Request path

    def _literal_count(self, messages: list[dict], tools: str | None = None) -> int:
        """Special-token spellings in message contents and the tools JSON."""
        texts = (json.dumps(messages, ensure_ascii=False), tools or "")
        return sum(len(self._literal.findall(text)) for text in texts)

    def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
        # Configuration errors fail before any request; data errors are per sample.
        for request in requests:
            context, gen_kwargs = request.args[0], request.args[1]
            if not isinstance(context, JsonChatStr):
                raise TypeError(
                    "apertus2-completions needs chat messages: run with "
                    "--apply_chat_template"
                )
            messages = json.loads(context.prompt)
            if (
                self.prefill == "reject"
                and messages
                and messages[-1].get("role") == "assistant"
            ):
                raise ValueError("prefill=reject: the task prefills the answer")
            # Only text the task wrote: assistant turns may hold model output.
            task_text = [m for m in messages if m.get("role") != "assistant"]
            if self.literal_special_tokens == "reject" and self._literal_count(
                task_text, gen_kwargs.get("tools")
            ):
                raise ValueError(
                    "literal_special_tokens=reject: a request contains "
                    "special-token text"
                )
        # Replay memory is per conversation (task, document); the context object
        # reaches create_message unchanged and carries the scope there. Each
        # multi-turn wave holds every ongoing conversation, so the others ended.
        self._scopes = {
            id(r.args[0]): (r.task_name, r.doc_id)
            for r in requests
            if r.task_name is not None
        }
        active = set(self._scopes.values())
        self._memory = {s: m for s, m in self._memory.items() if s in active}
        # Repeats of a document send the same context object; each gets its own
        # seed, or a seeded server would return one sample for all of them.
        self._repeats = {}
        return super().generate_until(requests, disable_tqdm)

    def create_message(self, messages, generate=False):
        decoded = _ScopedMessages(super().create_message(messages, generate))
        key = id(messages[0])
        decoded.scope = self._scopes.get(key)
        decoded.repeat = self._repeats.get(key, 0)
        self._repeats[key] = decoded.repeat + 1
        return decoded

    def _create_payload(
        self,
        messages,
        generate=False,
        gen_kwargs: dict | None = None,
        seed=1234,
        eos=None,
        **kwargs,
    ) -> dict:
        gen_kwargs = gen_kwargs or {}  # callers pass a fresh copy
        gen_kwargs.pop("do_sample", None)
        max_gen_toks = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
        tools = gen_kwargs.pop("tools", None)
        request = _Request(
            info=dict.fromkeys(_INFO_KEYS, 0),
            until=gen_kwargs.pop("until", None),
            has_tools=bool(tools),
            max_tokens=int(gen_kwargs.pop("max_tokens", None) or max_gen_toks),
            scope=getattr(messages, "scope", None),
        )
        temperature = gen_kwargs.pop("temperature", 0)
        try:
            self._prepare_request(request, list(messages), tools)
        except (self._lib.ApertusFormatError, ValueError) as error:
            eval_logger.warning(f"Apertus 2 request skipped: {error}")
            request.info["generation_request_failed"] = 1
        if request.prompt_ids is None:
            return {_REQUEST: request}
        stop = [self._wait]
        if request.prefix is not None and self.prefill == "after_think":
            stop = [self._output_end, self._wait]
        payload = {
            "model": self.model,
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": temperature,
            "seed": seed + getattr(messages, "repeat", 0),
            **gen_kwargs,
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
            "add_special_tokens": False,
            "stop_token_ids": stop,
        }
        if self.return_token_ids:
            payload["return_token_ids"] = True
        if self.mask_special_tokens:
            payload["logit_bias"] = self._logit_bias
        payload[_REQUEST] = request
        return payload

    def _prepare_request(
        self, request: _Request, messages: list[dict], tools: str | None
    ) -> None:
        """Map the messages and render the first prompt into ``request``."""
        lib = self._lib
        request.info["generation_literal_special_tokens"] = self._literal_count(
            messages, tools
        )
        if messages and messages[-1].get("role") == "assistant":
            request.prefix = messages.pop().get("content") or ""
            # Built here so an invalid prefix fails before any request.
            request.open_reply = lib.Reply(payload=request.prefix, open=True)
        replays, unmatched = self._recall(request, messages)
        prepared, context = _prepare_history(messages, lib)
        prepared = _with_reasoning(prepared, replays)
        mapped = lib.from_openai(
            prepared,
            tools=json.loads(tools) if tools else None,
            reasoning_effort=self.reasoning_effort,
            context=context,
            policy=self._policy,
        )
        request.conversation, request.context = mapped.value, mapped.context
        extra = []
        if request.open_reply is not None and self.prefill == "empty_think":
            # The empty think is the harness's, so the response is parsed after
            # the open reply only.
            request.head = [request.open_reply]
            extra = [lib.Think(payload=""), request.open_reply]
        prompt, prompt_ids = self._render(request, extra)
        budget = request.max_tokens
        if self.max_model_len is not None:
            room = self.max_model_len - len(prompt_ids)
            if room < 1:
                request.info["generation_prompt_too_long"] = 1
                return
            if budget > room:
                request.info["generation_budget_clamped"] = 1
                budget = room
        request.prompt = prompt
        request.prompt_ids = prompt_ids
        request.max_tokens = budget
        if unmatched is not None:  # the history holds earlier answers of the model
            thinks = [think for burst in replays.values() for think in burst]
            request.info["generation_reasoning_replayed"] = len(thinks)
            request.info["generation_reasoning_unmatched"] = int(unmatched)
            request.info["generation_literal_special_tokens"] += sum(
                len(self._literal.findall(think)) for think in thinks
            )

    def _render(self, request: _Request, extra: list) -> tuple[str, list[int]]:
        """Prompt text and native IDs of the inputs followed by ``extra`` items."""
        lib = self._lib
        conversation = lib.Conversation(
            system=request.conversation.system,
            items=[*request.conversation.items, *extra],
        )
        prompt_ids = self.encoding.encode_conversation(conversation)
        return self._bos_text + lib.render_str(conversation), prompt_ids

    # Reasoning replay: earlier thinks stay in context; tasks drop them

    def _recall(
        self, request: _Request, messages: list[dict]
    ) -> tuple[dict[int, list[str]], bool | None]:
        """Thinks to replay by message index, and whether the task changed an answer.

        The flag is None when the history holds no earlier answer of the model.
        Records the history digest on ``request``.
        """
        if not self.replay_reasoning or request.scope is None:
            return {}, None
        histories = self._memory.get(request.scope, {})
        replays, unmatched = {}, None
        history = hashlib.sha256()
        for index, message in enumerate(messages):
            role = message.get("role")
            answer = _history_answer(message) if role == "assistant" else None
            answers = histories.get(history.hexdigest())
            reasoned = message.get("reasoning_content") or message.get("reasoning")
            if answers is not None and answer is not None and not reasoned:
                # The model answered this history: a miss means the task changed the
                # answer.
                thinks = answers.get(answer)
                if thinks:
                    replays[index] = thinks
                unmatched = unmatched or thinks is None
            # System messages are left out: some tasks rebuild them per request.
            if role not in ("system", "developer"):
                history.update(json.dumps(message, sort_keys=True).encode())
        request.history_digest = history.hexdigest()
        return replays, unmatched

    def _remember(self, request: _Request, answer: str, thinks: list) -> None:
        """Keep a response's thinks for later requests of its conversation."""
        if request.history_digest is not None and request.prefix is None:
            histories = self._memory.setdefault(request.scope, {})
            histories.setdefault(request.history_digest, {})[answer.strip()] = thinks

    # Send hooks: one or two requests per generation, one combined response

    def _send(self, payload: dict) -> dict:
        request = payload.pop(_REQUEST)
        if request.prompt_ids is None:
            return self._response(request)
        response = self._post(payload)
        second_payload = self._phase_two_payload(request, payload, response)
        if second_payload is not None:
            response = self._post(second_payload)
        return self._response(request, response, sent=True)

    async def _asend(self, session, payload: dict) -> dict:
        request = payload.pop(_REQUEST)
        if request.prompt_ids is None:
            return self._response(request)
        response = await self._apost(session, payload)
        second_payload = self._phase_two_payload(request, payload, response)
        if second_payload is not None:
            response = await self._apost(session, second_payload)
        return self._response(request, response, sent=True)

    def _phase_two_payload(
        self, request: _Request, payload: dict, first: Any
    ) -> dict | None:
        """After-think prefill: the request continuing the reply after the think.

        None when there is no second request: no after-think prefill, a malformed
        first response, a reply before any think, a cut inside the think, or no
        budget left. Records the outcome on ``request`` (flags, the prefill
        ``head`` and prompt IDs, or ``out_of_budget``).
        """
        lib = self._lib
        if request.prefix is None or self.prefill != "after_think":
            return None
        choice = _first_choice(first)
        if choice is None:
            return None
        output, _, _ = self._parse(choice, [])
        if not output.items or not isinstance(output.items[0], lib.Think):
            request.info["generation_prefill_skipped"] = 1
            return None
        think = output.items[0]
        stopped = (
            choice.get("finish_reason") == "stop"
            and choice.get("stop_reason") == self._output_end
        )
        if think.open and not stopped:
            return None  # cut off inside thinking
        think = think.model_copy(update={"open": False})
        request.info["generation_literal_special_tokens"] += len(
            self._literal.findall(think.payload)
        )
        head = [think, request.open_reply]
        prompt, prompt_ids = self._render(request, head)
        # Stay within phase one's accepted prompt + budget: the server's context.
        total = len(request.prompt_ids) + request.max_tokens
        used = _used_tokens(choice, first)
        budget = min(request.max_tokens - used, total - len(prompt_ids))
        if budget < 1:
            request.out_of_budget = True
            return None
        # request.prompt stays the task prompt for logging.
        request.head, request.prompt_ids = head, prompt_ids
        return {
            **payload,
            "prompt": prompt,
            "max_tokens": budget,
            "stop_token_ids": [self._wait],
        }

    # Response path

    def _parse(self, choice: dict, head: list) -> tuple[Any, str, bool]:
        """Parse a choice after the ``head`` items.

        Returns (output, raw native text, parse failed). Token IDs parse exactly;
        text misreads control spellings the model wrote as ordinary tokens.
        """
        lib = self._lib
        encoding = self.encoding
        ids, text = choice.get("token_ids"), choice.get("text")
        valid_ids = ids is None or (
            isinstance(ids, list) and all(type(i) is int for i in ids)
        )
        valid_text = text is None or isinstance(text, str)
        if not valid_ids or not valid_text or (ids is None and text is None):
            return lib.Conversation(), "", True
        raw = ""
        try:
            if ids is not None:
                stopped = choice.get("finish_reason") == "stop"
                if stopped and ids and ids[-1] in self._droppable:
                    ids = ids[:-1]
                ids = encoding.encode_items(head) + ids
                raw = encoding.decode_text(ids)
                return encoding.parse_output(ids, allow_open=True), raw, False
            raw = lib.render_str(lib.Conversation(items=head)) + (text or "")
            return lib.parse_output_text(raw, allow_open=True), raw, False
        except lib.StreamParseError as error:
            return error.completed, raw, True
        except lib.ApertusFormatError:  # undecodable token IDs
            return lib.Conversation(), raw, True

    def _response(self, request: _Request, response: Any = None, sent=False) -> dict:
        """One combined completion response carrying the answer and its info."""
        info = request.info
        choice = _first_choice(response)
        if choice is None:
            info["generation_parse_failed"] = int(sent)
            return {"choices": [self._choice(request, "", None, "", [], [])]}
        output, raw, failed = self._parse(choice, request.head)
        info["generation_parse_failed"] = int(failed)
        prompt_ids = choice.get("prompt_token_ids")
        if prompt_ids is not None:
            info["generation_prompt_mismatch"] = int(prompt_ids != request.prompt_ids)
        items = list(output.items)
        finish_reason = choice.get("finish_reason")
        if request.out_of_budget:
            # Phase one stopped at the think's close, which the text omits.
            items[0] = items[0].model_copy(update={"open": False})
            finish_reason = "length"
        answer, thinks, replies = self._answer(request, items)
        self._remember(request, answer, thinks)
        return {
            "choices": [
                self._choice(request, answer, finish_reason, raw, thinks, replies)
            ]
        }

    def _answer(self, request: _Request, items: list) -> tuple[str, list, list]:
        """Task answer from parsed items, plus think and generated reply payloads."""
        lib = self._lib
        info = request.info
        # The prefix is the task's text, not generated; it is not part of the answer.
        prefix = request.prefix if request.head else None
        kept, thinks, replies = [], [], []
        for item in items:
            if isinstance(item, lib.UnknownOutput):
                info["generation_unknown_output"] = 1
                continue
            if isinstance(item, lib.Claim):
                info["generation_has_claim"] = 1
                continue
            if isinstance(item, lib.Think):
                thinks.append(item.payload)
                info["generation_think_truncated"] |= int(item.open)
            elif isinstance(item, lib.Reply):
                if prefix is not None and not replies:
                    item = item.model_copy(
                        update={"payload": item.payload.removeprefix(prefix)}
                    )
                replies.append(item.payload)
            elif isinstance(item, lib.Call) and item.open:
                continue  # an unfinished call is never dispatched
            if isinstance(item, (lib.Think, lib.Reply)):
                item = item.model_copy(update={"open": False})
            kept.append(item)
        try:
            mapped = lib.to_openai_output(
                lib.Conversation(items=kept),
                context=request.context,
                policy=self._policy,
            )
        except lib.ApertusFormatError as error:
            eval_logger.warning(f"Apertus 2 output not mappable: {error}")
            info["generation_map_failed"] = 1
            return "", thinks, replies
        message = mapped.value
        content = truncate_before_stops(message.get("content") or "", request.until)
        calls = message.get("tool_calls") or []
        known = {association.id for association in request.context.associations}
        if any(call.get("id") in known for call in calls):
            info["generation_call_counter_reused"] = 1
            calls = []
        if calls and not request.has_tools:
            info["generation_unexpected_call"] = 1
            calls = []
        return _join_calls(content, calls), thinks, replies

    def _choice(
        self,
        request: _Request,
        answer: str,
        finish_reason: str | None,
        raw: str,
        thinks: list,
        replies: list,
    ) -> dict:
        """Synthetic completion choice with the per-sample info."""
        info = request.info
        closed_think = bool(thinks) and not info["generation_think_truncated"]
        info["thinking_format_has_open"] = int(bool(thinks))
        info["thinking_format_has_close"] = int(closed_think)
        info["thinking_format_correct"] = int(
            closed_think and not info["generation_parse_failed"]
        )
        generated = [*thinks, *replies]
        info["response_length_words"] = sum(len(text.split()) for text in generated)
        info["response_length_chars"] = sum(map(len, generated))
        if info["thinking_format_correct"]:
            info["thinking_length_words"] = sum(len(text.split()) for text in thinks)
            info["thinking_length_chars"] = sum(map(len, thinks))
        info["apertus2_prefill_mode"] = (
            self.prefill if request.prefix is not None else "none"
        )
        info["apertus2_output"] = raw
        prompt = request.prompt
        if self.log_native != "full" and prompt:
            prompt = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        info["apertus2_prompt"] = prompt
        return {"index": 0, "text": answer, "finish_reason": finish_reason, _INFO: info}

    def parse_generations(self, outputs, **kwargs) -> list[str]:
        res = []
        for out in outputs if isinstance(outputs, list) else [outputs]:
            choice = out["choices"][0]
            gen = Generation(choice["text"])
            gen.generation_info = choice.get(_INFO)
            res.append(gen)
        return res
