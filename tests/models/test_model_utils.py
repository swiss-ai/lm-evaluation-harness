import pytest

from lm_eval.models.utils import (
    check_system_boilerplate,
    get_template_special_tokens,
    maybe_truncate,
    normalize_gen_kwargs,
    strip_system_boilerplate_from_template,
    truncate_tokens,
)


class TestTruncateTokens:
    def test_left(self):
        tokens = [1, 2, 3, 4, 5]
        assert truncate_tokens(tokens, 3, side="left") == [3, 4, 5]

    def test_right(self):
        tokens = [1, 2, 3, 4, 5]
        assert truncate_tokens(tokens, 3, side="right") == [1, 2, 3]

    def test_middle(self):
        tokens = [1, 2, 3, 4, 5]
        # max_length=3: left_length=1, right_length=2 -> [1] + [4, 5]
        assert truncate_tokens(tokens, 3, side="middle") == [1, 4, 5]

    def test_middle_even(self):
        tokens = [1, 2, 3, 4, 5, 6]
        # max_length=4: left_length=2, right_length=2 -> [1, 2] + [5, 6]
        assert truncate_tokens(tokens, 4, side="middle") == [1, 2, 5, 6]

    def test_no_truncation_needed(self):
        tokens = [1, 2, 3]
        assert truncate_tokens(tokens, 5, side="left") == [1, 2, 3]

    def test_unknown_strategy(self):
        with pytest.raises(ValueError) as execinfo:
            truncate_tokens([1, 2, 3], 2, side="unknown")  # type: ignore
        assert "Unknown truncation side" in str(execinfo.value)


class TestMaybeTruncate:
    """Tests for maybe_truncate with different truncation strategies."""

    # Case 1: Everything fits
    def test_case1_no_truncation(self):
        tokens = [1, 2, 3, 4, 5]
        result_tokens, result_gen = maybe_truncate(
            tokens, max_gen_toks=5, max_model_len=10
        )
        assert result_tokens == [1, 2, 3, 4, 5]
        assert result_gen == 5

    def test_case1_no_truncation_with_adjust(self):
        tokens = [1, 2, 3, 4, 5]
        result_tokens, result_gen = maybe_truncate(
            tokens, max_gen_toks=5, max_model_len=10, shrink_gen_toks=True
        )
        assert result_tokens == [1, 2, 3, 4, 5]
        assert result_gen == 5

    # Case 2: shrink_gen_toks=False — truncate prompt to max_len - max_gen_toks, keep max_gen_toks
    def test_case2_truncate_prompt_no_adjust(self):
        tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result_tokens, result_gen = maybe_truncate(
            tokens, max_gen_toks=5, max_model_len=6, shrink_gen_toks=False
        )
        # Left-truncates prompt to max_len - max_gen_toks = 1, keeps max_gen_toks=5
        assert result_tokens == [10]
        assert result_gen == 5

    def test_case2_no_adjust_is_default(self):
        tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result_tokens, result_gen = maybe_truncate(
            tokens, max_gen_toks=5, max_model_len=6
        )
        assert result_tokens == [10]
        assert result_gen == 5

    def test_case2_prompt_fits_but_gen_too_large_no_adjust(self):
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        result_tokens, result_gen = maybe_truncate(
            tokens, max_gen_toks=3, max_model_len=8, shrink_gen_toks=False
        )
        # Prompt (8) + gen (3) > max_len (8), truncate prompt to 8 - 3 = 5
        assert result_tokens == [4, 5, 6, 7, 8]
        assert result_gen == 3

    # Case 3: adjust_gen_toks=True — reduce gen toks if prompt fits
    def test_case3_reduce_gen_toks(self):
        tokens = [1, 2, 3, 4, 5]
        result_tokens, result_gen = maybe_truncate(
            tokens, max_gen_toks=10, max_model_len=8, shrink_gen_toks=True
        )
        assert result_tokens == [1, 2, 3, 4, 5]
        assert result_gen == 3

    # Case 4: adjust_gen_toks=True — truncate prompt with strategy
    def test_case4_truncate_left(self):
        tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result_tokens, result_gen = maybe_truncate(
            tokens,
            max_gen_toks=5,
            max_model_len=6,
            min_gen_toks=2,
            side="left",
            shrink_gen_toks=True,
        )
        assert result_tokens == [7, 8, 9, 10]
        assert result_gen == 2

    def test_case4_truncate_right(self):
        tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result_tokens, result_gen = maybe_truncate(
            tokens,
            max_gen_toks=5,
            max_model_len=6,
            min_gen_toks=2,
            side="right",
            shrink_gen_toks=True,
        )
        assert result_tokens == [1, 2, 3, 4]
        assert result_gen == 2

    def test_case4_truncate_middle(self):
        tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result_tokens, result_gen = maybe_truncate(
            tokens,
            max_gen_toks=5,
            max_model_len=6,
            min_gen_toks=2,
            side="middle",
            shrink_gen_toks=True,
        )
        # max_ctx_len=4: left=2, right=2 -> [1, 2] + [9, 10]
        assert result_tokens == [1, 2, 9, 10]
        assert result_gen == 2

    def test_case4_default_strategy_is_left(self):
        tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result_tokens, result_gen = maybe_truncate(
            tokens,
            max_gen_toks=5,
            max_model_len=6,
            min_gen_toks=2,
            shrink_gen_toks=True,
        )
        assert result_tokens == [7, 8, 9, 10]
        assert result_gen == 2

    def test_min_gen_toks_zero_reduces_to_zero(self):
        # Prompt exactly fills context window, gen toks reduced to 0
        tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result_tokens, result_gen = maybe_truncate(
            tokens,
            max_gen_toks=5,
            max_model_len=10,
            min_gen_toks=0,
            shrink_gen_toks=True,
        )
        assert result_tokens == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert result_gen == 0

    def test_min_gen_toks_zero_truncates_prompt(self):
        # Prompt exceeds max_len, but min_gen_toks=0 means all space goes to prompt
        tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result_tokens, result_gen = maybe_truncate(
            tokens,
            max_gen_toks=5,
            max_model_len=8,
            min_gen_toks=0,
            shrink_gen_toks=True,
        )
        # max_ctx_len = 8 - 0 = 8, left-truncate to 8
        assert result_tokens == [3, 4, 5, 6, 7, 8, 9, 10]
        assert result_gen == 0

    def test_raises_when_max_len_too_small(self):
        tokens = [1, 2, 3, 4, 5]
        with pytest.raises(ValueError):
            maybe_truncate(
                tokens,
                max_gen_toks=5,
                max_model_len=2,
                min_gen_toks=3,
                shrink_gen_toks=True,
            )


class TestNormalizeGenKwargs:
    """Tests for normalize_gen_kwargs utility function."""

    # --- until normalization ---

    def test_until_string_converted_to_list(self):
        result = normalize_gen_kwargs({"until": "stop"})
        assert result["until"] == ["stop"]

    def test_until_list_passed_through(self):
        result = normalize_gen_kwargs({"until": ["stop1", "stop2"]})
        assert result["until"] == ["stop1", "stop2"]

    def test_until_missing_defaults_to_empty_list(self):
        result = normalize_gen_kwargs({})
        assert result["until"] == []

    # --- max token aliases ---

    def test_max_gen_toks_used_directly(self):
        result = normalize_gen_kwargs({"max_gen_toks": 100})
        assert result["max_gen_toks"] == 100

    def test_max_new_tokens_converted(self):
        result = normalize_gen_kwargs({"max_new_tokens": 150})
        assert result["max_gen_toks"] == 150

    def test_max_tokens_converted(self):
        result = normalize_gen_kwargs({"max_tokens": 200})
        assert result["max_gen_toks"] == 200

    def test_max_completion_tokens_converted(self):
        result = normalize_gen_kwargs({"max_completion_tokens": 250})
        assert result["max_gen_toks"] == 250

    def test_default_max_gen_toks_when_none_provided(self):
        result = normalize_gen_kwargs({})
        assert result["max_gen_toks"] == 256

    def test_custom_default_max_gen_toks(self):
        result = normalize_gen_kwargs({}, default_max_gen_toks=512)
        assert result["max_gen_toks"] == 512

    def test_max_token_priority_max_gen_toks_first(self):
        result = normalize_gen_kwargs(
            {
                "max_gen_toks": 100,
                "max_new_tokens": 200,
                "max_tokens": 300,
            }
        )
        assert result["max_gen_toks"] == 100

    def test_max_token_priority_max_new_tokens_second(self):
        result = normalize_gen_kwargs(
            {
                "max_new_tokens": 200,
                "max_tokens": 300,
                "max_completion_tokens": 400,
            }
        )
        assert result["max_gen_toks"] == 200

    def test_max_token_priority_max_tokens_third(self):
        result = normalize_gen_kwargs(
            {
                "max_tokens": 300,
                "max_completion_tokens": 400,
            }
        )
        assert result["max_gen_toks"] == 300

    # --- do_sample and temperature interaction ---

    def test_do_sample_none_temperature_zero_sets_do_sample_false(self):
        result = normalize_gen_kwargs({"temperature": 0.0})
        assert result["do_sample"] is False

    def test_do_sample_none_temperature_positive_sets_do_sample_true(self):
        result = normalize_gen_kwargs({"temperature": 0.7})
        assert result["do_sample"] is True

    def test_do_sample_false_sets_temperature_zero(self):
        result = normalize_gen_kwargs({"do_sample": False})
        assert result["temperature"] == 0.0

    def test_do_sample_false_temperature_positive_forces_temperature_zero(self):
        result = normalize_gen_kwargs({"do_sample": False, "temperature": 0.8})
        assert result["temperature"] == 0.0

    def test_do_sample_true_temperature_positive_preserved(self):
        result = normalize_gen_kwargs({"do_sample": True, "temperature": 0.9})
        assert result["do_sample"] is True
        assert result["temperature"] == 0.9

    def test_do_sample_true_temperature_zero_preserved(self):
        result = normalize_gen_kwargs({"do_sample": True, "temperature": 0.0})
        assert result["do_sample"] is True
        assert result["temperature"] == 0.0

    # --- other behaviors ---

    def test_extra_kwargs_passed_through(self):
        result = normalize_gen_kwargs(
            {
                "top_p": 0.95,
                "top_k": 50,
                "repetition_penalty": 1.1,
            }
        )
        assert result["top_p"] == 0.95  # type: ignore
        assert result["top_k"] == 50  # type: ignore
        assert result["repetition_penalty"] == 1.1  # type: ignore

    def test_original_dict_not_mutated(self):
        original = {"until": "stop", "max_gen_toks": 100, "temperature": 0.5}
        original_copy = original.copy()
        normalize_gen_kwargs(original)
        assert original == original_copy


def _render_fn(header, *, drop_system=False):
    """Build a render_fn that emits ``header`` before the system content.

    Mimics a chat template: the system content lands right after ``header``.
    If ``drop_system`` the system message is folded away (marker disappears).
    """

    def fn(messages):
        sys = next((m for m in messages if m["role"] == "system"), None)
        if sys is None or drop_system:
            return "<|bos|><|eot|>user\nHi<|eot|>"
        return header + sys["content"] + "<|eot|>user\nHi<|eot|>"

    return fn


# Structural special-token sets for the various template families.
_LLAMA = ["<|bos|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot|>"]
_CHATML = ["<|im_start|>", "<|im_end|>", "<|eot|>"]
_COMMANDR = ["<|bos|>", "<|SYSTEM_TOKEN|>", "<|END_OF_TURN_TOKEN|>", "<|eot|>"]


class TestCheckSystemBoilerplate:
    """The probe must pass clean templates and raise on injected text, across
    special-token and plain-text families (system prompt must be authoritative).
    """

    @pytest.mark.parametrize(
        "header,special_tokens",
        [
            # multi-token strip + literal role-word removal (Llama)
            ("<|bos|><|start_header_id|>system<|end_header_id|>\n\n", _LLAMA),
            # role word lives *inside* a special token, so no literal label
            # remains after the token strip (Command-R)
            ("<|bos|><|SYSTEM_TOKEN|>", _COMMANDR),
            # plain-text family (no special tokens, role-word removal fires)
            ("### System:\n", []),
        ],
    )
    def test_clean_templates_pass(self, header, special_tokens):
        # should not raise
        check_system_boilerplate(_render_fn(header), special_tokens=special_tokens)

    @pytest.mark.parametrize(
        "header,special_tokens",
        [
            # Llama 3.x date boilerplate
            (
                "<|bos|><|start_header_id|>system<|end_header_id|>\n\n"
                "Cutting Knowledge Date: December 2023\n"
                "Today Date: 26 Jul 2024\n\n",
                _LLAMA,
            ),
            # plain-text preamble
            ("### System:\nYou are a helpful assistant.\n", []),
            # injection where the role word lives inside a special token
            ("<|bos|><|SYSTEM_TOKEN|>Hard-coded preamble\n", _COMMANDR),
            # injection placed *before* the role declaration
            (
                "<|bos|>Sneaky preamble.<|start_header_id|>system<|end_header_id|>\n\n",
                _LLAMA,
            ),
        ],
    )
    def test_injected_templates_raise(self, header, special_tokens):
        with pytest.raises(RuntimeError, match="not authoritative"):
            check_system_boilerplate(_render_fn(header), special_tokens=special_tokens)

    def test_fail_open_when_render_errors(self):
        def boom(messages):
            raise ValueError("no system role")

        # must not propagate — can't assess, so silently passes
        check_system_boilerplate(boom, special_tokens=[])

    def test_fail_open_when_marker_dropped(self):
        # template folds the system message away → marker absent → can't assess
        check_system_boilerplate(
            _render_fn("<|im_start|>system\n", drop_system=True),
            special_tokens=_CHATML,
        )

    def test_purely_numeric_injection_passes(self):
        # documented limitation: only alphabetic chars are treated as injection
        check_system_boilerplate(
            _render_fn("<|im_start|>system\n2024-12-01\n"), special_tokens=_CHATML
        )

    def test_error_message_contains_injected_text(self):
        with pytest.raises(RuntimeError) as exc:
            check_system_boilerplate(
                _render_fn("<|im_start|>system\nLeaked preamble\n"),
                special_tokens=_CHATML,
            )
        assert "Leaked preamble" in str(exc.value)

    def test_special_tokens_stripped_longest_first(self):
        # When a shorter token is a prefix of a longer one, stripping
        # shortest-first would leave the longer token's tail ("X|>") as
        # alphabetic debris and false-raise. Longest-first removes the whole
        # token cleanly. This pins the `sorted(..., reverse=True)` ordering.
        header = "<|HDRX|>system\n"
        check_system_boilerplate(
            _render_fn(header), special_tokens=["<|HDR", "<|HDRX|>"]
        )


class TestGetTemplateSpecialTokens:
    def test_union_of_special_and_added(self):
        class Tok:
            all_special_tokens = ["<s>", "</s>"]

            def get_added_vocab(self):
                return {"<|start_header_id|>": 1, "</s>": 2}

        assert get_template_special_tokens(Tok()) == {
            "<s>",
            "</s>",
            "<|start_header_id|>",
        }

    def test_missing_get_added_vocab_is_tolerated(self):
        class Tok:
            all_special_tokens = ["<s>"]

        assert get_template_special_tokens(Tok()) == {"<s>"}

    def test_get_added_vocab_error_is_tolerated(self):
        class Tok:
            all_special_tokens = ["<s>"]

            def get_added_vocab(self):
                raise RuntimeError("unsupported tokenizer")

        assert get_template_special_tokens(Tok()) == {"<s>"}


class TestStripSystemBoilerplate:
    LLAMA_TEMPLATE = (
        '{{- bos_token }}{{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}'
        "{%- if not date_string is defined %}\n"
        '    {%- set date_string = "26 Jul 2024" %}\n'
        "{%- endif %}\n"
        '{{- "Cutting Knowledge Date: December 2023\\n" }}'
        '{{- "Today Date: " + date_string + "\\n\\n" }}'
        "{{- system_message }}"
    )

    def test_strips_llama_boilerplate(self):
        cleaned = strip_system_boilerplate_from_template(self.LLAMA_TEMPLATE)
        assert cleaned is not None
        assert "Cutting Knowledge Date" not in cleaned
        assert "Today Date" not in cleaned
        assert "date_string is defined" not in cleaned
        # legitimate parts survive
        assert "system_message" in cleaned
        assert "<|start_header_id|>system<|end_header_id|>" in cleaned

    # Llama 3.2+ nests if/else/endif inside the date_string block; the
    # nesting-depth scanner must find the *outer* endif (a naive non-greedy
    # regex would stop at the inner one and leave a dangling tag).
    NESTED_TEMPLATE = (
        '{{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}'
        "{%- if not date_string is defined %}\n"
        "    {%- if tools is defined %}\n"
        '        {%- set date_string = "26 Jul 2024" %}\n'
        "    {%- else %}\n"
        '        {%- set date_string = "26 Jul 2024" %}\n'
        "    {%- endif %}\n"
        "{%- endif %}\n"
        '{{- "Cutting Knowledge Date: December 2023\\n" }}'
        '{{- "Today Date: " + date_string + "\\n\\n" }}'
        "{{- system_message }}"
    )

    def test_strips_nested_date_block(self):
        cleaned = strip_system_boilerplate_from_template(self.NESTED_TEMPLATE)
        assert cleaned is not None
        # the whole outer block (including the nested if) is removed
        assert "date_string is defined" not in cleaned
        assert "tools is defined" not in cleaned
        assert "Cutting Knowledge" not in cleaned
        assert "system_message" in cleaned
        # no dangling Jinja tag left behind
        assert cleaned.count("{%- if") == cleaned.count("{%- endif")

    def test_clean_template_returns_none(self):
        clean = '{{- "<|im_start|>system\\n" + system_message }}'
        assert strip_system_boilerplate_from_template(clean) is None

    def test_none_input_returns_none(self):
        assert strip_system_boilerplate_from_template(None) is None
