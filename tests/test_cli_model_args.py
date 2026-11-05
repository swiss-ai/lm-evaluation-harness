from lm_eval._cli.utils import key_val_to_dict


def test_key_val_to_dict_preserves_quoted_comma_in_extra_args():
    args = (
        "load=/checkpoint,"
        "extra_args=--window-size '(1024,0)' --window-attn-skip-freq 2,"
        "devices=32"
    )

    assert key_val_to_dict(args) == {
        "load": "/checkpoint",
        "extra_args": "--window-size (1024,0) --window-attn-skip-freq 2",
        "devices": 32,
    }


def test_key_val_to_dict_keeps_quotes_inside_literals():
    # e.g. evals-post-train's `--reasoning-effort medium` for gpt-oss
    args = 'pretrained=m,chat_template_args={"reasoning_effort":"medium"},enable_thinking=True'

    assert key_val_to_dict(args) == {
        "pretrained": "m",
        "chat_template_args": {"reasoning_effort": "medium"},
        "enable_thinking": True,
    }


def test_key_val_to_dict_reads_quoted_lists_and_strings():
    assert key_val_to_dict('stop=["\\n\\n"]') == {"stop": ["\n\n"]}
    assert key_val_to_dict("stop=['a,b', 'c'],top_p=0.95") == {
        "stop": ["a,b", "c"],
        "top_p": 0.95,
    }
    assert key_val_to_dict('name="x, y"') == {"name": "x, y"}


def test_key_val_to_dict_plain_values_unchanged():
    assert key_val_to_dict(
        "think_end_token=<|channel|>final<|message|>,max_gen_toks=8000,dtype=auto"
    ) == {
        "think_end_token": "<|channel|>final<|message|>",
        "max_gen_toks": 8000,
        "dtype": "auto",
    }
    # an unbalanced quote is kept as it is
    assert key_val_to_dict("note=it's") == {"note": "it's"}


def test_key_val_to_dict_apostrophes_and_stray_brackets_do_not_swallow_keys():
    assert key_val_to_dict("system_instruction=Don't guess,max_gen_toks=5") == {
        "system_instruction": "Don't guess",
        "max_gen_toks": 5,
    }
    assert key_val_to_dict("extra_args=--foo (bar,devices=4") == {
        "extra_args": "--foo (bar",
        "devices": 4,
    }


def test_key_val_to_dict_escaped_quote_inside_quotes():
    assert key_val_to_dict('stop=["\\"", "x"],top_p=0.9') == {
        "stop": ['"', "x"],
        "top_p": 0.9,
    }


def test_key_val_to_dict_reads_shell_quoted_literals():
    # evals-post-train's sbatch passes the quotes backslash-escaped
    assert key_val_to_dict(
        'pretrained=m,chat_template_args={\\"reasoning_effort\\":\\"medium\\"}'
    ) == {"pretrained": "m", "chat_template_args": {"reasoning_effort": "medium"}}
    assert key_val_to_dict("""chat_template_args='{"reasoning_effort":"low"}'""") == {
        "chat_template_args": {"reasoning_effort": "low"}
    }
    # a quoted number or word stays a string
    assert key_val_to_dict("""a='5',b="x" """.strip()) == {"a": "5", "b": "x"}
