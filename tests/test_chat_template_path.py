import pytest

from lm_eval.models.utils import maybe_strip_system_boilerplate


def test_chat_template_path_loads_template(tmp_path):
    template = tmp_path / "template.jinja"
    template.write_text("{{ bos_token }}{{ messages[0].content }}", encoding="utf-8")

    args = maybe_strip_system_boilerplate(
        chat_template_source=None,
        chat_template_args=None,
        strip=False,
        chat_template_path=str(template),
    )

    assert args["chat_template"] == "{{ bos_token }}{{ messages[0].content }}"


def test_nested_chat_template_path_loads_template(tmp_path):
    template = tmp_path / "template.jinja"
    template.write_text("nested-template", encoding="utf-8")

    args = maybe_strip_system_boilerplate(
        chat_template_source=None,
        chat_template_args={
            "chat_template_path": str(template),
            "enable_thinking": False,
        },
        strip=False,
    )

    assert args == {"chat_template": "nested-template", "enable_thinking": False}


def test_chat_template_path_rejects_raw_template_conflict(tmp_path):
    template = tmp_path / "template.jinja"
    template.write_text("from-file", encoding="utf-8")

    with pytest.raises(ValueError, match="chat_template_args.chat_template"):
        maybe_strip_system_boilerplate(
            chat_template_source=None,
            chat_template_args={"chat_template": "inline"},
            strip=False,
            chat_template_path=str(template),
        )


def test_chat_template_path_rejects_duplicate_path_sources(tmp_path):
    template = tmp_path / "template.jinja"
    template.write_text("from-file", encoding="utf-8")

    with pytest.raises(ValueError, match="top-level model arg"):
        maybe_strip_system_boilerplate(
            chat_template_source=None,
            chat_template_args={"chat_template_path": str(template)},
            strip=False,
            chat_template_path=str(template),
        )


def test_empty_chat_template_path_is_ignored():
    args = maybe_strip_system_boilerplate(
        chat_template_source=None,
        chat_template_args=None,
        strip=False,
        chat_template_path="",
    )

    assert args == {}


def test_nested_empty_chat_template_path_is_ignored():
    args = maybe_strip_system_boilerplate(
        chat_template_source=None,
        chat_template_args={"chat_template_path": "", "enable_thinking": False},
        strip=False,
    )

    assert args == {"enable_thinking": False}
