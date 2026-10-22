from .utils import parser


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_parser_keeps_only_outermost_changed_items(tmp_path):
    # top group -> subgroup -> task, all changed, plus an unrelated task
    files = [
        _write(tmp_path / "_top.yaml", "group: top\ntask:\n  - sub\n"),
        _write(tmp_path / "sub/_sub.yaml", "group: sub\ntask:\n  - leaf\n"),
        _write(tmp_path / "sub/leaf.yaml", "task: leaf\n"),
        _write(tmp_path / "other.yaml", "task: other\n"),
    ]

    # loading "top" already loads "sub" and "leaf"
    assert parser(files) == ["other", "top"]


def test_parser_keeps_members_of_unchanged_groups(tmp_path):
    files = [_write(tmp_path / "leaf.yaml", "task: leaf\n")]

    assert parser(files) == ["leaf"]
