from tests.utils import parser


def test_parser_skips_yaml_without_task(tmp_path):
    template = tmp_path / "_template.yaml"
    template.write_text("output_type: generate_until\n")
    task = tmp_path / "my_task.yaml"
    task.write_text("task: my_task\n")
    assert parser([str(template), str(task)]) == ["my_task"]
