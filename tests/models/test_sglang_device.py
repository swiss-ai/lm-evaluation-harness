import pytest

from lm_eval.models.sglang_causallms import _sglang_device_type


@pytest.mark.parametrize(
    ("device", "expected"),
    [("cuda:0", "cuda"), ("cuda:3", "cuda"), ("cuda", "cuda"), (None, None)],
)
def test_device_index_is_dropped(device, expected):
    """SGLang only accepts a bare device type; "cuda:0" is the CLI default."""
    assert _sglang_device_type(device) == expected
