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
