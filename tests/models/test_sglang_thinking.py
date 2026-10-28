from unittest.mock import MagicMock, patch

import pytest

from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.models.sglang_causallms import SGLangLM


@pytest.mark.parametrize(
    ("enable_thinking", "raises"), [(True, True), (False, False), (None, False)]
)
def test_loglikelihood_rejects_only_explicit_enable_thinking(enable_thinking, raises):
    """Matches vllm/hf: only an explicit True blocks loglikelihood tasks."""
    lm = MagicMock(spec=SGLangLM)
    lm.enable_thinking = enable_thinking
    request = MagicMock(spec=Instance)
    request.task_name = "arc_easy"
    with patch.object(TemplateLM, "loglikelihood", return_value=[]) as parent:
        if raises:
            with pytest.raises(ValueError, match="arc_easy"):
                SGLangLM.loglikelihood(lm, [request])
            parent.assert_not_called()
        else:
            assert SGLangLM.loglikelihood(lm, [request]) == []
            parent.assert_called_once()
