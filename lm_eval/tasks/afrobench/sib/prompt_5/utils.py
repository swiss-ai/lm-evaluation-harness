# referenced as `!function utils.weighted_f1_score` in the task configs;
# keep the import even though it looks unused (noqa guards against ruff --fix)
from lm_eval.utils import weighted_f1_score  # noqa: F401
