# referenced as `!function utils.weighted_f1_score` in the task configs;
# keep the import even though it looks unused (noqa guards against ruff --fix)
from lm_eval.utils import weighted_f1_score  # noqa: F401


def doc_to_target(doc):
    replacements = {0: "True", 1: "Neither", 2: "False"}
    return replacements[doc["label"]]
