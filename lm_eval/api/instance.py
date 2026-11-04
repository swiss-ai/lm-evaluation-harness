from dataclasses import dataclass, field
from typing import Literal


OutputType = Literal[
    "loglikelihood",
    "loglikelihood_rolling",
    "generate_until",
    "multiple_choice",
    "multi_turn_generate",
]


@dataclass
class Instance:
    request_type: OutputType
    doc: dict
    arguments: tuple
    idx: int
    metadata: tuple[str | None, int | None, int | None] = field(
        default_factory=lambda: (None, None, None)
    )
    resps: list = field(default_factory=list)
    filtered_resps: dict = field(default_factory=dict)
    # Per-response generation info measured before any thinking-strip, on the backend's
    # view of the response (hf bounds it to exclude batched over-generation); one dict per
    # response (mirroring `resps`), empty otherwise. Holds length, thinking-format and
    # `generation_*` keys, aggregated by `evaluator_utils.promote_generation_info_metrics`
    # (non-numeric backend values such as `apertus2_*` strings are kept, not promoted).
    length_info: list = field(default_factory=list)
    # Which of a multi-turn document's `repeats` episodes sent this request
    # (evaluator.run_multi_turn_rollout); 0 otherwise.
    episode: int = 0

    # initialized after init
    task_name: str | None = None
    doc_id: int | None = None
    repeats: int | None = None

    def __post_init__(self) -> None:
        # unpack metadata field
        self.task_name, self.doc_id, self.repeats = self.metadata

    @property
    def args(self):
        """
        Returns (string,) where `string` is the string to calculate loglikelihood over
        """
        return (
            self.arguments if isinstance(self.arguments, tuple) else (self.arguments,)
        )
