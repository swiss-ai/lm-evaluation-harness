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
    # Per-response generation info measured on the raw text before any thinking-strip;
    # one dict per response (mirroring `resps`), empty otherwise. Holds length +
    # thinking-format keys, aggregated by `evaluator_utils.promote_generation_info_metrics`.
    length_info: list = field(default_factory=list)

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
