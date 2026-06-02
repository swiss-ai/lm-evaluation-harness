"""End-to-end test for the ``multi_turn_generate`` rollout driver.

Drives ``lm_eval.evaluator.run_multi_turn_rollout`` with a stub model
and a stub task to verify:

  * cross-doc batching at each turn (one ``generate_until`` call per pass);
  * early-exit drops failed docs from later passes;
  * ``next_user_turn -> None`` ends the rollout for that doc cleanly;
  * the resulting ``task._instances`` list is in ``(doc_id, idx)`` order
    so the existing post-process loop's grouping + sort works.

The test does NOT exercise filters, sample logging, or process_results;
those are in the standard post-process path which we already cover via
the existing test suite for single-shot tasks.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from lm_eval.api.multi_turn import MultiTurnState
from lm_eval.api.utils import Message
from lm_eval.evaluator import run_multi_turn_rollout


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubLM:
    """Minimal LM-like object used by the rollout driver.

    Records every call so tests can assert pass count, batch sizes, and
    prompt content. Returns a deterministic per-call response that
    encodes both the doc id and the turn index, so we can check the
    routing of responses back to the right docs/turns.
    """

    def __init__(self):
        self.rank = 0
        self.world_size = 1
        self.calls: list[list[Any]] = []  # one entry per generate_until call

    def apply_chat_template(self, messages, add_generation_prompt: bool = True):
        # Render to a simple delimited string for inspection.
        return "|".join(f"{m['role']}:{m['content']}" for m in messages)

    def generate_until(self, requests):
        self.calls.append(list(requests))
        # Echo: response_<doc_id>_<turn_idx> — based on the most recent
        # 'user' chunk in the rendered prompt.
        responses = []
        for inst in requests:
            # doc_id comes from the Instance metadata; turn_idx is inst.idx
            doc_id = inst.doc_id
            responses.append(f"resp_{doc_id}_t{inst.idx}")
        return responses


class _StubTask:
    """Minimal task stub matching the surface the driver consults."""

    OUTPUT_TYPE = "multi_turn_generate"

    def __init__(self, docs: list[dict], *, fail_at: dict[int, int] | None = None):
        """
        Args:
            docs: list of ``{"id": int, "n_turns": int}`` dicts.
            fail_at: optional ``{doc_id: turn_idx}`` — if set, ``should_stop``
                returns True after the assistant response for that turn is
                appended for that doc.
        """
        self._docs = docs
        self._fail_at = fail_at or {}
        self.config = SimpleNamespace(task="stub_task", generation_kwargs={}, repeats=1)
        self._instances: list = []
        self._multi_turn_states: dict[int, MultiTurnState] = {
            doc["id"]: MultiTurnState(
                doc=doc,
                history=[Message("system", f"sys_{doc['id']}")],
                done=False,
            )
            for doc in self._docs
        }

    def next_user_turn(self, doc, history, turn_idx):
        if turn_idx < doc["n_turns"]:
            return f"user_{doc['id']}_t{turn_idx}"
        return None

    def should_stop(self, doc, history, turn_idx):
        target_turn = self._fail_at.get(doc["id"])
        return target_turn is not None and turn_idx == target_turn


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def task_output_factory():
    def make(task):
        return SimpleNamespace(task=task, task_name="stub_task")

    return make


def test_driver_handles_single_turn_docs(task_output_factory):
    """Docs with n_turns=1 cause exactly one generate_until pass, then
    next_user_turn returns None and the rollout ends.

    """
    docs = [{"id": 0, "n_turns": 1}, {"id": 1, "n_turns": 1}]
    task = _StubTask(docs)
    lm = _StubLM()

    run_multi_turn_rollout(lm, [task_output_factory(task)])

    assert len(lm.calls) == 1, "exactly one batched call for two single-turn docs"
    assert {inst.doc_id for inst in lm.calls[0]} == {0, 1}
    # task._instances accumulated, one per doc
    assert sorted((i.doc_id, i.idx) for i in task._instances) == [(0, 0), (1, 0)]
    # Each Instance carries the model response
    by_doc = {i.doc_id: i for i in task._instances}
    assert by_doc[0].resps == ["resp_0_t0"]
    assert by_doc[1].resps == ["resp_1_t0"]


def test_driver_batches_per_turn_with_mixed_turn_counts(task_output_factory):
    """Multi-turn docs are batched within each turn; later passes shrink
    as shorter docs drop out.

    """
    docs = [
        {"id": 0, "n_turns": 1},
        {"id": 1, "n_turns": 3},
        {"id": 2, "n_turns": 2},
    ]
    task = _StubTask(docs)
    lm = _StubLM()

    run_multi_turn_rollout(lm, [task_output_factory(task)])

    # Pass 0: all 3 docs (turn_idx=0 for all)
    assert len(lm.calls) == 3
    assert {i.doc_id for i in lm.calls[0]} == {0, 1, 2}
    # Pass 1: only docs that still have a queued user turn (1, 2)
    assert {i.doc_id for i in lm.calls[1]} == {1, 2}
    # Pass 2: only doc 1 still has a queued user turn
    assert {i.doc_id for i in lm.calls[2]} == {1}

    by_key = {(i.doc_id, i.idx): i for i in task._instances}
    # Doc 0: turns [0]; Doc 1: turns [0,1,2]; Doc 2: turns [0,1].
    assert set(by_key) == {(0, 0), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)}


def test_driver_early_exits_on_should_stop(task_output_factory):
    """Doc 2 fails should_stop after turn 0 and is dropped from pass 1."""
    docs = [
        {"id": 0, "n_turns": 1},
        {"id": 1, "n_turns": 3},
        {"id": 2, "n_turns": 3},
    ]
    task = _StubTask(docs, fail_at={2: 0})
    lm = _StubLM()

    run_multi_turn_rollout(lm, [task_output_factory(task)])

    # Pass 0: all 3 docs
    assert {i.doc_id for i in lm.calls[0]} == {0, 1, 2}
    # Pass 1: doc 0 finished (n_turns=1), doc 2 early-exited
    assert {i.doc_id for i in lm.calls[1]} == {1}
    # Pass 2: doc 1 third turn
    assert {i.doc_id for i in lm.calls[2]} == {1}

    by_doc = {i.doc_id: [] for i in task._instances}
    for i in task._instances:
        by_doc[i.doc_id].append(i.idx)
    assert sorted(by_doc[0]) == [0]  # 1 turn
    assert sorted(by_doc[1]) == [0, 1, 2]  # full 3 turns
    assert sorted(by_doc[2]) == [0]  # early-exited after turn 0


def test_driver_returns_immediately_on_empty(task_output_factory):
    """Empty task list = no work, no model calls."""
    lm = _StubLM()
    run_multi_turn_rollout(lm, [])
    assert lm.calls == []


def test_driver_cross_task_batching(task_output_factory):
    """Two multi-turn tasks share a single batched call per turn."""
    task_a = _StubTask([{"id": 10, "n_turns": 2}])
    task_b = _StubTask([{"id": 20, "n_turns": 2}])
    lm = _StubLM()

    run_multi_turn_rollout(
        lm, [task_output_factory(task_a), task_output_factory(task_b)]
    )

    assert len(lm.calls) == 2
    # Both tasks share each batch
    assert {i.doc_id for i in lm.calls[0]} == {10, 20}
    assert {i.doc_id for i in lm.calls[1]} == {10, 20}
    # Instances ended up on the right task lists
    assert {i.doc_id for i in task_a._instances} == {10}
    assert {i.doc_id for i in task_b._instances} == {20}


def test_instance_idx_matches_turn_idx(task_output_factory):
    """Instance.idx encodes the turn index, used by the post-process
    sort. Verify per-doc ordering is contiguous from 0.

    """
    docs = [{"id": 0, "n_turns": 4}]
    task = _StubTask(docs)
    lm = _StubLM()

    run_multi_turn_rollout(lm, [task_output_factory(task)])

    sorted_idx = sorted(i.idx for i in task._instances)
    assert sorted_idx == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# Higher-fidelity assertions added per reviewer feedback
# ---------------------------------------------------------------------------


class _RecordingStubLM(_StubLM):
    """Variant that records ``apply_chat_template`` inputs in addition to
    ``generate_until`` calls. Used to verify the driver passes the right
    history shape and `add_generation_prompt=True` kwarg at the chat
    template seam.

    """

    def __init__(self):
        super().__init__()
        self.template_calls: list[tuple[list[dict], bool]] = []

    def apply_chat_template(self, messages, add_generation_prompt: bool = True):
        # Snapshot the input so a mutation that swaps history in place
        # between turns is visible.
        self.template_calls.append(([dict(m) for m in messages], add_generation_prompt))
        return super().apply_chat_template(messages, add_generation_prompt)


def test_apply_chat_template_invoked_with_growing_history_and_gen_prompt(
    task_output_factory,
):
    """Each turn's chat-template call sees the conversation through
    THAT turn's user message (history grows by 2 per turn between calls)
    and ``add_generation_prompt=True``.
    """
    task = _StubTask([{"id": 0, "n_turns": 3}])
    lm = _RecordingStubLM()

    run_multi_turn_rollout(lm, [task_output_factory(task)])

    # 3 generate calls → 3 template renders
    assert len(lm.template_calls) == 3
    for messages, add_gp in lm.template_calls:
        assert add_gp is True, "add_generation_prompt must be True for generation"
        # Last message in each render is the user turn the model
        # is asked to answer.
        assert messages[-1]["role"] == "user"

    # History grows by 2 between calls (one assistant + one user appended
    # per turn).
    lengths = [len(m) for m, _ in lm.template_calls]
    assert lengths == [2, 4, 6], (
        f"history lengths per turn should be 2/4/6 (sys+user, +asst+user, "
        f"+asst+user); got {lengths}"
    )


class _StubTaskWithStopRecorder(_StubTask):
    """Stub that records the (turn_idx, history-length, last-role) tuple
    at the moment ``should_stop`` is called. Lets us assert the driver
    calls it AFTER appending the assistant response, never before.
    """

    def __init__(self, docs, *, fail_at=None):
        super().__init__(docs, fail_at=fail_at)
        self.stop_calls: list[tuple[int, int, str]] = []

    def should_stop(self, doc, history, turn_idx):
        self.stop_calls.append(
            (turn_idx, len(history), history[-1].role if history else "")
        )
        return super().should_stop(doc, history, turn_idx)


def test_should_stop_called_after_assistant_append(task_output_factory):
    """`should_stop` runs only after the model's assistant response was
    appended for that turn — never before, never on a stale history.
    """
    task = _StubTaskWithStopRecorder([{"id": 0, "n_turns": 2}])
    lm = _StubLM()

    run_multi_turn_rollout(lm, [task_output_factory(task)])

    # Two assistant turns → two should_stop calls, both at the moment
    # the freshly-appended assistant turn is the LAST message.
    assert len(task.stop_calls) == 2
    for turn_idx, _hist_len, last_role in task.stop_calls:
        assert last_role == "assistant", (
            f"should_stop must run after assistant append; got last_role={last_role!r} "
            f"at turn_idx={turn_idx}"
        )
    # Per-turn history grows: initial=1 (system), +2 per turn = 3, 5.
    assert [hist_len for _, hist_len, _ in task.stop_calls] == [3, 5]


def test_driver_raises_loudly_when_lm_missing_chat_template(task_output_factory):
    """Missing `apply_chat_template` on the LM is a loud RuntimeError, not
    a silent no-op (multi-turn requires it for the system prompt to land
    in the right role).

    """
    task = _StubTask([{"id": 0, "n_turns": 1}])

    class _NoTemplateLM:
        """A standalone LM-like object lacking ``apply_chat_template``.

        Defines only the attributes the driver consults BEFORE the
        ``hasattr`` guard so the guard is the actual failure point.

        """

        rank = 0
        world_size = 1

    lm = _NoTemplateLM()
    assert not hasattr(lm, "apply_chat_template")

    with pytest.raises(RuntimeError, match="apply_chat_template"):
        run_multi_turn_rollout(lm, [task_output_factory(task)])


def test_driver_raises_on_distributed_world_size(task_output_factory):
    """Distributed multi-rank runs are not yet supported; the driver
    must hard-error rather than silently deadlock on collective ops.
    """
    task = _StubTask([{"id": 0, "n_turns": 1}])
    lm = _StubLM()
    lm.world_size = 2  # simulate accelerate-launch with 2 processes

    with pytest.raises(NotImplementedError, match="world_size"):
        run_multi_turn_rollout(lm, [task_output_factory(task)])
