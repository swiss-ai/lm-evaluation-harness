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


def test_driver_raises_when_distributed_lm_lacks_accelerator(task_output_factory):
    """Distributed runs require ``lm.accelerator``. Backends exposing
    ``world_size > 1`` without it (a wiring bug, since vLLM-style internal
    DP should report ``world_size == 1``) must hard-error rather than
    crash deep inside the gather helper.
    """
    task = _StubTask([{"id": 0, "n_turns": 1}])
    lm = _StubLM()
    lm.world_size = 2  # but no `accelerator` attribute

    with pytest.raises(NotImplementedError, match="accelerator"):
        run_multi_turn_rollout(lm, [task_output_factory(task)])


# ---------------------------------------------------------------------------
# Distributed (accelerate launch) — stub accelerator simulating cross-rank
# ---------------------------------------------------------------------------


class _StubAccelerator:
    """Fake ``accelerator`` mimicking a 2-rank gather.

    The driver is single-process here; this stub pretends the other rank
    exists by pulling its turn-by-turn live-doc counts from a pre-seeded
    queue (``peer_live_counts``). On each ``gather(t)`` it returns a
    1D tensor where slot 0 is THIS rank's value and slot 1 is the next
    item from the queue.

    The entry guard's ``gather`` (one boolean before the per-turn loop)
    pulls from ``peer_any_pending`` separately so the per-turn queue
    doesn't need a sentinel for it.

    What this stub DOES validate:
        * Driver-side protocol: which collectives are called, in what
          order, with what payload (live count, any-pending bool).
        * That ranks with fewer live docs pad to the global max.
        * That termination is global (driver does not break before
          every rank reports zero).
        * That ``wait_for_everyone`` follows every issued ``generate_until``.

    What this stub does NOT validate:
        * Real cross-process gather ordering (NCCL / gloo collective
          ordering bugs that only manifest when ranks are actual
          separate processes).
        * Deadlocks from one rank skipping a collective the others
          execute — single-process tests cannot model this; only a real
          ``accelerate launch -m pytest`` smoke would catch it.
        * NCCL timeout semantics, device-tensor placement on multi-GPU.
        * That ``lm.device`` is a real ``torch.device``: stubbed as CPU.

    A real-DP smoke would launch this test file under
    ``accelerate launch --num_processes 2 -m pytest tests/test_multi_turn_rollout.py``
    against the actual harness; that path is not currently wired into
    CI. See ``test_real_dp_smoke_placeholder`` below.
    """

    def __init__(
        self,
        *,
        peer_live_counts: list[int],
        peer_any_pending: bool = True,
    ):
        import collections

        self.peer_live_q = collections.deque(peer_live_counts)
        self.peer_any_pending = peer_any_pending
        self.entry_gather_calls = 0
        self.wait_calls = 0

    def gather(self, t):
        import torch

        # First gather of the run is the entry guard (boolean any-pending).
        if self.entry_gather_calls == 0:
            self.entry_gather_calls += 1
            return torch.tensor([int(t.item()), int(self.peer_any_pending)])

        # All later gathers are per-turn live-count gathers.
        peer = self.peer_live_q.popleft() if self.peer_live_q else 0
        return torch.tensor([int(t.item()), int(peer)])

    def wait_for_everyone(self):
        self.wait_calls += 1


class _DistStubLM(_StubLM):
    """``_StubLM`` posing as rank 0 of a 2-process world."""

    def __init__(self, *, peer_live_counts: list[int], peer_any_pending: bool = True):
        super().__init__()
        import torch

        self.world_size = 2
        self.rank = 0
        self.device = torch.device("cpu")
        self.accelerator = _StubAccelerator(
            peer_live_counts=peer_live_counts,
            peer_any_pending=peer_any_pending,
        )


def test_distributed_pads_to_global_max_when_local_batch_smaller(task_output_factory):
    """When this rank has fewer live docs than the peer, the driver pads
    the local batch up to the global max with the sentinel dummy and
    discards the padded responses. Real responses still route correctly.
    """
    # This rank holds 1 doc with 2 turns; peer has 3 live docs per turn.
    task = _StubTask([{"id": 0, "n_turns": 2}])
    lm = _DistStubLM(peer_live_counts=[3, 3])

    run_multi_turn_rollout(lm, [task_output_factory(task)])

    # Two turns of generate_until; each padded up to global_max=3.
    assert len(lm.calls) == 2
    for batch in lm.calls:
        assert len(batch) == 3, "rank should pad to the global max each turn"

    # Verify the padded slots actually carry the sentinel shape — not
    # a real instance accidentally cloned. The real instance is at
    # index 0 each turn; indices 1 and 2 must be the sentinel.
    for batch in lm.calls:
        for pad_inst in batch[1:]:
            assert pad_inst.idx == -1, (
                f"pad slot must use the sentinel (idx=-1); got idx={pad_inst.idx}"
            )
            assert pad_inst.arguments[0] == "<pad>", (
                f"pad slot must carry sentinel arguments; "
                f"got args={pad_inst.arguments}"
            )
            assert pad_inst.arguments[1].get("max_gen_toks") == 1, (
                "pad slot must request only 1 token to minimize wasted compute"
            )

    # Real instance accounting: this rank has exactly 2 instances
    # (one per turn for doc 0); padded slots never make it onto _instances.
    assert len(task._instances) == 2
    assert sorted((i.doc_id, i.idx) for i in task._instances) == [(0, 0), (0, 1)]

    # wait_for_everyone called once per turn that issued a generate call.
    assert lm.accelerator.wait_calls == 2


def test_distributed_terminates_globally_after_peer_finishes(task_output_factory):
    """If this rank finishes early (next_user_turn → None) but the peer
    still has docs, this rank must keep participating in collective ops
    (issuing dummy-only batches) until the peer is also done. Termination
    is global, not local.
    """
    task = _StubTask([{"id": 0, "n_turns": 1}])
    lm = _DistStubLM(peer_live_counts=[2, 1, 0])

    run_multi_turn_rollout(lm, [task_output_factory(task)])

    # Turn 0: we have 1, peer has 2 → call with batch=2 (1 real + 1 pad)
    # Turn 1: we have 0, peer has 1 → call with batch=1 (1 pad only)
    # Turn 2: we have 0, peer has 0 → break before calling.
    assert len(lm.calls) == 2
    assert len(lm.calls[0]) == 2
    assert len(lm.calls[1]) == 1
    # Only the real turn-0 instance ends up on _instances.
    assert [(i.doc_id, i.idx) for i in task._instances] == [(0, 0)]


def test_distributed_entry_guard_short_circuits_when_peer_also_empty(
    task_output_factory,
):
    """If neither rank has any docs (small dataset + many ranks), the
    entry guard returns before issuing any generate_until call.
    """
    task = _StubTask([])  # no docs on this rank
    lm = _DistStubLM(peer_live_counts=[], peer_any_pending=False)

    run_multi_turn_rollout(lm, [task_output_factory(task)])

    assert lm.calls == []
    # Entry guard collective happened exactly once.
    assert lm.accelerator.entry_gather_calls == 1


def test_distributed_entry_guard_proceeds_when_peer_has_docs(task_output_factory):
    """If this rank has no docs but the peer does, this rank must still
    enter the loop and contribute dummy batches until the peer is done.
    """
    task = _StubTask([])  # no docs on this rank
    lm = _DistStubLM(peer_live_counts=[2, 0], peer_any_pending=True)

    run_multi_turn_rollout(lm, [task_output_factory(task)])

    # Turn 0: peer=2, we=0 → padded call of size 2.
    # Turn 1: peer=0, we=0 → break.
    assert len(lm.calls) == 1
    assert len(lm.calls[0]) == 2
    assert task._instances == []


# ---------------------------------------------------------------------------
# Helper unit tests — direct coverage of `_any_rank_has_pending` and
# `_gather_max_live` so an accidental `all()`/`any()` flip or wrong-rank
# index extraction would fail an isolated test, not be masked by the
# downstream padding behaviour.
# ---------------------------------------------------------------------------


def test_any_rank_has_pending_returns_true_when_only_peer_has_docs():
    """The entry guard helper must say "proceed" if ANY rank has docs,
    even when this rank's pending list is empty.
    """
    from lm_eval.evaluator import _any_rank_has_pending

    lm = _DistStubLM(peer_live_counts=[], peer_any_pending=True)
    assert _any_rank_has_pending(lm, pending=[]) is True


def test_any_rank_has_pending_returns_false_when_no_rank_has_docs():
    from lm_eval.evaluator import _any_rank_has_pending

    lm = _DistStubLM(peer_live_counts=[], peer_any_pending=False)
    assert _any_rank_has_pending(lm, pending=[]) is False


def test_gather_max_live_returns_peer_max_when_local_zero():
    """When this rank is idle (local_n=0) but the peer has docs, the
    helper must return the peer's count — used as the global_max the
    rank pads up to.
    """
    from lm_eval.evaluator import _gather_max_live

    # First gather is the entry guard; the second is the live-count one.
    # We're calling _gather_max_live directly, so simulate the entry
    # guard already happened by burning the first stub call.
    lm = _DistStubLM(peer_live_counts=[7], peer_any_pending=True)
    lm.accelerator.entry_gather_calls = 1  # mark entry guard as already done
    assert _gather_max_live(lm, local_n=0) == 7


def test_gather_max_live_picks_local_when_local_larger():
    from lm_eval.evaluator import _gather_max_live

    lm = _DistStubLM(peer_live_counts=[3], peer_any_pending=True)
    lm.accelerator.entry_gather_calls = 1
    assert _gather_max_live(lm, local_n=5) == 5


# ---------------------------------------------------------------------------
# Asymmetric / safety-cap coverage requested by review feedback
# ---------------------------------------------------------------------------


def test_distributed_asymmetric_should_stop_keeps_collectives_aligned(
    task_output_factory,
):
    """This rank's doc early-exits at turn 1 via should_stop; the peer
    still has 2 live docs at turn 2. The driver must keep this rank
    participating in the turn-2 gather (and dispatching a padded-only
    batch) rather than breaking and stranding the peer at a deadlocked
    collective.
    """
    docs = [{"id": 0, "n_turns": 5}]
    task = _StubTask(docs, fail_at={0: 1})  # early-exit AFTER turn 1
    lm = _DistStubLM(peer_live_counts=[3, 3, 2, 0])

    run_multi_turn_rollout(lm, [task_output_factory(task)])

    # Turn 0: we have 1, peer 3 → call size 3
    # Turn 1: we have 1 (still alive), peer 3 → call size 3.
    #         After turn 1 we early-exit.
    # Turn 2: we have 0, peer 2 → call size 2 (we pad, all dummies)
    # Turn 3: we have 0, peer 0 → break before generating.
    assert len(lm.calls) == 3, (
        f"expected 3 generate_until calls (turn 0, 1, 2 with peer still alive); "
        f"got {len(lm.calls)}"
    )
    assert [len(b) for b in lm.calls] == [3, 3, 2]
    # We contributed 2 real instances (turns 0 and 1 before early-exit).
    assert sorted((i.doc_id, i.idx) for i in task._instances) == [(0, 0), (0, 1)]


class _SafetyCapStubLM(_DistStubLM):
    """Pretends the peer always has 1 live doc — forces the safety cap
    to be the only termination mechanism. Used to verify the cap break
    runs AFTER the per-turn gather, not before.
    """

    def __init__(self):
        super().__init__(peer_live_counts=[1] * 200, peer_any_pending=True)


def test_distributed_safety_cap_does_not_skip_gather(task_output_factory):
    """A task whose user-turn generator never returns None and never
    triggers should_stop would loop forever; the safety cap stops it.
    Under DP, the cap must trigger AFTER each turn's gather so a future
    refactor that diverges turn_idx across ranks cannot cause a deadlock
    where one rank breaks while another waits at the next gather.

    Witness: with the cap at 64 turns, the driver should perform 64
    gathers (one per turn that ran) before the break, NOT 0 gathers
    (which would indicate the old broken order: cap check before
    gather).
    """
    # An infinitely-turn-yielding doc:
    docs = [{"id": 0, "n_turns": 1_000_000}]
    task = _StubTask(docs)
    lm = _SafetyCapStubLM()

    run_multi_turn_rollout(lm, [task_output_factory(task)])

    # Cap is 64; the driver should have issued exactly 64 generate_until
    # calls before breaking. (Per-turn gather happens before the cap
    # check; cap check happens before the generate call. So calls == cap.)
    assert len(lm.calls) == 64, (
        f"safety cap should produce exactly 64 generate calls before "
        f"breaking; got {len(lm.calls)}. If 0, the cap check ran before "
        f"the per-turn gather — a DP deadlock footgun."
    )
    # All wait_for_everyone calls (one per generated turn) executed in
    # lockstep with the gathers — no skipped collective.
    assert lm.accelerator.wait_calls == 64


# ---------------------------------------------------------------------------
# Placeholder for a real cross-process DP smoke. Not run in CI; documents
# where the real validation needs to live.
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="Real cross-process DP smoke: launch with "
    "`accelerate launch --num_processes 2 -m pytest tests/test_multi_turn_rollout.py "
    "-k real_dp_smoke`. The in-process `_StubAccelerator` cannot catch "
    "actual collective-ordering deadlocks; this placeholder marks the gap."
)
def test_real_dp_smoke_placeholder():  # pragma: no cover
    """Documented gap. To validate real DP coordination end-to-end,
    instantiate an actual HF LM, wrap with accelerate, and run the
    driver against a 4-doc stub task. Compare per-rank ``_instances``
    counts to expected shard size.
    """
    raise AssertionError("must be run under accelerate launch; see skip reason")
