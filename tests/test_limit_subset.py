from types import SimpleNamespace

from lm_eval.api.task import Task


def _task(n):
    return SimpleNamespace(
        eval_docs=[{"i": i} for i in range(n)], config=SimpleNamespace(task="t")
    )


def _ids(limit, **kw):
    return [i for i, _ in Task.doc_iterator(_task(1000), limit=limit, **kw)]


def test_limit_is_a_random_subset_not_the_first_docs():
    ids = _ids(50)
    assert len(ids) == 50 and ids != list(range(50))
    assert ids == sorted(ids) and max(ids) > 500


def test_same_subset_on_every_call():
    assert _ids(50) == _ids(50)


def test_doc_ids_index_the_full_dataset():
    for i, doc in Task.doc_iterator(_task(1000), limit=20):
        assert doc == {"i": i}


def test_ranks_split_the_same_subset():
    both = sorted(_ids(50, rank=0, world_size=2) + _ids(50, rank=1, world_size=2))
    assert both == _ids(50)


def test_no_limit_or_limit_above_size_keeps_every_doc_in_order():
    assert _ids(None) == list(range(1000))
    assert _ids(5000) == list(range(1000))
