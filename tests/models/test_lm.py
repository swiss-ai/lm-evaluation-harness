from lm_eval.api.model import LM


class _DummyLM(LM):
    def loglikelihood(self, requests):
        return []

    def loglikelihood_rolling(self, requests):
        return []

    def generate_until(self, requests):
        return []


def test_process_properties_default_to_data_parallel_rank():
    lm = _DummyLM()
    lm._rank = 2

    assert lm.process_rank == 2
    assert lm.cache_rank == 2
    assert not lm.is_main_process


def test_process_properties_use_accelerator_process_rank():
    lm = _DummyLM()
    lm._rank = 0
    lm.accelerator = type(
        "Accelerator", (), {"process_index": 3, "is_main_process": False}
    )()

    assert lm.process_rank == 3
    assert lm.cache_rank == 3
    assert not lm.is_main_process


def test_process_properties_use_global_environment_rank(monkeypatch):
    lm = _DummyLM()
    lm._rank = 2
    monkeypatch.setenv("RANK", "4")
    monkeypatch.setenv("LOCAL_RANK", "0")

    assert lm.process_rank == 4
    assert lm.cache_rank == 4
    assert not lm.is_main_process


def test_object_gathering_is_a_noop_for_one_worker():
    lm = _DummyLM()

    assert lm.all_gather_object({"value": 1}) == [{"value": 1}]
    assert lm.gather_object({"value": 1}) == [{"value": 1}]


def test_model_metrics_are_recorded():
    lm = _DummyLM()

    lm.add_model_metrics({})
    lm.add_model_metrics({"metric": 1.0})

    assert lm.get_model_metrics() == [{"metric": 1.0}]
