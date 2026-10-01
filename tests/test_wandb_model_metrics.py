from unittest.mock import Mock

from lm_eval.loggers.wandb_logger import WandbLogger


def test_model_metrics_are_logged_as_inference_records():
    logger = object.__new__(WandbLogger)
    logger.results = {"model_metrics": [{"router_max": 0.25}, {"router_max": 0.5}]}
    logger.step_metrics = {"OptimizerStep": 2000, "ConsumedTokens": 4000000}
    logger.run = Mock()
    logger.run.summary = {"inference/global_step": 4}

    logger._log_model_metrics()

    payloads = [call.args[0] for call in logger.run.log.call_args_list]
    assert payloads == [
        {
            "inference/global_step": 5,
            "inference/local_step": 1,
            "OptimizerStep": 2000,
            "ConsumedTokens": 4000000,
            "inference/router_max": 0.25,
        },
        {
            "inference/global_step": 6,
            "inference/local_step": 2,
            "OptimizerStep": 2000,
            "ConsumedTokens": 4000000,
            "inference/router_max": 0.5,
        },
    ]
