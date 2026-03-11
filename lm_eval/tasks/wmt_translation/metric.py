import gc
import logging


COMETKIWI_MODEL_NAME = "Unbabel/wmt22-cometkiwi-da"

_comet_model = None
_vllm_freed = False
eval_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transformers 5.x compatibility patches for unbabel-comet
# ---------------------------------------------------------------------------


def _patch_comet_for_transformers5():
    _patch_tokenizer()
    _patch_encoder_forward()


def _patch_tokenizer():
    for cls_name in ("XLMRobertaTokenizer", "XLMRobertaTokenizerFast"):
        try:
            import transformers

            cls = getattr(transformers, cls_name, None)
            if cls is None or hasattr(cls, "build_inputs_with_special_tokens"):
                continue

            def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
                cls_id = [self.cls_token_id]
                sep_id = [self.sep_token_id]
                if token_ids_1 is None:
                    return cls_id + token_ids_0 + sep_id
                return cls_id + token_ids_0 + sep_id + sep_id + token_ids_1 + sep_id

            cls.build_inputs_with_special_tokens = build_inputs_with_special_tokens
            eval_logger.info(f"Patched {cls_name}.build_inputs_with_special_tokens")
        except Exception as e:
            eval_logger.warning(f"Could not patch {cls_name}: {e}")


def _patch_encoder_forward():
    try:
        from comet.encoders.xlmr import XLMREncoder

        def patched_forward(self, input_ids, attention_mask, **kwargs):
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            return {
                "sentemb": output.last_hidden_state[:, 0, :],
                "wordemb": output.last_hidden_state,
                "all_layers": output.hidden_states,
            }

        XLMREncoder.forward = patched_forward
        eval_logger.info("Patched XLMREncoder.forward for transformers>=5.0")
    except Exception as e:
        eval_logger.warning(f"Could not patch XLMREncoder.forward: {e}")


# ---------------------------------------------------------------------------
# GPU memory management
# ---------------------------------------------------------------------------


def _free_vllm_memory():
    """Free GPU memory held by vLLM.  Called once before CometKiwi22 scoring.

    Safe to call after all generation tasks have completed (the harness runs
    aggregation only after every generate_until request is done).
    """
    global _vllm_freed
    if _vllm_freed:
        return
    _vllm_freed = True

    import torch

    eval_logger.info("Freeing vLLM GPU memory before CometKiwi22 scoring...")

    # 1. Find the lm_eval VLLM wrapper and delete its LLM instance
    try:
        from lm_eval.models.vllm_causallms import VLLM as VLLMWrapper

        for obj in gc.get_objects():
            if isinstance(obj, VLLMWrapper) and hasattr(obj, "model"):
                eval_logger.info("Found lm_eval VLLM wrapper — deleting LLM instance")
                try:
                    if hasattr(obj.model, "llm_engine"):
                        obj.model.llm_engine.shutdown()
                except Exception as e:
                    eval_logger.debug(f"Could not shutdown vLLM engine: {e}")
                del obj.model
                break
    except Exception as e:
        eval_logger.warning(f"Could not find/delete VLLM model: {e}")

    # 2. Tear down vLLM distributed state
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel

        destroy_model_parallel()
        eval_logger.info("Destroyed vLLM model parallel state")
    except Exception as e:
        eval_logger.debug(f"Could not destroy model parallel: {e}")

    # 3. Garbage-collect and release CUDA caches
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()

    _log_gpu_memory("After vLLM cleanup")


def _log_gpu_memory(label=""):
    import torch

    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        eval_logger.info(
            f"  GPU {i}: {free / 2**30:.1f} GiB free / {total / 2**30:.1f} GiB total"
            + (f"  [{label}]" if label else "")
        )


def _get_gpu_config():
    """Return (num_gpus, batch_size) based on available GPU memory."""
    import torch

    if not torch.cuda.is_available():
        eval_logger.info("No CUDA GPUs — CometKiwi22 will run on CPU (slow)")
        return 0, 8

    # SLURM's ntasks-per-node=1 conflicts with Lightning multi-GPU,
    # so we always use a single GPU for CometKiwi22 scoring.
    num_gpus = 1

    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    free_gib = free_bytes / 2**30

    # CometKiwi22 (XLM-R large): ~2 GiB for model weights + ~80 MiB per
    # batch item (embedding + attention over 512-token sequences).
    overhead_gib = 3.0
    available_gib = max(free_gib - overhead_gib, 0.5)
    batch_size = max(int(available_gib * 1024 / 80), 4)
    batch_size = min(batch_size, 256)

    eval_logger.info(
        f"CometKiwi22 config: {num_gpus} GPU, batch_size={batch_size} "
        f"(free memory: {free_gib:.1f} GiB)"
    )
    return num_gpus, batch_size


# ---------------------------------------------------------------------------
# COMET model
# ---------------------------------------------------------------------------


def _get_comet_model():
    global _comet_model
    if _comet_model is None:
        _patch_comet_for_transformers5()
        from comet import download_model, load_from_checkpoint

        model_path = download_model(COMETKIWI_MODEL_NAME)
        _comet_model = load_from_checkpoint(model_path)
    return _comet_model


# ---------------------------------------------------------------------------
# lm-eval-harness interface
# ---------------------------------------------------------------------------


def translation_score(doc, predictions):
    """Per-document: collect src and translation for batch CometKiwi22 scoring."""
    return {"cometkiwi22": {"src": doc["src"], "mt": predictions[0]}}


def cometkiwi22_agg(items):
    """Batch CometKiwi22 scoring.  Runs after all generation tasks complete."""
    _free_vllm_memory()

    gpus, batch_size = _get_gpu_config()
    model = _get_comet_model()

    eval_logger.info(
        f"Scoring {len(items)} translations with CometKiwi22 "
        f"(gpus={gpus}, batch_size={batch_size})..."
    )
    output = model.predict(items, batch_size=batch_size, gpus=gpus)
    score = sum(output.scores) / len(output.scores) if output.scores else 0.0
    eval_logger.info(f"CometKiwi22 mean score: {score:.4f}")
    return score


def _gpu_available():
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False
