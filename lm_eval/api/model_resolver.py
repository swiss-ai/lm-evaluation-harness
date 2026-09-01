"""Resolve host-scoped model IDs exposed by OpenAI-compatible endpoints."""

from __future__ import annotations

import getpass
import json
import logging
import os
import ssl
import threading
import urllib.request
from urllib.parse import urlparse


eval_logger = logging.getLogger(__name__)

CSCS_PROVIDER_PREFIX = "CSCS-Inference"
JUDGE_MODEL_PREFIX_ENV = "JUDGE_MODEL_PREFIX"
JUDGE_MODEL_DISCOVERY_ENV = "LM_EVAL_JUDGE_MODEL_DISCOVERY"
JUDGE_MODEL_DISCOVERY_TIMEOUT_ENV = "LM_EVAL_JUDGE_MODEL_DISCOVERY_TIMEOUT"

_RESOLUTION_CACHE: dict[tuple[str, str, str, str, str], str] = {}
_RESOLUTION_LOCK = threading.Lock()


def _is_cscs_endpoint(api_base: str | None) -> bool:
    if not api_base:
        return False
    hostname = (urlparse(api_base).hostname or "").lower()
    return hostname == "cscs.ch" or hostname.endswith(".cscs.ch")


def _username() -> str:
    username = os.getenv("USER", "").strip()
    if username:
        return username
    try:
        return getpass.getuser().strip()
    except (KeyError, OSError):
        return ""


def _judge_model_prefix(api_base: str | None) -> str:
    configured = os.getenv(JUDGE_MODEL_PREFIX_ENV)
    if configured is not None:
        return configured.strip().strip("/")
    return _username() if _is_cscs_endpoint(api_base) else ""


def _is_scoped(model: str, prefix: str) -> bool:
    parts = model.split("/")
    if len(parts) >= 3:
        return True
    known_prefixes = {prefix, _username(), CSCS_PROVIDER_PREFIX}
    return any(
        known_prefix and model.startswith(f"{known_prefix}/")
        for known_prefix in known_prefixes
    )


def preferred_judge_model(model: str, api_base: str | None) -> str:
    """Return the configured model with the preferred host scope applied."""

    configured_model = os.path.expandvars(model).strip().strip("/")
    if not configured_model:
        raise ValueError("Judge model name must not be empty.")

    prefix = _judge_model_prefix(api_base)
    if not prefix or _is_scoped(configured_model, prefix):
        return configured_model
    return f"{prefix}/{configured_model}"


def _models_url(api_base: str) -> str:
    url = f"{api_base.rstrip('/')}/models"
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported judge API URL scheme: {parsed.scheme!r}")
    return url


def _hosted_model_ids(api_base: str, api_key: str | None) -> tuple[str, ...]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    request = urllib.request.Request(  # noqa: S310 - URL scheme is validated.
        _models_url(api_base), headers=headers
    )
    timeout = float(os.getenv(JUDGE_MODEL_DISCOVERY_TIMEOUT_ENV, "10"))
    context = ssl.create_default_context()
    with urllib.request.urlopen(  # noqa: S310 - URL scheme is validated above.
        request,
        timeout=timeout,
        context=context,
    ) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return tuple(
        model["id"]
        for model in payload.get("data", [])
        if isinstance(model, dict) and isinstance(model.get("id"), str)
    )


def _select_hosted_model(
    configured_model: str,
    preferred_model: str,
    hosted_models: tuple[str, ...],
) -> str:
    base_model = configured_model
    candidates = [
        preferred_model,
        f"{CSCS_PROVIDER_PREFIX}/{base_model}",
    ]

    for candidate in dict.fromkeys(candidates):
        if candidate in hosted_models:
            return candidate

    suffix_matches = sorted(
        model for model in hosted_models if model.endswith(f"/{base_model}")
    )
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    if len(suffix_matches) > 1:
        eval_logger.warning(
            "Multiple hosted judge models match %s: %s; using preferred model %s",
            configured_model,
            suffix_matches,
            preferred_model,
        )
    return preferred_model


def resolve_judge_model(
    model: str,
    *,
    api_base: str | None,
    api_key: str | None = None,
) -> str:
    """Resolve a judge model to the ID advertised by an endpoint.

    CSCS models default to ``$USER/<model>``. The endpoint's ``/models`` list
    is consulted to fall back to ``CSCS-Inference/<model>`` or another unique
    host scope. Explicitly scoped model names are preserved.
    """

    configured_model = os.path.expandvars(model).strip().strip("/")
    preferred_model = preferred_judge_model(configured_model, api_base)
    prefix = _judge_model_prefix(api_base)
    if _is_scoped(configured_model, prefix):
        return configured_model
    discovery_enabled = os.getenv(JUDGE_MODEL_DISCOVERY_ENV, "1").lower() not in {
        "0",
        "false",
        "no",
    }
    should_discover = (
        bool(api_base)
        and discovery_enabled
        and (_is_cscs_endpoint(api_base) or JUDGE_MODEL_PREFIX_ENV in os.environ)
    )
    if not should_discover:
        return preferred_model

    cache_key = (
        configured_model,
        api_base or "",
        api_key or "",
        prefix,
        os.getenv(JUDGE_MODEL_DISCOVERY_ENV, "1"),
    )
    with _RESOLUTION_LOCK:
        if cache_key in _RESOLUTION_CACHE:
            return _RESOLUTION_CACHE[cache_key]

        try:
            hosted_models = _hosted_model_ids(api_base or "", api_key)
            resolved_model = _select_hosted_model(
                configured_model,
                preferred_model,
                hosted_models,
            )
        except Exception as exc:  # noqa: BLE001 - discovery must not block judging.
            eval_logger.warning(
                "Could not discover hosted judge models at %s (%s); using %s",
                api_base,
                exc,
                preferred_model,
            )
            resolved_model = preferred_model

        if resolved_model != configured_model:
            eval_logger.info(
                "Resolved judge model %s to hosted ID %s",
                configured_model,
                resolved_model,
            )
        _RESOLUTION_CACHE[cache_key] = resolved_model
        return resolved_model


def get_judge_model(
    default_model: str,
    *,
    env_var: str,
    api_base: str | None,
    api_key: str | None = None,
) -> str:
    """Read a task-specific model override and resolve its hosted model ID."""

    configured_model = os.getenv(env_var, default_model)
    if not configured_model:
        raise OSError(f"Set {env_var} to a non-empty judge model name.")
    return resolve_judge_model(
        configured_model,
        api_base=api_base,
        api_key=api_key,
    )


def clear_judge_model_resolution_cache() -> None:
    """Clear cached endpoint discovery results (primarily for tests)."""

    with _RESOLUTION_LOCK:
        _RESOLUTION_CACHE.clear()
