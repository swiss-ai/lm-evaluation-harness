"""Style-controlled win rate for Arena-Hard v2.0.

The style-controlled win rate answers: "what would the win rate be if
the model had the same style (length, formatting) as the baseline?"

Mathematical model:
    P(model wins) = sigmoid(quality_coef + style_features @ style_coefs)
    style-controlled win rate = sigmoid(quality_coef)

For the single-model-vs-baseline case, the multi-model one-hot encoding
simplifies to a single intercept column (all 1s).

Feature extraction (count_markdown_elements, remove_pattern) is copied from
arena-hard-auto/utils/add_markdown_info.py (Apache 2.0 license).

Feature normalization (compute_relative_features, zscore_normalize) is adapted
from arena-hard-auto/show_result.py lines 149-174 (Apache 2.0 license).

Bradley-Terry model fitting (BTModel, bt_loss, fit_pairwise_model) is adapted
from arena-hard-auto/utils/math_utils.py (Apache 2.0 license).

Reference: https://github.com/lm-sys/arena-hard-auto
"""

import logging
import re

import numpy as np
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


logger = logging.getLogger(__name__)

# tiktoken encoder matching the reference implementation
# (add_markdown_info.py line 66)
_ENCODER = tiktoken.encoding_for_model("gpt-4o")

# Compiled pattern for stripping code blocks before markdown counting
# (add_markdown_info.py line 41)
_CODE_BLOCK_PATTERN = re.compile(r"```([^`]*)```")


# ── Feature Extraction ───────────────────────────────────────────────
# Copied from arena-hard-auto/utils/add_markdown_info.py (Apache 2.0)


def count_markdown_elements(markdown_text):
    """Count markdown formatting elements in text.

    Copied from add_markdown_info.py::count_markdown_elements() (lines 11-30).

    Returns:
        Dict with keys 'header_count', 'list_count', 'bold_count',
        each mapping to a dict of sub-counts.
    """
    return {
        "header_count": {
            "h1": len(re.findall(r"^#{1}\s", markdown_text, re.MULTILINE)),
            "h2": len(re.findall(r"^#{2}\s", markdown_text, re.MULTILINE)),
            "h3": len(re.findall(r"^#{3}\s", markdown_text, re.MULTILINE)),
            "h4": len(re.findall(r"^#{4}\s", markdown_text, re.MULTILINE)),
            "h5": len(re.findall(r"^#{5}\s", markdown_text, re.MULTILINE)),
            "h6": len(re.findall(r"^#{6}\s", markdown_text, re.MULTILINE)),
        },
        "list_count": {
            "ordered": len(re.findall(r"^\s*\d+\.\s", markdown_text, re.MULTILINE)),
            "unordered": len(re.findall(r"^\s*[-*+]\s", markdown_text, re.MULTILINE)),
        },
        "bold_count": {
            "**": len(re.findall(r"\*\*[^*\n]+\*\*", markdown_text)),
            "__": len(re.findall(r"__[^_\n]+__", markdown_text)),
        },
    }


def remove_pattern(answer, pattern):
    """Remove all matches of a compiled regex from text.

    Copied from add_markdown_info.py::remove_pattern() (lines 33-37).
    """
    blocks = pattern.findall(answer)
    for block in blocks:
        answer = answer.replace(block, "")
    return answer


def extract_style_features(text):
    """Extract the 4-element style feature vector from a text.

    Features: [token_len, header_count, list_count, bold_count]

    Matches the reference arena-hard-auto implementation:
      - token_len via tiktoken (gpt-4o encoder)
        (add_markdown_info.py::add_markdown_meta(), line 56)
      - markdown counts via count_markdown_elements after stripping code blocks
        (add_markdown_info.py::get_element_counts(), lines 40-52)
      - sub-counts collapsed via sum(v.values())
        (show_result.py line 140)

    Args:
        text: Model or baseline output text.

    Returns:
        np.ndarray of shape (4,) with raw feature counts.
    """
    if not text:
        return np.array([0, 0, 0, 0], dtype=np.float64)

    token_len = len(_ENCODER.encode(text, disallowed_special=()))
    cleaned = remove_pattern(text, _CODE_BLOCK_PATTERN)
    counters = count_markdown_elements(cleaned)

    return np.array([
        token_len,
        sum(counters["header_count"].values()),
        sum(counters["list_count"].values()),
        sum(counters["bold_count"].values()),
    ], dtype=np.float64)


# ── Feature Normalization ────────────────────────────────────────────
# From arena-hard-auto/show_result.py::print_leaderboard_with_style_features()


def compute_relative_features(model_features, baseline_features):
    """Compute relative style features (model vs baseline).

    From show_result.py lines 149-166:
      - Length (col 0): (model - baseline) / (model + baseline)
      - Markdown (cols 1-3): per-token density, then relative difference
        density = count / (token_len + 1)
        relative = (m_density - b_density) / (m_density + b_density + 1)

    Args:
        model_features: Shape (N, 4) torch tensor of raw features.
        baseline_features: Shape (N, 4) torch tensor of raw features.

    Returns:
        torch.Tensor of shape (N, 4) with relative features.
    """
    final = torch.zeros_like(model_features)

    # show_result.py lines 150-154: relative token length
    final[:, 0] = (
        model_features[:, 0] - baseline_features[:, 0]
    ) / (
        model_features[:, 0] + baseline_features[:, 0]
    )

    # show_result.py lines 156-157: per-token markdown density
    model_md_density = model_features[:, 1:] / (model_features[:, :1] + 1)
    baseline_md_density = baseline_features[:, 1:] / (baseline_features[:, :1] + 1)

    # show_result.py lines 162-166: relative density difference
    final[:, 1:] = (
        model_md_density - baseline_md_density
    ) / (
        model_md_density + baseline_md_density + 1
    )

    # Replace NaN with 0 (can happen if both token lengths are 0)
    final = torch.nan_to_num(final, nan=0.0)

    return final


def zscore_normalize(features):
    """Z-score normalize feature columns.

    From show_result.py lines 170-174.

    Args:
        features: Shape (N, K) torch tensor.

    Returns:
        torch.Tensor of shape (N, K), z-score normalized.
        Constant columns (std=0) are set to 0.
    """
    normalized = (
        features - torch.mean(features, dim=0)
    ) / torch.std(features, dim=0)

    # Replace NaN with 0 (constant columns have std=0)
    normalized = torch.nan_to_num(normalized, nan=0.0)

    return normalized


# ── Bradley-Terry Model ─────────────────────────────────────────────
# From arena-hard-auto/utils/math_utils.py (Apache 2.0)


class BTModel(nn.Module):
    """Bradley-Terry model with learnable logit coefficients.

    From math_utils.py::BTModel (lines 41-50).
    """

    def __init__(self, num_components):
        super().__init__()
        self.logits = nn.Parameter(
            nn.init.constant_(torch.empty(num_components), 0.5)
        )

    def forward(self):
        return self.logits, None


def bt_loss(logits, outcomes, **kwargs):
    """Binary cross-entropy loss for the BT model.

    From math_utils.py::bt_loss (lines 66-80).
    """
    return F.binary_cross_entropy_with_logits(
        logits, outcomes.float(), reduction="sum"
    )


def fit_pairwise_model(
    features, outcomes, indices=None, lr=0.1, tol=1e-9, max_epochs=50
):
    """Fit Bradley-Terry model via L-BFGS optimization.

    From math_utils.py::fit_pairwise_model (lines 109-154),
    simplified for the BT loss type only.

    Args:
        features: Shape (N, K) torch tensor.
        outcomes: Shape (N,) torch tensor of outcomes in [0, 1].
        indices: Optional bootstrap sample indices.
        lr: Learning rate for L-BFGS.
        tol: Gradient and change tolerance.
        max_epochs: Max L-BFGS iterations.

    Returns:
        torch.Tensor of shape (K,) fitted coefficients.
    """
    if indices is not None:
        features = features[indices]
        outcomes = outcomes[indices]

    model = BTModel(num_components=features.shape[1])

    # math_utils.py lines 129-135
    optimizer = optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=max_epochs,
        tolerance_grad=tol,
        tolerance_change=tol,
    )

    # math_utils.py lines 137-151
    def closure():
        optimizer.zero_grad()
        logits, eta = model()
        _logits = features @ logits
        loss = bt_loss(logits=_logits, outcomes=outcomes, eta=eta)
        loss.backward()
        return loss

    optimizer.step(closure)

    logits, _ = model()
    return logits.detach()


def bootstrap_style_controlled_winrate(
    features, outcomes, num_style_features, num_bootstrap=100, seed=42
):
    """Bootstrap BT fitting and extract style-controlled win rate.

    Adapted from math_utils.py::bootstrap_pairwise_model (lines 162-182)
    and show_result.py lines 201-212.

    For our single-model-vs-baseline case, the design matrix is
    [intercept | style_features]. After fitting, the intercept coefficient
    (index 0) represents quality, and sigmoid(intercept) gives the
    style-controlled win rate.

    The reference uses coefs[:, :-num_features] to strip style coefs
    (show_result.py line 203), then to_winrate_probabilities() to convert
    to win rates (show_result.py lines 205-212). In our single-model case
    this simplifies to sigmoid(coefs[:, 0]).

    Args:
        features: Shape (N, 1 + num_style_features) torch tensor with
            intercept column first.
        outcomes: Shape (N,) torch tensor of outcomes in [0, 1].
        num_style_features: Number of style feature columns (after intercept).
        num_bootstrap: Number of bootstrap rounds.
        seed: RNG seed for reproducibility.

    Returns:
        Tuple of (win_rate, ci_lower, ci_upper) on 0-1 scale.
    """
    rng = np.random.default_rng(seed=seed)
    n = features.shape[0]

    # math_utils.py lines 168-171
    boot_idxs = rng.integers(0, n, size=(num_bootstrap, n))

    # math_utils.py line 173
    results = [
        fit_pairwise_model(features, outcomes, boot_idxs[i])
        for i in range(num_bootstrap)
    ]

    # math_utils.py line 175
    coef_stacks = torch.stack(results)

    # show_result.py line 203: strip style coefs, keep quality (intercept)
    quality_coefs = coef_stacks[:, 0]

    # Convert to win probabilities via sigmoid
    winrates = torch.sigmoid(quality_coefs)

    # show_result.py lines 214-218: median and 90% CI
    mean_wr = torch.quantile(winrates, 0.5).item()
    ci_lower = torch.quantile(winrates, 0.05).item()
    ci_upper = torch.quantile(winrates, 0.95).item()

    return mean_wr, ci_lower, ci_upper
