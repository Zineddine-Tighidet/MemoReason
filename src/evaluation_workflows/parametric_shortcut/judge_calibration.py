"""Calibration helpers for binary LLM-as-a-judge estimators.

This module implements the setup described in the project discussion:

- calibration dataset:
    compare binary judge outputs against human consensus labels to estimate
    the positive and negative recalls of the judge
- estimation dataset:
    apply the calibrated judge to a separate sample and debias the observed
    judge-positive rate into an estimate of the true success rate

The corrected estimator follows:

    p_hat = (q + r_minus - 1) / (r_plus + r_minus - 1)

where:
    q       = observed judge-positive rate on the estimation set
    r_plus  = P(judge=1 | truth=1)
    r_minus = P(judge=0 | truth=0)
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import NormalDist
from typing import Iterable


def _coerce_binary_label(value: bool | int) -> int:
    if isinstance(value, bool):
        return int(value)
    if value in (0, 1):
        return int(value)
    raise ValueError(f"Binary labels must be 0/1 or bool, got {value!r}.")


def _coerce_binary_labels(values: Iterable[bool | int], *, name: str) -> list[int]:
    labels = [_coerce_binary_label(value) for value in values]
    if not labels:
        raise ValueError(f"{name} must contain at least one label.")
    return labels


def z_score_for_confidence(confidence_level: float) -> float:
    """Return the two-sided Gaussian z-score for a confidence level."""
    confidence = float(confidence_level)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must lie strictly between 0 and 1.")
    tail_mass = 0.5 + confidence / 2.0
    return float(NormalDist().inv_cdf(tail_mass))


@dataclass(frozen=True)
class BinaryJudgeConfusionCounts:
    """Confusion counts for judge outputs against human labels."""

    true_positives: int
    false_negatives: int
    true_negatives: int
    false_positives: int

    @property
    def n_examples(self) -> int:
        return self.true_positives + self.false_negatives + self.true_negatives + self.false_positives

    @property
    def n_human_positive(self) -> int:
        return self.true_positives + self.false_negatives

    @property
    def n_human_negative(self) -> int:
        return self.true_negatives + self.false_positives


@dataclass(frozen=True)
class BinaryJudgeCalibration:
    """Empirical calibration summary for a binary judge."""

    positive_recall: float
    negative_recall: float
    confusion_counts: BinaryJudgeConfusionCounts

    @property
    def youden_j(self) -> float:
        return self.positive_recall + self.negative_recall - 1.0


@dataclass(frozen=True)
class CalibratedSuccessEstimate:
    """Debiased estimate of a system success rate under a calibrated judge."""

    estimated_success_rate: float
    observed_judge_success_rate: float
    n_trials: int
    z_score: float
    margin: float
    lower_bound: float
    upper_bound: float
    raw_lower_bound: float
    raw_upper_bound: float


def compute_binary_judge_confusion_counts(
    human_labels: Iterable[bool | int],
    judge_labels: Iterable[bool | int],
) -> BinaryJudgeConfusionCounts:
    """Return confusion counts for judge outputs against human labels."""
    truth = _coerce_binary_labels(human_labels, name="human_labels")
    judge = _coerce_binary_labels(judge_labels, name="judge_labels")
    if len(truth) != len(judge):
        raise ValueError("human_labels and judge_labels must have the same length.")

    true_positives = 0
    false_negatives = 0
    true_negatives = 0
    false_positives = 0

    for human_label, judge_label in zip(truth, judge, strict=True):
        if human_label == 1 and judge_label == 1:
            true_positives += 1
        elif human_label == 1 and judge_label == 0:
            false_negatives += 1
        elif human_label == 0 and judge_label == 0:
            true_negatives += 1
        else:
            false_positives += 1

    return BinaryJudgeConfusionCounts(
        true_positives=true_positives,
        false_negatives=false_negatives,
        true_negatives=true_negatives,
        false_positives=false_positives,
    )


def calibrate_binary_judge(
    human_labels: Iterable[bool | int],
    judge_labels: Iterable[bool | int],
) -> BinaryJudgeCalibration:
    """Estimate positive and negative recall from a calibration sample."""
    confusion_counts = compute_binary_judge_confusion_counts(human_labels, judge_labels)
    if confusion_counts.n_human_positive == 0:
        raise ValueError("Calibration requires at least one human-positive example.")
    if confusion_counts.n_human_negative == 0:
        raise ValueError("Calibration requires at least one human-negative example.")

    positive_recall = confusion_counts.true_positives / confusion_counts.n_human_positive
    negative_recall = confusion_counts.true_negatives / confusion_counts.n_human_negative
    return BinaryJudgeCalibration(
        positive_recall=float(positive_recall),
        negative_recall=float(negative_recall),
        confusion_counts=confusion_counts,
    )


def estimate_true_success_rate(
    observed_judge_success_rate: float,
    *,
    n_trials: int,
    positive_recall: float,
    negative_recall: float,
    z_score: float = 1.96,
) -> CalibratedSuccessEstimate:
    """Estimate the true success rate from a calibrated binary judge.

    This uses the approximation:

        p_hat = (q + r_minus - 1) / (r_plus + r_minus - 1)
        delta = z * sqrt(q * (1 - q) / n) / (r_plus + r_minus - 1)

    The returned interval is clipped to ``[0, 1]`` as recommended in the note.
    """
    q = float(observed_judge_success_rate)
    n = int(n_trials)
    r_plus = float(positive_recall)
    r_minus = float(negative_recall)
    z = float(z_score)

    if not 0.0 <= q <= 1.0:
        raise ValueError("observed_judge_success_rate must lie in [0, 1].")
    if n <= 0:
        raise ValueError("n_trials must be strictly positive.")
    if not 0.0 <= r_plus <= 1.0:
        raise ValueError("positive_recall must lie in [0, 1].")
    if not 0.0 <= r_minus <= 1.0:
        raise ValueError("negative_recall must lie in [0, 1].")
    if z < 0.0:
        raise ValueError("z_score must be non-negative.")

    denominator = r_plus + r_minus - 1.0
    if denominator <= 0.0:
        raise ValueError(
            "The calibrated judge is not identifiable because r_plus + r_minus - 1 <= 0. "
            "The correction would be unstable or undefined."
        )

    estimated_success_rate = (q + r_minus - 1.0) / denominator
    margin = z * math.sqrt(max(0.0, q * (1.0 - q)) / n) / denominator
    raw_lower_bound = estimated_success_rate - margin
    raw_upper_bound = estimated_success_rate + margin

    return CalibratedSuccessEstimate(
        estimated_success_rate=float(estimated_success_rate),
        observed_judge_success_rate=q,
        n_trials=n,
        z_score=z,
        margin=float(margin),
        lower_bound=max(0.0, raw_lower_bound),
        upper_bound=min(1.0, raw_upper_bound),
        raw_lower_bound=float(raw_lower_bound),
        raw_upper_bound=float(raw_upper_bound),
    )


def estimate_true_success_rate_from_labels(
    judge_labels: Iterable[bool | int],
    *,
    calibration: BinaryJudgeCalibration,
    z_score: float = 1.96,
) -> CalibratedSuccessEstimate:
    """Convenience wrapper using raw judge outputs on the estimation set."""
    labels = _coerce_binary_labels(judge_labels, name="judge_labels")
    observed_rate = sum(labels) / len(labels)
    return estimate_true_success_rate(
        observed_rate,
        n_trials=len(labels),
        positive_recall=calibration.positive_recall,
        negative_recall=calibration.negative_recall,
        z_score=z_score,
    )


def estimate_true_success_rate_with_confidence(
    observed_judge_success_rate: float,
    *,
    n_trials: int,
    positive_recall: float,
    negative_recall: float,
    confidence_level: float = 0.95,
) -> CalibratedSuccessEstimate:
    """Like :func:`estimate_true_success_rate`, but parameterized by confidence."""
    return estimate_true_success_rate(
        observed_judge_success_rate,
        n_trials=n_trials,
        positive_recall=positive_recall,
        negative_recall=negative_recall,
        z_score=z_score_for_confidence(confidence_level),
    )


__all__ = [
    "BinaryJudgeCalibration",
    "BinaryJudgeConfusionCounts",
    "CalibratedSuccessEstimate",
    "calibrate_binary_judge",
    "compute_binary_judge_confusion_counts",
    "estimate_true_success_rate",
    "estimate_true_success_rate_from_labels",
    "estimate_true_success_rate_with_confidence",
    "z_score_for_confidence",
]
