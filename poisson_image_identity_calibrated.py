#!/usr/bin/env python3
"""L2-regularized and probability-calibrated natural-image decoder.

This script uses the same NWB loading and Psycode unit-selection code as the
baseline, but reserves separate training, calibration, and final test sets.

Example:
    python poisson_image_identity_calibrated.py --nwb-path session.nwb --segment pre
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    log_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from poisson_image_identity_baseline import load_dataset


DEFAULT_COVERAGE_LEVELS = (0.50, 0.80, 0.90, 0.95)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Decode and calibrate repeated natural-image identity."
    )
    parser.add_argument("--nwb-path", required=True, type=Path, help="NWB file path.")
    parser.add_argument("--segment", choices=("pre", "post"), default="pre")
    parser.add_argument("--window-start", type=float, default=0.050)
    parser.add_argument("--window-end", type=float, default=0.250)
    parser.add_argument("--bin-size", type=float, default=0.005)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--calibration-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=314159)
    parser.add_argument("--calibration-random-state", type=int, default=271828)
    parser.add_argument("--l2-strength", type=float, default=0.01)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--confidence-bins", type=int, default=10)
    return parser.parse_args()


def validate_args(args):
    if not args.nwb_path.is_file():
        raise FileNotFoundError(f"NWB file not found: {args.nwb_path}")
    if args.bin_size <= 0:
        raise ValueError("--bin-size must be positive.")
    if args.window_end <= args.window_start:
        raise ValueError("--window-end must be greater than --window-start.")
    if not 0 < args.test_size < 1 or not 0 < args.calibration_size < 1:
        raise ValueError("Split sizes must be between 0 and 1.")
    if args.test_size + args.calibration_size >= 1:
        raise ValueError("Test and calibration sizes must sum to less than 1.")
    if args.l2_strength < 0:
        raise ValueError("--l2-strength cannot be negative.")
    if args.max_iter < 1 or args.confidence_bins < 1:
        raise ValueError("Iteration and confidence-bin counts must be positive.")


class RegularizedPoissonNB(ClassifierMixin, BaseEstimator):
    """Naive Bayes with class rates estimated by L2-regularized Poisson GLMs."""

    def __init__(self, l2_strength=0.01, max_iter=500):
        self.l2_strength = l2_strength
        self.max_iter = max_iter

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if np.any(X < 0):
            raise ValueError("Poisson counts must be nonnegative.")

        self.classes_, encoded_y = np.unique(y, return_inverse=True)
        self.n_features_in_ = X.shape[1]
        n_classes = len(self.classes_)
        class_counts = np.bincount(encoded_y, minlength=n_classes)
        self.class_log_prior_ = np.log(class_counts / class_counts.sum())

        indicators = np.eye(n_classes)[encoded_y]
        design = np.column_stack([np.ones(len(y)), indicators])
        feature_rates = np.empty((n_classes, X.shape[1]), dtype=float)

        for feature_index in range(X.shape[1]):
            counts = X[:, feature_index]
            initial = np.zeros(n_classes + 1)
            initial[0] = np.log(max(counts.mean(), 1e-6))

            def objective(parameters):
                predictor = design @ parameters
                rates = np.exp(np.clip(predictor, -20, 20))
                negative_log_likelihood = np.sum(rates - counts * predictor)
                penalty = 0.5 * self.l2_strength * np.sum(parameters[1:] ** 2)
                return negative_log_likelihood + penalty

            def gradient(parameters):
                predictor = design @ parameters
                rates = np.exp(np.clip(predictor, -20, 20))
                result = design.T @ (rates - counts)
                result[1:] += self.l2_strength * parameters[1:]
                return result

            result = minimize(
                objective,
                initial,
                jac=gradient,
                method="L-BFGS-B",
                bounds=[(-20, 20)] * (n_classes + 1),
                options={"maxiter": self.max_iter},
            )
            if not result.success:
                raise RuntimeError(
                    f"Poisson GLM failed for feature {feature_index}: {result.message}"
                )
            feature_rates[:, feature_index] = np.exp(
                np.clip(result.x[0] + result.x[1:], -20, 20)
            )

        self.feature_rate_ = feature_rates
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        scores = (
            X @ np.log(self.feature_rate_).T
            - self.feature_rate_.sum(axis=1)
            + self.class_log_prior_
        )
        return np.exp(scores - logsumexp(scores, axis=1, keepdims=True))

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]


def labels_to_indices(y, classes):
    lookup = {label: index for index, label in enumerate(classes)}
    return np.asarray([lookup[label] for label in y], dtype=int)


def power_calibrate(probabilities, power):
    """Return Q(c|x) proportional to P(c|x) raised to ``power``."""
    logs = np.log(np.clip(probabilities, 1e-300, 1.0))
    scaled = power * logs
    return np.exp(scaled - logsumexp(scaled, axis=1, keepdims=True))


def fit_calibration_power(y, probabilities, classes):
    true_indices = labels_to_indices(y, classes)

    def objective(power):
        calibrated = power_calibrate(probabilities, power)
        true_probabilities = calibrated[np.arange(len(y)), true_indices]
        return -np.mean(np.log(np.clip(true_probabilities, 1e-300, 1.0)))

    result = minimize_scalar(objective, bounds=(0.05, 5.0), method="bounded")
    if not result.success:
        raise RuntimeError(f"Calibration failed: {result.message}")
    return result.x


def multiclass_brier_score(y, probabilities, classes):
    targets = np.eye(len(classes))[labels_to_indices(y, classes)]
    return np.mean(np.sum((probabilities - targets) ** 2, axis=1))


def confidence_bin_rows(y, probabilities, classes, n_bins):
    predicted = classes[np.argmax(probabilities, axis=1)]
    confidence = probabilities.max(axis=1)
    correct = predicted == y
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for index in range(n_bins):
        lower, upper = edges[index:index + 2]
        mask = (
            (confidence >= lower) & (confidence <= upper)
            if index == n_bins - 1
            else (confidence >= lower) & (confidence < upper)
        )
        if np.any(mask):
            rows.append(
                (lower, upper, mask.sum(), confidence[mask].mean(), correct[mask].mean())
            )
    return rows


def expected_calibration_error(y, probabilities, classes, n_bins):
    rows = confidence_bin_rows(y, probabilities, classes, n_bins)
    return sum(
        (count / len(y)) * abs(mean_confidence - observed_accuracy)
        for _, _, count, mean_confidence, observed_accuracy in rows
    )


def top_k_accuracy(y, probabilities, classes, k):
    true_indices = labels_to_indices(y, classes)
    top_indices = np.argsort(-probabilities, axis=1)[:, :k]
    return np.mean([
        true_indices[row] in top_indices[row] for row in range(len(y))
    ])


def credible_set_statistics(y, probabilities, classes, levels):
    true_indices = labels_to_indices(y, classes)
    sorted_indices = np.argsort(-probabilities, axis=1)
    sorted_probabilities = np.take_along_axis(probabilities, sorted_indices, axis=1)
    cumulative = np.cumsum(sorted_probabilities, axis=1)
    statistics = []
    for level in levels:
        sizes = np.argmax(cumulative >= level, axis=1) + 1
        included = np.asarray([
            true_indices[row] in sorted_indices[row, :sizes[row]]
            for row in range(len(y))
        ])
        statistics.append((level, included.mean(), sizes.mean()))
    return statistics


def print_probability_report(title, y, probabilities, classes, n_bins):
    predicted = classes[np.argmax(probabilities, axis=1)]
    confidence = probabilities.max(axis=1)
    accuracy = accuracy_score(y, predicted)
    print(f"\n{title}\n{'-' * len(title)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y, predicted):.4f}")
    print(f"Mean confidence: {confidence.mean():.4f}")
    print(f"Confidence minus accuracy: {confidence.mean() - accuracy:.4f}")
    print(f"Log loss: {log_loss(y, probabilities, labels=classes):.4f}")
    print(f"Multiclass Brier score: {multiclass_brier_score(y, probabilities, classes):.4f}")
    print(f"Expected calibration error: {expected_calibration_error(y, probabilities, classes, n_bins):.4f}")
    for k in (1, 3, 5):
        if k <= len(classes):
            print(f"Top-{k} accuracy: {top_k_accuracy(y, probabilities, classes, k):.4f}")
    print("\nConfidence bins:")
    print("range       trials  mean confidence  observed accuracy")
    for lower, upper, count, mean_confidence, observed_accuracy in confidence_bin_rows(
        y, probabilities, classes, n_bins
    ):
        print(
            f"{lower:.1f}-{upper:.1f}    {count:5d}        "
            f"{mean_confidence:.4f}             {observed_accuracy:.4f}"
        )
    print("\nCredible-set coverage:")
    print("target level  empirical coverage  mean images in set")
    for level, coverage, mean_size in credible_set_statistics(
        y, probabilities, classes, DEFAULT_COVERAGE_LEVELS
    ):
        print(f"{level:11.2f}  {coverage:18.4f}  {mean_size:18.2f}")
    return predicted


def run(args):
    X, y, locations, spike_shape, injection_time = load_dataset(
        args.nwb_path, args.segment, args.window_start, args.window_end, args.bin_size
    )
    print(f"Selected {X.shape[1]} units")
    for location, count in zip(*np.unique(locations, return_counts=True)):
        print(f"{location}: {count}")
    print(f"Injection time: {injection_time:.4f} seconds")
    print("Spike matrix shape:", spike_shape)
    print(f"\n{args.segment.capitalize()}-injection repeated-image counts:")
    for image, count in zip(*np.unique(y, return_counts=True)):
        print(f"{image}: {count} trials")

    X_development, X_test, y_development, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    relative_calibration_size = args.calibration_size / (1.0 - args.test_size)
    X_train, X_calibration, y_train, y_calibration = train_test_split(
        X_development,
        y_development,
        test_size=relative_calibration_size,
        random_state=args.calibration_random_state,
        stratify=y_development,
    )
    decoder = Pipeline([
        ("remove_constant_units", VarianceThreshold(threshold=0)),
        ("regularized_poisson_nb", RegularizedPoissonNB(
            l2_strength=args.l2_strength, max_iter=args.max_iter
        )),
    ])
    decoder.fit(X_train, y_train)
    classes = decoder.named_steps["regularized_poisson_nb"].classes_
    calibration_raw = decoder.predict_proba(X_calibration)
    calibration_power = fit_calibration_power(
        y_calibration, calibration_raw, classes
    )
    test_raw = decoder.predict_proba(X_test)
    test_calibrated = power_calibrate(test_raw, calibration_power)

    print(f"\n{args.segment.capitalize()}-injection calibrated image decoder")
    print("-" * 50)
    print(f"Total trials: {len(y)}")
    print(f"Training trials: {len(y_train)}")
    print(f"Calibration trials: {len(y_calibration)}")
    print(f"Final test trials: {len(y_test)}")
    print(f"Selected units: {X.shape[1]}")
    print(f"Number of image classes: {len(classes)}")
    print(f"Uniform random chance: {1 / len(classes):.4f}")
    print(f"L2 strength: {args.l2_strength:g}")
    print(f"Learned calibration power: {calibration_power:.4f}")
    print("A power below 1 softens overconfident probabilities.")

    raw_predictions = print_probability_report(
        "Final test set: raw probabilities",
        y_test,
        test_raw,
        classes,
        args.confidence_bins,
    )
    calibrated_predictions = print_probability_report(
        "Final test set: calibrated probabilities",
        y_test,
        test_calibrated,
        classes,
        args.confidence_bins,
    )
    print("\nImage label order:")
    print(classes)
    print("\nCalibrated confusion matrix:")
    print(confusion_matrix(y_test, calibrated_predictions, labels=classes))
    if np.array_equal(raw_predictions, calibrated_predictions):
        print("\nCalibration changed confidence values but not predicted labels.")
    else:
        print("\nCalibration changed at least one label because of numerical ties.")


def main():
    args = parse_args()
    validate_args(args)
    run(args)


if __name__ == "__main__":
    main()
