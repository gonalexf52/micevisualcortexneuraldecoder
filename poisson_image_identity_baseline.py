#!/usr/bin/env python3
"""Baseline Poisson Naive Bayes decoder for natural-image identity.

Extracts 50-250 ms post-stimulus spike counts from an OpenScope Psycode NWB
file, selects quality-controlled visual-cortex units, and predicts which of
eight natural images was presented.

Example:
    python poisson_image_identity_baseline.py --nwb-path session.nwb --segment pre
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from pynwb import NWBHDF5IO
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DEFAULT_BRAIN_REGIONS = (
    "VISp1", "VISp2/3", "VISp4", "VISp5", "VISp6a", "VISp6b",
    "VISam2/3", "VISam4", "VISam5", "VISam6a", "VISam6b",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode repeated natural-image identity from Psycode spike data."
    )
    parser.add_argument("--nwb-path", required=True, type=Path, help="NWB file path.")
    parser.add_argument("--segment", choices=("pre", "post"), default="pre")
    parser.add_argument("--window-start", type=float, default=0.050)
    parser.add_argument("--window-end", type=float, default=0.250)
    parser.add_argument("--bin-size", type=float, default=0.005)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=314159)
    parser.add_argument("--alpha", type=float, default=0.01)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.nwb_path.is_file():
        raise FileNotFoundError(f"NWB file not found: {args.nwb_path}")
    if args.bin_size <= 0:
        raise ValueError("--bin-size must be positive.")
    if args.window_end <= args.window_start:
        raise ValueError("--window-end must be greater than --window-start.")
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1.")
    if args.alpha <= 0:
        raise ValueError("--alpha must be positive.")


def get_unit_location(unit_row) -> str:
    """Assign a unit to the location of its peak waveform channel."""
    waveform_mins = np.min(unit_row["waveform_mean"], axis=0)
    peak_channel_index = np.argmin(waveform_mins)
    return unit_row["electrodes"].iloc[peak_channel_index].location


def select_units(units, brain_regions=DEFAULT_BRAIN_REGIONS):
    """Apply the official Psycode quality criteria and area selection."""
    quality_units = units.query(
        '(firing_range > 8) & '
        '(decoder_label == "sua") & '
        '(presence_ratio > .95)'
    )
    spike_times = []
    locations = []
    for _, unit_row in quality_units.iterrows():
        location = get_unit_location(unit_row)
        if location in brain_regions:
            spike_times.append(unit_row.spike_times)
            locations.append(location)
    return spike_times, locations


def get_spike_matrix(stim_times, units_spike_times, bin_edges):
    """Return a units x trials x time-bins matrix of spike counts."""
    bin_size = np.mean(np.diff(bin_edges))
    matrix = np.zeros(
        (len(units_spike_times), len(stim_times), len(bin_edges) - 1),
        dtype=np.uint8,
    )
    for unit_index, spike_times in enumerate(units_spike_times):
        spike_times = np.asarray(spike_times)
        for trial_index, stim_time in enumerate(stim_times):
            first_time = stim_time + bin_edges[0]
            last_time = stim_time + bin_edges[-1]
            first_spike, last_spike = np.searchsorted(
                spike_times, [first_time, last_time]
            )
            spikes = spike_times[first_spike:last_spike]
            indices = ((spikes - first_time) / bin_size).astype(int)
            for index in indices:
                if 0 <= index < matrix.shape[2]:
                    matrix[unit_index, trial_index, index] += 1
    return matrix


class PoissonNB(ClassifierMixin, BaseEstimator):
    """Independent-Poisson Naive Bayes with additive rate smoothing."""

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if np.any(X < 0):
            raise ValueError("PoissonNB requires nonnegative spike counts.")
        self.classes_, encoded_y = np.unique(y, return_inverse=True)
        self.n_features_in_ = X.shape[1]
        class_counts = np.bincount(encoded_y)
        self.class_log_prior_ = np.log(class_counts / class_counts.sum())
        self.feature_rate_ = np.asarray([
            X[encoded_y == index].mean(axis=0) + self.alpha
            for index in range(len(self.classes_))
        ])
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        scores = (
            X @ np.log(self.feature_rate_).T
            - self.feature_rate_.sum(axis=1)
            + self.class_log_prior_
        )
        scores -= scores.max(axis=1, keepdims=True)
        probabilities = np.exp(scores)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]


def load_dataset(nwb_path, segment, window_start, window_end, bin_size):
    """Load one NWB session and return counts, labels, and metadata."""
    io = NWBHDF5IO(str(nwb_path), mode="r", load_namespaces=True)
    try:
        nwb = io.read()
        units_spike_times, unit_locations = select_units(nwb.units.to_dataframe())
        table = nwb.intervals["behavior_presentations"]
        all_times = np.asarray(table["start_time"][:], dtype=float)
        all_names = np.asarray(table["image_name"][:]).astype(str)
        omitted_values = np.asarray(table["omitted"][:])
        omitted = np.asarray([
            str(value).strip().lower() == "true" for value in omitted_values
        ], dtype=bool)
        injection_time = float(
            nwb.intervals["injection_times"].to_dataframe()["start_time"].iloc[0]
        )
        valid = (
            ~omitted & (all_names != "") & (all_names != "nan")
            & (all_names != "None")
        )
        repeated = np.zeros(len(all_names), dtype=bool)
        valid_indices = np.flatnonzero(valid)
        if len(valid_indices) > 1:
            current = valid_indices[1:]
            previous = valid_indices[:-1]
            repeated[current] = all_names[current] == all_names[previous]
        segment_mask = (
            all_times < injection_time if segment == "pre"
            else all_times >= injection_time
        )
        selected = valid & repeated & segment_mask
        stim_times = all_times[selected]
        image_names = all_names[selected]
        if not units_spike_times:
            raise ValueError("No units passed the Psycode selection criteria.")
        if len(stim_times) == 0:
            raise ValueError(f"No repeated images found in the {segment} segment.")
        n_bins = int(round((window_end - window_start) / bin_size))
        edges = np.linspace(window_start, window_end, n_bins + 1)
        spike_matrix = get_spike_matrix(stim_times, units_spike_times, edges)
    finally:
        io.close()
    return (
        spike_matrix.sum(axis=2).T,
        image_names,
        unit_locations,
        spike_matrix.shape,
        injection_time,
    )


def run(args: argparse.Namespace) -> None:
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    decoder = Pipeline([
        ("remove_constant_units", VarianceThreshold(threshold=0)),
        ("poisson_nb", PoissonNB(alpha=args.alpha)),
    ])
    decoder.fit(X_train, y_train)
    train_predictions = decoder.predict(X_train)
    predictions = decoder.predict(X_test)
    probabilities = decoder.predict_proba(X_test)
    labels = decoder.named_steps["poisson_nb"].classes_
    accuracy = accuracy_score(y_test, predictions)
    confidence = probabilities.max(axis=1)

    print(f"\n{args.segment.capitalize()}-injection natural-image baseline")
    print("-" * 45)
    print(f"Trials: {len(y)}")
    print(f"Selected units: {X.shape[1]}")
    print(f"Number of image classes: {len(labels)}")
    print(f"Uniform random chance: {1 / len(labels):.4f}")
    print(f"Training accuracy: {accuracy_score(y_train, train_predictions):.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test balanced accuracy: {balanced_accuracy_score(y_test, predictions):.4f}")
    print(f"Mean test confidence: {confidence.mean():.4f}")
    print(f"Confidence minus accuracy: {confidence.mean() - accuracy:.4f}")
    print("\nImage label order:")
    print(labels)
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=labels))


def main() -> None:
    args = parse_args()
    validate_args(args)
    run(args)


if __name__ == "__main__":
    main()
