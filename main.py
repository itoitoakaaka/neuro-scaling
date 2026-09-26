"""Command-line entry point for the neuro-scaling demo."""

import sys
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from data_loader import load_eeg_from_mne, preprocess, extract_epochs, epochs_to_array
from scaling_analysis import compute_scaling_curve, fit_power_law, plot_scaling_curve
from decoder import EEGDecoder


SUBJECTS = list(range(1, 6))
RUNS = [4, 8, 12]


def load_subject(subject_id):
    raw = load_eeg_from_mne(subject_id=subject_id, runs=RUNS)
    raw = preprocess(raw)
    epochs = extract_epochs(raw)
    return epochs_to_array(epochs)


def download_data():
    raw = load_eeg_from_mne(subject_id=1, runs=RUNS)
    print(f"Downloaded subject 1: {len(raw.ch_names)} channels, {raw.n_times} samples")
    return raw


def run_scaling_analysis():
    """Evaluate a simple band-power decoder across increasing data fractions."""
    all_X, all_y = [], []

    for subject in SUBJECTS:
        print(f"Processing subject {subject}...")
        X, y = load_subject(subject)
        X_feat = np.log10(np.mean(X ** 2, axis=-1) + np.finfo(float).eps)
        all_X.append(X_feat)
        all_y.append(y)

    X_all = np.vstack(all_X)
    y_all = np.hstack(all_y)

    fractions, scores_mean, scores_std = compute_scaling_curve(
        X_all,
        y_all,
        random_state=42,
    )
    params = fit_power_law(fractions, scores_mean)
    plot_scaling_curve(fractions, scores_mean, scores_std, params)


def run_decoding():
    """Train on several subjects and evaluate on held-out subjects."""
    xs, ys, groups = [], [], []

    for subject in SUBJECTS:
        X, y = load_subject(subject)
        xs.append(X)
        ys.append(y)
        groups.append(np.full(len(y), subject, dtype=int))

    X_all = np.vstack(xs)
    y_all = np.hstack(ys)
    group_all = np.hstack(groups)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X_all, y_all, groups=group_all))

    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]

    decoder = EEGDecoder(
        n_channels=X_all.shape[1],
        n_times=X_all.shape[2],
        n_classes=len(np.unique(y_all)),
    )
    decoder.build_model()
    decoder.train(X_train, y_train, epochs=30)
    accuracy = decoder.evaluate(X_test, y_test)

    train_subjects = sorted(np.unique(group_all[train_idx]).tolist())
    test_subjects = sorted(np.unique(group_all[test_idx]).tolist())
    print(f"Train subjects: {train_subjects}")
    print(f"Test subjects: {test_subjects}")
    print(f"Held-out-subject accuracy: {accuracy:.3f}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py [download|scaling|decode|all]")
        raise SystemExit(1)

    command = sys.argv[1]
    if command == "download":
        download_data()
    elif command == "scaling":
        run_scaling_analysis()
    elif command == "decode":
        run_decoding()
    elif command == "all":
        download_data()
        run_scaling_analysis()
        run_decoding()
    else:
        print(f"Unknown command: {command}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
