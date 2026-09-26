"""Utilities for loading and preprocessing the PhysioNet EEGBCI motor-imagery dataset."""

import numpy as np


def load_eeg_from_mne(subject_id, runs, dataset="eegbci"):
    """Load EEG data through MNE."""
    import mne
    from mne.datasets import eegbci

    if dataset != "eegbci":
        raise ValueError(f"Unsupported dataset: {dataset}")

    fnames = eegbci.load_data(subject_id, runs)
    raws = [mne.io.read_raw_edf(path, preload=True, verbose=False) for path in fnames]
    raw = mne.concatenate_raws(raws)
    eegbci.standardize(raw)
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, on_missing="ignore")
    return raw


def preprocess(raw, l_freq=1.0, h_freq=40.0):
    """Band-pass filter a copy of an MNE Raw object."""
    filtered = raw.copy()
    filtered.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
    return filtered


def extract_epochs(raw, event_id=None, tmin=0.0, tmax=4.0):
    """Extract motor-imagery epochs, excluding T0/rest by default."""
    import mne

    events, event_dict = mne.events_from_annotations(raw, verbose=False)
    if event_id is None:
        event_id = {
            label: event_dict[label]
            for label in ("T1", "T2")
            if label in event_dict
        }
        if len(event_id) != 2:
            raise RuntimeError(f"Expected T1/T2 annotations, found: {event_dict}")

    return mne.Epochs(
        raw,
        events,
        event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        picks="eeg",
        reject_by_annotation=True,
        verbose=False,
    )


def epochs_to_array(epochs):
    """Convert epochs to NumPy arrays and remap labels to 0..K-1."""
    x = epochs.get_data()
    raw_labels = epochs.events[:, -1]
    unique = np.unique(raw_labels)
    label_map = {label: idx for idx, label in enumerate(unique)}
    y = np.array([label_map[label] for label in raw_labels], dtype=int)
    return x, y
