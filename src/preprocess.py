"""
Preprocessing pipeline: load raw EDF, clean the signal 
(channel drop, bandpass, bad-channel detection, CAR, interpolation), epoch trials.
Used identically by every model training script and by predict.py, so
training and inference always go through the exact same transformation.
"""
import os
import pickle

import mne
import numpy as np
from mne.io import read_raw_edf
from mne.datasets import eegbci

IMAGINED_RUNS = [4, 8, 12]
EXCLUDE_CHANNELS = ['Fp1', 'Fp2', 'Fpz', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8']
MOTOR_PICKS = ['FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C3', 'C1', 'Cz', 'C2', 'C4',
               'CP3', 'CP1', 'CPz', 'CP2', 'CP4']


def load_subject(subject, runs, data_root=None):
    """Load and concatenate a subject's runs, standardize channel names.
    data_root=None fetches via eegbci.load_data; pass a local PhysioNet
    folder path to read directly from disk instead."""
    if data_root:
        paths = [f"{data_root}/S{subject:03d}/S{subject:03d}R{r:02d}.edf" for r in runs]
    else:
        paths = eegbci.load_data(subject, runs)

    raws = [read_raw_edf(p, preload=True, verbose=False) for p in paths]
    raw = mne.concatenate_raws(raws)
    eegbci.standardize(raw)
    raw.set_montage(mne.channels.make_standard_montage('standard_1005'), on_missing='warn')
    return raw


def find_bad_channels_by_variance(raw, thresh_multiplier=10.0):
    """Flags channels whose variance exceeds thresh_multiplier times the median."""
    data = raw.get_data()
    ch_vars = np.var(data, axis=1)
    median_var = np.median(ch_vars)
    bad_idx = np.where(ch_vars > median_var * thresh_multiplier)[0]
    return [raw.ch_names[idx] for idx in bad_idx]


def preprocess_raw(raw):
    """Channel drop, bandpass, bad-channel detection, CAR, interpolation.
    Returns the cleaned Raw — does not epoch, since predict.py may want
    epochs built differently than training did for a given model."""
    available_to_drop = [ch for ch in EXCLUDE_CHANNELS if ch in raw.ch_names]
    raw.drop_channels(available_to_drop)

    raw.filter(8., 30., method='fir', verbose=False)

    detected_bads = find_bad_channels_by_variance(raw, thresh_multiplier=10.0)
    raw.info['bads'] = detected_bads
    raw.set_eeg_reference('average', verbose=False)
    if len(raw.info['bads']) > 0:
        raw.interpolate_bads(reset_bads=True, mode='accurate', verbose=False)

    return raw, detected_bads


def epoch_subject(raw, tmin=-1.0, tmax=2.5, reject_uv=200.0):
    """Epochs T1/T2 trials, applies the motor-channel amplitude rejection gate.
    Returns (epochs, n_dropped) or (None, None) if T1/T2 annotations are absent —
    the latter matters for predict.py, since a new subject's file might not
    have usable annotations at all."""
    events, eid = mne.events_from_annotations(raw, verbose=False)
    if 'T1' not in eid or 'T2' not in eid:
        return None, None

    epochs = mne.Epochs(
        raw, events, event_id={'left': eid['T1'], 'right': eid['T2']},
        tmin=tmin, tmax=tmax, baseline=None, preload=True, verbose=False
    )

    motor_idx = [epochs.ch_names.index(ch) for ch in MOTOR_PICKS if ch in epochs.ch_names]
    task_mask = (epochs.times >= 0.5) & (epochs.times <= 2.5)
    motor_task_data = epochs.get_data(copy=True)[:, motor_idx][:, :, task_mask]

    keep = np.abs(motor_task_data).max(axis=(1, 2)) <= reject_uv * 1e-6
    n_before = len(epochs)
    epochs = epochs[keep]
    return epochs, n_before - len(epochs)


def build_usable_epochs(subjects, runs=IMAGINED_RUNS, min_trials=10):
    """Runs load → clean → epoch across a subject list, skipping any subject
    that fails to load, has no T1/T2 annotations, or ends up with too few
    surviving trials after rejection. Every model script calls this same
    function so every model is evaluated on identically-cleaned data."""
    usable = {}
    for s in subjects:
        try:
            raw = load_subject(s, runs)
        except Exception as e:
            print(f"S{s:03d}: failed to load ({e}), skipping")
            continue

        raw, detected_bads = preprocess_raw(raw)
        epochs, n_dropped = epoch_subject(raw)

        if epochs is None:
            print(f"S{s:03d}: missing T1/T2 annotations, skipping")
            continue
        if len(epochs) < min_trials:
            print(f"S{s:03d}: only {len(epochs)} trials survived, skipping")
            continue

        print(f"S{s:03d}: {len(epochs)} kept | {n_dropped} dropped | Bads: {detected_bads}")
        usable[s] = epochs

    print(f"\nUsable subjects: {len(usable)} / {len(subjects)}")
    return usable


def get_or_build_usable_epochs(subjects, runs=IMAGINED_RUNS, min_trials=10,
                                 cache_path="usable_epochs_cache.pkl"):
    """Maintains one growing cache, keyed by subject number. Only subjects
    not already cached get processed; the cache is updated and saved after
    each run, so requesting a different subject range later reuses whatever
    overlaps rather than reprocessing everything from scratch."""
    subjects = list(subjects)

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        print(f"Loaded cache with {len(cache)} subjects already processed.")
    else:
        cache = {}

    missing = [s for s in subjects if s not in cache]
    if missing:
        print(f"Processing {len(missing)} new subject(s): {missing}")
        new_epochs = build_usable_epochs(missing, runs=runs, min_trials=min_trials)
        cache.update(new_epochs)
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
        print(f"Cache now holds {len(cache)} subjects, saved to {cache_path}")
    else:
        print("All requested subjects already cached, nothing to process.")

    return {s: cache[s] for s in subjects if s in cache}