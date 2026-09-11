"""
predict.py — CLI prediction script for left vs right fist motor imagery.

Usage:
    python predict.py /path/to/subject_run.edf

Loads a single raw EDF file, applies the same preprocessing used in
training, aligns it via Riemannian Alignment, predicts left/right per trial
using MDM (nearest Riemannian mean), and reports accuracy and a confusion
matrix against the file's own T1/T2 annotations as a consistency check.
"""
import sys
import pickle
import numpy as np
from mne.io import read_raw_edf
from mne.datasets import eegbci
from sklearn.metrics import confusion_matrix
import mne

from src.preprocess import preprocess_raw, epoch_subject, EXCLUDE_CHANNELS
from src.features import riemannian_align

FINAL_MODEL = "04_riemannian_mdm_aligned"
MODEL_PATH = f"models/saved/{FINAL_MODEL}_final.pkl"


def load_single_edf(path):
    raw = read_raw_edf(path, preload=True, verbose=False)
    eegbci.standardize(raw)
    raw.set_montage(mne.channels.make_standard_montage('standard_1005'), on_missing='warn')
    return raw


def compute_erd(epochs):
    """Log power ratio, task window vs pre-stimulus baseline, at C3/C4 —
    the one number that tells you whether this subject's data actually
    carries real motor imagery signal, independent of what the classifier
    predicts. Strongly negative = real desynchronization present. Near zero
    or positive = weak/absent signal, and low accuracy here isn't a pipeline
    problem, it's the subject. Computed separately from the classifier —
    not fed into the model, purely a diagnostic printed alongside it."""
    if 'C3' not in epochs.ch_names or 'C4' not in epochs.ch_names:
        return None

    sfreq = epochs.info['sfreq']
    base_slice = slice(0, int(1.0 * sfreq))
    task_slice = slice(int(1.5 * sfreq), int(3.5 * sfreq))
    data = epochs.get_data(copy=True)
    c3, c4 = epochs.ch_names.index('C3'), epochs.ch_names.index('C4')

    erd_c3 = np.log10(np.var(data[:, c3, task_slice]) / np.var(data[:, c3, base_slice]))
    erd_c4 = np.log10(np.var(data[:, c4, task_slice]) / np.var(data[:, c4, base_slice]))
    return erd_c3, erd_c4


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py /path/to/file.edf")
        sys.exit(1)

    edf_path = sys.argv[1]
    print(f"Loading {edf_path}...")
    raw = load_single_edf(edf_path)

    raw, detected_bads = preprocess_raw(raw)
    print(f"Dropped frontal channels: {EXCLUDE_CHANNELS}")
    print(f"Filtered 8-30Hz, CAR applied. Bad channels detected: {detected_bads}")

    epochs, n_dropped = epoch_subject(raw)
    if epochs is None:
        print("No T1/T2 annotations found in this file — cannot extract trials.")
        sys.exit(1)
    print(f"Extracted {len(epochs)} trials ({n_dropped} dropped by amplitude rejection)")

    erd = compute_erd(epochs)
    if erd is not None:
        erd_c3, erd_c4 = erd
        print(f"ERD (log power ratio, negative = real desync): C3={erd_c3:+.3f}  C4={erd_c4:+.3f}")

    task_mask = (epochs.times >= 0.5) & (epochs.times <= 2.5)
    X = epochs.get_data(copy=True)[:, :, task_mask].astype(np.float32)
    y_true = np.where(epochs.events[:, 2] == epochs.event_id['left'], 0, 1)

    X_aligned, _ = riemannian_align(X)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_aligned)

    acc = (y_pred == y_true).mean()
    print(f"\nAccuracy against embedded T1/T2 ground truth: {acc*100:.1f}%")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print("\nConfusion matrix (rows=true, cols=predicted):")
    print("            pred_left  pred_right")
    print(f"true_left      {cm[0,0]:>4d}       {cm[0,1]:>4d}")
    print(f"true_right     {cm[1,0]:>4d}       {cm[1,1]:>4d}")


if __name__ == "__main__":
    main()