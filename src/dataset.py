"""
Turns preprocessed epochs into model-ready arrays, and provides the shared
LOSO splitting logic. Every model script in models/ imports get_loso_split
from here rather than reimplementing it 
"""
import numpy as np

def build_subject_dataset(usable, tmin=0.5, tmax=2.5):
    """Converts {subject: epochs} into {subject: (X, y)} arrays, cropped to
    the active task window. usable comes from preprocess.build_usable_epochs().
    """
    subject_dataset = {}
    for s, epochs in usable.items():
        task_mask = (epochs.times >= tmin) & (epochs.times <= tmax)
        X = epochs.get_data(copy=True)[:, :, task_mask].astype(np.float32)
        y = np.where(epochs.events[:, 2] == epochs.event_id['left'], 0, 1)
        subject_dataset[s] = (X, y)
    return subject_dataset


def get_loso_split(dataset, train_subject_pool, test_subject_id):
    """Pools every subject in train_subject_pool except test_subject_id for
    training; keeps test_subject_id fully isolated as the held-out fold.
    """
    X_train_list, y_train_list = [], []
    X_test, y_test = None, None

    for subj in train_subject_pool:
        X, y = dataset[subj]
        if subj == test_subject_id:
            X_test, y_test = X, y
        else:
            X_train_list.append(X)
            y_train_list.append(y)

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    return X_train, y_train, X_test, y_test