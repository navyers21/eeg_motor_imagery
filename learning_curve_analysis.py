"""
learning_curve_analysis.py — accuracy vs training-set-size for classical
models, including MDM. Substitutes for a loss curve, since these are
closed-form fits with no epoch-based training loop. Diagnoses overfitting
(train/test gap) and whether more subjects would help.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from src.preprocess import get_or_build_usable_epochs, MOTOR_PICKS
from src.dataset import build_subject_dataset
from src.features import extract_band_power, build_csp_pipeline

TRAIN_POOL = list(range(1, 61))
TEST_SUBJECTS = list(range(61, 81))
TRAIN_SIZES = [5, 10, 20, 30, 40, 50, 60]
RNG = np.random.default_rng(seed=42)


def pool(dataset, subjects):
    X = np.concatenate([dataset[s][0] for s in subjects])
    y = np.concatenate([dataset[s][1] for s in subjects])
    return X, y


def run_curve(name, fit_predict_fn, subject_dataset):
    print(f"\n{name}:")
    train_accs, test_accs = [], []
    X_test, y_test = pool(subject_dataset, TEST_SUBJECTS)

    for size in TRAIN_SIZES:
        train_subjects = RNG.choice(TRAIN_POOL, size=size, replace=False)
        X_train, y_train = pool(subject_dataset, train_subjects)

        train_pred, test_pred = fit_predict_fn(X_train, y_train, X_train, X_test)
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        print(f"  n_train_subjects={size:3d}: train_acc={train_acc*100:.2f}%  "
              f"test_acc={test_acc*100:.2f}%")

    return train_accs, test_accs


def main():
    usable = get_or_build_usable_epochs(range(1, 81))
    subject_dataset = build_subject_dataset(usable)

    sample_epochs = usable[TRAIN_POOL[0]]
    motor_idx = [sample_epochs.ch_names.index(ch) for ch in MOTOR_PICKS
                 if ch in sample_epochs.ch_names]

    def fit_predict_band_power(X_tr, y_tr, X_train_eval, X_test_eval):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        clf = make_pipeline(StandardScaler(), LDA(solver='eigen', shrinkage='auto'))
        clf.fit(extract_band_power(X_tr, channel_idx=motor_idx), y_tr)
        return (clf.predict(extract_band_power(X_train_eval, channel_idx=motor_idx)),
                clf.predict(extract_band_power(X_test_eval, channel_idx=motor_idx)))

    def fit_predict_csp(X_tr, y_tr, X_train_eval, X_test_eval):
        crop = lambda X: X[:, motor_idx, :]
        clf = build_csp_pipeline()
        clf.fit(crop(X_tr), y_tr)
        return clf.predict(crop(X_train_eval)), clf.predict(crop(X_test_eval))

    def fit_predict_mdm(X_tr, y_tr, X_train_eval, X_test_eval):
        from src.features import build_mdm_pipeline
        clf = build_mdm_pipeline()
        clf.fit(X_tr, y_tr)
        return clf.predict(X_train_eval), clf.predict(X_test_eval)
    
    def fit_predict_mdm_aligned(X_tr, y_tr, X_train_eval, X_test_eval):
        from src.features import riemannian_align, build_mdm_pipeline
        X_tr_a, ref = riemannian_align(X_tr)
        X_train_eval_a, _ = riemannian_align(X_train_eval, ref_cov=ref)
        X_test_eval_a, _ = riemannian_align(X_test_eval, ref_cov=ref)
        clf = build_mdm_pipeline(pre_aligned=True)
        clf.fit(X_tr_a, y_tr)
        return clf.predict(X_train_eval_a), clf.predict(X_test_eval_a)    
    
    model_configs = [
        ("01_band_power_lda", fit_predict_band_power),
        ("02_csp_lda", fit_predict_csp),
        ("03_riemannian_mdm", fit_predict_mdm),  # add this variant if you want it plotted
        ("04_riemannian_mdm_aligned", fit_predict_mdm_aligned),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (name, fn) in zip(axes.flat, model_configs):
        train_accs, test_accs = run_curve(name, fn, subject_dataset)
        ax.plot(TRAIN_SIZES, train_accs, 'o-', label='train accuracy (resubstitution)')
        ax.plot(TRAIN_SIZES, test_accs, 's-', label='held-out accuracy (20 subjects)')
        ax.axhline(0.5, linestyle='--', color='gray', alpha=0.5)
        ax.set_xlabel("number of training subjects")
        ax.set_ylabel("accuracy")
        ax.set_title(name)
        ax.legend()
        ax.set_ylim(0.3, 1.0)

    plt.tight_layout()
    os.makedirs("assets", exist_ok=True)
    plt.savefig("assets/learning_curves_all_models.png")
    print("\nSaved assets/learning_curves_all_models.png")


if __name__ == "__main__":
    main()