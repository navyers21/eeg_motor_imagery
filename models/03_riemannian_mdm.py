import os, sys, pickle, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from src.preprocess import get_or_build_usable_epochs
from src.dataset import build_subject_dataset, get_loso_split
from src.features import build_mdm_pipeline
from src.logging_utils import log_experiment

MODEL_NAME = "03_riemannian_mdm"
SUBJECTS = range(1, 81)
os.makedirs("assets", exist_ok=True)
os.makedirs("models/saved", exist_ok=True)
os.makedirs("assets/per_subject", exist_ok=True)


def main():
    usable = get_or_build_usable_epochs(SUBJECTS)
    subject_dataset = build_subject_dataset(usable)
    cv_subjects = list(subject_dataset.keys())

    print(f"Running {len(cv_subjects)}-fold LOSO for {MODEL_NAME}...")
    per_subject_acc = {}
    for idx, test_subj in enumerate(cv_subjects):
        X_tr, y_tr, X_te, y_te = get_loso_split(subject_dataset, cv_subjects, test_subj)
        clf = build_mdm_pipeline()
        clf.fit(X_tr, y_tr)
        per_subject_acc[test_subj] = accuracy_score(y_te, clf.predict(X_te))
        if (idx + 1) % 20 == 0:
            print(f"  {idx+1}/{len(cv_subjects)} folds complete")

    accs = list(per_subject_acc.values())
    print(f"{MODEL_NAME}: mean {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}% "
          f"(min {min(accs)*100:.2f}%, max {max(accs)*100:.2f}%)")

    with open(f"assets/per_subject/{MODEL_NAME}.json", "w") as f:
        json.dump({str(k): v for k, v in per_subject_acc.items()}, f, indent=2)

    sorted_accs = sorted(accs)
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(sorted_accs)), sorted_accs)
    plt.axhline(0.5, linestyle='--', color='gray', label='chance')
    plt.axhline(np.mean(accs), linestyle='--', color='red', label=f'mean ({np.mean(accs)*100:.1f}%)')
    plt.xlabel("subjects, sorted by accuracy"); plt.ylabel("accuracy")
    plt.title(f"{MODEL_NAME}: LOSO accuracy across {len(cv_subjects)} subjects")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"assets/loso_accuracy_{MODEL_NAME}.png"); plt.close()

    X_all = np.concatenate([subject_dataset[s][0] for s in cv_subjects])
    y_all = np.concatenate([subject_dataset[s][1] for s in cv_subjects])
    final_clf = build_mdm_pipeline()
    final_clf.fit(X_all, y_all)
    with open(f"models/saved/{MODEL_NAME}_final.pkl", "wb") as f:
        pickle.dump(final_clf, f)

    log_experiment(MODEL_NAME, accs, len(cv_subjects), "80-fold LOSO",
                    "MDM, no alignment — baseline for comparing against aligned version")
    print("Done.")


if __name__ == "__main__":
    main()