import os, sys, pickle, argparse, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from src.preprocess import get_or_build_usable_epochs, MOTOR_PICKS
from src.dataset import build_subject_dataset, get_loso_split
from src.features import build_csp_pipeline
from src.logging_utils import log_experiment

MODEL_NAME = "02_csp_lda"
os.makedirs("assets", exist_ok=True)
os.makedirs("models/saved", exist_ok=True)
os.makedirs("assets/per_subject", exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_subjects", type=int, default=80)
    parser.add_argument("--start_subject", type=int, default=1)
    parser.add_argument("--channel_scope", choices=["all", "motor"], default="all")
    args = parser.parse_args()

    subjects = range(args.start_subject, args.start_subject + args.n_subjects)
    usable = get_or_build_usable_epochs(subjects)
    subject_dataset = build_subject_dataset(usable)
    cv_subjects = list(subject_dataset.keys())

    sample_epochs = usable[cv_subjects[0]]
    if args.channel_scope == "motor":
        channel_idx = [sample_epochs.ch_names.index(ch) for ch in MOTOR_PICKS
                       if ch in sample_epochs.ch_names]
        print(f"Restricting to {len(channel_idx)} motor-strip channels")
    else:
        channel_idx = None
        print(f"Using all {len(sample_epochs.ch_names)} channels")

    def crop(X):
        return X[:, channel_idx, :] if channel_idx is not None else X

    print(f"Running {len(cv_subjects)}-fold LOSO for {MODEL_NAME} ({args.channel_scope})...")
    per_subject_acc = {}
    for idx, test_subj in enumerate(cv_subjects):
        X_tr, y_tr, X_te, y_te = get_loso_split(subject_dataset, cv_subjects, test_subj)
        clf = build_csp_pipeline()
        clf.fit(crop(X_tr), y_tr)
        per_subject_acc[test_subj] = accuracy_score(y_te, clf.predict(crop(X_te)))
        if (idx + 1) % 20 == 0:
            print(f"  {idx+1}/{len(cv_subjects)} folds complete")

    print(f"\n{'Subject':<10} {'Accuracy':>10}")
    print("-" * 22)
    for subj, acc in per_subject_acc.items():
        print(f"S{subj:03d}     {acc*100:>8.2f}%")

    accs = list(per_subject_acc.values())
    print(f"\n{MODEL_NAME} ({args.channel_scope}): mean {np.mean(accs)*100:.2f}% "
          f"± {np.std(accs)*100:.2f}% (min {min(accs)*100:.2f}%, max {max(accs)*100:.2f}%)")

    sorted_accs = sorted(accs)
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(sorted_accs)), sorted_accs)
    plt.axhline(0.5, linestyle='--', color='gray', label='chance')
    plt.axhline(np.mean(accs), linestyle='--', color='red', label=f'mean ({np.mean(accs)*100:.1f}%)')
    plt.xlabel("subjects, sorted by accuracy"); plt.ylabel("accuracy")
    plt.title(f"{MODEL_NAME} ({args.channel_scope}): LOSO accuracy, {len(cv_subjects)} subjects")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"assets/loso_accuracy_{MODEL_NAME}_{args.channel_scope}.png"); plt.close()

    with open(f"assets/per_subject/{MODEL_NAME}_{args.channel_scope}.json", "w") as f:
        json.dump({str(k): v for k, v in per_subject_acc.items()}, f, indent=2)

    X_all = np.concatenate([subject_dataset[s][0] for s in cv_subjects])
    y_all = np.concatenate([subject_dataset[s][1] for s in cv_subjects])
    final_clf = build_csp_pipeline()
    final_clf.fit(crop(X_all), y_all)
    with open(f"models/saved/{MODEL_NAME}_{args.channel_scope}_final.pkl", "wb") as f:
        pickle.dump({"model": final_clf, "channel_idx": channel_idx}, f)

    log_experiment(MODEL_NAME, accs, len(cv_subjects), f"{len(cv_subjects)}-fold LOSO",
                    f"shrinkage LDA, channels={args.channel_scope}")
    print("Done.")


if __name__ == "__main__":
    main()