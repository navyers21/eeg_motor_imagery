import os, sys, pickle, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from src.preprocess import get_or_build_usable_epochs, MOTOR_PICKS
from src.dataset import build_subject_dataset, get_loso_split
from src.features import extract_band_power
from src.logging_utils import log_experiment

MODEL_NAME = "01_band_power_lda"
os.makedirs("assets", exist_ok=True)
os.makedirs("models/saved", exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_subjects", type=int, default=80)
    parser.add_argument("--start_subject", type=int, default=1)
    parser.add_argument("--channel_scope", choices=["all", "motor"], default="all",
                         help="'motor' restricts band power to sensorimotor channels only")
    args = parser.parse_args()

    subjects = range(args.start_subject, args.start_subject + args.n_subjects)
    usable = get_or_build_usable_epochs(subjects)
    subject_dataset = build_subject_dataset(usable)
    cv_subjects = list(subject_dataset.keys())

    # resolve channel indices once, from any subject's epochs (channel order
    # is identical across subjects post-preprocessing)
    sample_epochs = usable[cv_subjects[0]]
    if args.channel_scope == "motor":
        channel_idx = [sample_epochs.ch_names.index(ch) for ch in MOTOR_PICKS
                       if ch in sample_epochs.ch_names]
        print(f"Restricting to {len(channel_idx)} motor-strip channels")
    else:
        channel_idx = None
        print(f"Using all {len(sample_epochs.ch_names)} channels")

    def make_clf():
        return make_pipeline(StandardScaler(), LDA(solver='eigen', shrinkage='auto'))

    print(f"Running {len(cv_subjects)}-fold LOSO for {MODEL_NAME} ({args.channel_scope} channels)...")
    per_subject_acc = {}
    for idx, test_subj in enumerate(cv_subjects):
        X_tr, y_tr, X_te, y_te = get_loso_split(subject_dataset, cv_subjects, test_subj)
        clf = make_clf()
        clf.fit(extract_band_power(X_tr, channel_idx=channel_idx), y_tr)
        y_pred = clf.predict(extract_band_power(X_te, channel_idx=channel_idx))
        per_subject_acc[test_subj] = accuracy_score(y_te, y_pred)
        if (idx + 1) % 20 == 0:
            print(f"  {idx+1}/{len(cv_subjects)} folds complete")

    print(f"\n{'Subject':<10} {'Accuracy':>10}")
    print("-" * 22)
    for subj, acc in per_subject_acc.items():
        print(f"S{subj:03d}     {acc*100:>8.2f}%")

    accs = list(per_subject_acc.values())
    print(f"\n{MODEL_NAME} ({args.channel_scope}): mean {np.mean(accs)*100:.2f}% "
          f"± {np.std(accs)*100:.2f}% (min {min(accs)*100:.2f}%, max {max(accs)*100:.2f}%)")

    import json
    os.makedirs("assets/per_subject", exist_ok=True)
    with open(f"assets/per_subject/{MODEL_NAME}_{args.channel_scope}.json", "w") as f:
        json.dump({str(k): v for k, v in per_subject_acc.items()}, f, indent=2)

    sorted_accs = sorted(accs)
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(sorted_accs)), sorted_accs)
    plt.axhline(0.5, linestyle='--', color='gray', label='chance')
    plt.axhline(np.mean(accs), linestyle='--', color='red', label=f'mean ({np.mean(accs)*100:.1f}%)')
    plt.xlabel("subjects, sorted by accuracy"); plt.ylabel("accuracy")
    plt.title(f"{MODEL_NAME} ({args.channel_scope}): LOSO accuracy, {len(cv_subjects)} subjects")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"assets/loso_accuracy_{MODEL_NAME}_{args.channel_scope}.png"); plt.close()

    X_all = np.concatenate([subject_dataset[s][0] for s in cv_subjects])
    y_all = np.concatenate([subject_dataset[s][1] for s in cv_subjects])
    final_clf = make_clf()
    final_clf.fit(extract_band_power(X_all, channel_idx=channel_idx), y_all)
    with open(f"models/saved/{MODEL_NAME}_{args.channel_scope}_final.pkl", "wb") as f:
        pickle.dump({"model": final_clf, "channel_idx": channel_idx}, f)

    log_experiment(MODEL_NAME, accs, len(cv_subjects), f"{len(cv_subjects)}-fold LOSO",
                    f"log-power, StandardScaler, shrinkage LDA, channels={args.channel_scope}")
    print("Done.")


if __name__ == "__main__":
    main()