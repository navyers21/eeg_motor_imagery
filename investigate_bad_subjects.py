"""
investigate_bad_subjects.py — cross-model bad-subject detection + preprocessing diagnostics.

Two stages:
  1. Pool every assets/per_subject/*.json file that exists so far and find
     subjects that are consistently bad across model types, not just one.
  2. For the worst subjects, re-run the actual preprocessing steps and print
     what's structurally different about them — bad channels detected,
     task-window amplitude range, trial counts, ERD strength — so you have
     concrete evidence for what (if anything) to change in preprocess.py,
     rather than guessing from an accuracy number alone.

Run from the repo root: python investigate_bad_subjects.py
"""
import os
import sys
import json
import glob
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from src.preprocess import load_subject, preprocess_raw, epoch_subject, IMAGINED_RUNS

N_WORST = 8  # how many subjects to deep-dive on


def load_all_per_subject_results():
    files = glob.glob("assets/per_subject/*.json")
    if not files:
        print("No files found in assets/per_subject/ — run at least one model script first.")
        sys.exit(1)

    model_accs = {}
    for f in files:
        name = os.path.basename(f).replace(".json", "")
        with open(f) as fh:
            model_accs[name] = {int(k): v for k, v in json.load(fh).items()}
    return model_accs


def stage_1_cross_model_table(model_accs):
    print(f"Pooling {len(model_accs)} model result files: {sorted(model_accs.keys())}\n")

    all_subjects = sorted(set().union(*[set(a.keys()) for a in model_accs.values()]))
    rows = []
    for subj in all_subjects:
        vals = [model_accs[name].get(subj, np.nan) for name in model_accs]
        valid_vals = [v for v in vals if not np.isnan(v)]
        mean_acc = np.mean(valid_vals) if valid_vals else np.nan
        below_chance_count = sum(1 for v in valid_vals if v < 0.5)
        rows.append((subj, vals, mean_acc, below_chance_count, len(valid_vals)))

    rows.sort(key=lambda r: r[2])  # worst mean first

    model_names = sorted(model_accs.keys())
    header = f"{'Subject':<10}" + "".join(f"{n[:18]:>20}" for n in model_names) + f"{'mean':>10}{'<chance':>10}"
    print(header)
    print("-" * len(header))
    for subj, vals, mean_acc, below_chance, n_valid in rows[:20]:
        row_str = "".join(f"{v*100:>19.2f}%" if not np.isnan(v) else f"{'—':>20}" for v in vals)
        print(f"S{subj:03d}     {row_str}{mean_acc*100:>9.2f}%{below_chance:>7}/{n_valid}")

    return rows


def stage_2_investigate_worst(rows, n_worst=N_WORST):
    worst_subjects = [r[0] for r in rows[:n_worst]]
    print(f"\n{'='*70}")
    print(f"DEEP-DIVE: preprocessing diagnostics for {n_worst} worst subjects")
    print(f"{'='*70}\n")

    for subj in worst_subjects:
        try:
            raw = load_subject(subj, IMAGINED_RUNS)
        except Exception as e:
            print(f"S{subj:03d}: FAILED TO LOAD — {e}\n")
            continue

        raw_clean, detected_bads = preprocess_raw(raw)
        epochs, n_dropped = epoch_subject(raw_clean)

        if epochs is None:
            print(f"S{subj:03d}: no T1/T2 annotations found\n")
            continue

        # amplitude range in the task window, same metric used by the rejection gate
        task_mask = (epochs.times >= 0.5) & (epochs.times <= 2.5)
        task_data = epochs.get_data(copy=True)[:, :, task_mask]
        amp_min, amp_max = task_data.min() * 1e6, task_data.max() * 1e6

        # quick ERD check: log power ratio at C3/C4, task window vs pre-stimulus baseline
        sfreq = epochs.info['sfreq']
        base_slice = slice(0, int(1.0 * sfreq))          # -1.0 to 0s
        task_slice = slice(int(1.5 * sfreq), int(3.5 * sfreq))  # 0.5 to 2.5s
        full_data = epochs.get_data(copy=True)

        erd_str = "n/a"
        if 'C3' in epochs.ch_names and 'C4' in epochs.ch_names:
            c3, c4 = epochs.ch_names.index('C3'), epochs.ch_names.index('C4')
            c3_base = np.var(full_data[:, c3, base_slice])
            c3_task = np.var(full_data[:, c3, task_slice])
            c4_base = np.var(full_data[:, c4, base_slice])
            c4_task = np.var(full_data[:, c4, task_slice])
            erd_c3 = np.log10(c3_task / c3_base) if c3_base > 0 else np.nan
            erd_c4 = np.log10(c4_task / c4_base) if c4_base > 0 else np.nan
            erd_str = f"C3={erd_c3:+.3f}  C4={erd_c4:+.3f}"

        print(f"S{subj:03d}:")
        print(f"  bad channels detected: {detected_bads}")
        print(f"  trials kept/dropped:   {len(epochs)} kept, {n_dropped} dropped by amplitude gate")
        print(f"  task-window amplitude: [{amp_min:.1f}, {amp_max:.1f}] μV")
        print(f"  ERD (log power ratio, negative=desync): {erd_str}")
        print()


if __name__ == "__main__":
    model_accs = load_all_per_subject_results()
    rows = stage_1_cross_model_table(model_accs)
    stage_2_investigate_worst(rows)