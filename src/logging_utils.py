"""Appends one row per completed evaluation to experiments_log.csv."""
import csv
import os
from datetime import datetime
import numpy as np


def log_experiment(model_name, accs, n_subjects, evaluation_method, notes=""):
    log_path = "experiments_log.csv"
    file_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "model", "mean_acc", "std_acc", "min_acc",
                              "max_acc", "n_subjects", "evaluation_method", "notes"])
        writer.writerow([
            datetime.now().isoformat(timespec='seconds'), model_name,
            f"{np.mean(accs)*100:.2f}", f"{np.std(accs)*100:.2f}",
            f"{min(accs)*100:.2f}", f"{max(accs)*100:.2f}",
            n_subjects, evaluation_method, notes,
        ])