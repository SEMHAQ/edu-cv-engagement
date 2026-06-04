"""Simple training logger."""

import json
import os
import time
from datetime import datetime


class ExperimentLogger:
    """Logs training metrics to JSON file and console."""

    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        os.makedirs(log_dir, exist_ok=True)

        self.log_file = os.path.join(log_dir, f"{experiment_name}.json")
        self.history = {
            "experiment": experiment_name,
            "start_time": datetime.now().isoformat(),
            "epochs": [],
            "config": {},
        }

    def log_config(self, config: dict):
        self.history["config"] = config

    def log_epoch(self, epoch: int, metrics: dict, phase: str = "train"):
        entry = {
            "epoch": epoch,
            "phase": phase,
            "time": datetime.now().isoformat(),
            **metrics,
        }
        self.history["epochs"].append(entry)
        self._save()

    def log_result(self, result: dict):
        self.history["final_result"] = result
        self._save()

    def _save(self):
        with open(self.log_file, "w") as f:
            json.dump(self.history, f, indent=2, default=str)

    def get_best_epoch(self, metric: str = "val_accuracy", mode: str = "max"):
        """Get the epoch with the best validation metric."""
        val_epochs = [e for e in self.history["epochs"] if e.get("phase") == "val"]
        if not val_epochs:
            return None
        if mode == "max":
            return max(val_epochs, key=lambda e: e.get(metric, 0))
        return min(val_epochs, key=lambda e: e.get(metric, float("inf")))
