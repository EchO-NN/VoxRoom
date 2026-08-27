from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_history(path: str | Path) -> list[dict[str, object]]:
    history_path = Path(path)
    if not history_path.is_file():
        return []
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
        raise ValueError("training history must be a JSON list of objects")
    return payload


def rejected_seed_accuracy(row: dict[str, object]) -> float:
    if "rejected_seed_accuracy" in row:
        return float(row["rejected_seed_accuracy"])
    true_negative = int(row.get("tn", 0))
    false_negative = int(row.get("fn", 0))
    rejected_count = true_negative + false_negative
    return float(true_negative) / float(rejected_count) if rejected_count else float("nan")


class TrainingMonitor:
    def __init__(self, history_path: str | Path, *, refresh_seconds: float = 1.0) -> None:
        import matplotlib.pyplot as plt

        self.plt = plt
        self.history_path = Path(history_path).expanduser().resolve()
        self.summary_path = self.history_path.with_name("training_summary.json")
        self.refresh_seconds = max(0.2, float(refresh_seconds))
        self.last_epoch = -1
        self.finished = False
        self.fig, (self.loss_ax, self.metric_ax) = plt.subplots(2, 1, figsize=(14.0, 9.0), sharex=True)
        self.fig.patch.set_facecolor("white")
        self.manager = getattr(self.fig.canvas, "manager", None)
        if self.manager is not None and hasattr(self.manager, "set_window_title"):
            self.manager.set_window_title("VoxRoom DoorSeed Training Monitor")

        (self.train_loss_line,) = self.loss_ax.plot(
            [], [], color="#1565c0", linewidth=2.2, marker="o", markersize=5.0, label="Train loss"
        )
        (self.val_loss_line,) = self.loss_ax.plot(
            [], [], color="#d32f2f", linewidth=2.0, marker="o", markersize=5.0, label="Validation loss"
        )
        self.loss_ax.set_ylabel("Loss")
        self.loss_ax.set_yscale("log")
        self.loss_ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.8)
        self.loss_ax.legend(loc="upper right", frameon=False, ncol=2)

        metric_specs = (
            ("accuracy", "Accuracy", "#1565c0"),
            ("precision", "Precision", "#5e35b1"),
            ("recall", "Recall", "#00838f"),
            ("f1", "F1", "#c62828"),
            ("negative_rejection_rate", "Reject coverage", "#ef6c00"),
            ("rejected_seed_accuracy", "Rejected-seed accuracy", "#2e7d32"),
            ("pr_auc", "PR-AUC", "#6a1b9a"),
            ("roc_auc", "ROC-AUC", "#455a64"),
        )
        self.metric_lines = {}
        for key, label, color in metric_specs:
            (line,) = self.metric_ax.plot(
                [], [], color=color, linewidth=2.0, marker="o", markersize=5.0, label=label
            )
            self.metric_lines[key] = line
        self.metric_ax.set_xlabel("Epoch")
        self.metric_ax.set_ylabel("Validation metric")
        self.metric_ax.set_ylim(0.0, 1.01)
        self.metric_ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.8)
        self.metric_ax.legend(loc="lower right", frameon=False, ncol=2)
        self.status_text = self.fig.suptitle("Waiting for the first completed epoch", fontsize=12, fontweight="bold")
        self.fig.tight_layout(rect=(0.04, 0.04, 0.98, 0.89))

    def refresh(self) -> bool:
        try:
            history = load_history(self.history_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self.status_text.set_text("Waiting for valid training history: %s" % exc)
            self.fig.canvas.draw_idle()
            return True
        if not history:
            return True
        latest = history[-1]
        threshold_mode = str(latest.get("threshold_selection_mode", "target_recall"))
        fixed_mode = threshold_mode == "fixed"
        threshold = float(latest.get("threshold", 0.5))
        latest_epoch = int(latest.get("epoch", len(history)))
        if self.finished and latest_epoch == self.last_epoch:
            return False
        if latest_epoch == self.last_epoch and not self.summary_path.is_file():
            return True
        self.last_epoch = latest_epoch
        epochs = [int(row.get("epoch", index + 1)) for index, row in enumerate(history)]
        self.train_loss_line.set_data(epochs, [float(row["train_loss"]) for row in history])
        self.val_loss_line.set_data(epochs, [float(row["val_loss"]) for row in history])
        for key, line in self.metric_lines.items():
            threshold_dependent = key not in {"pr_auc", "roc_auc"}
            line.set_visible(fixed_mode or not threshold_dependent)
            values = (
                [rejected_seed_accuracy(row) for row in history]
                if key == "rejected_seed_accuracy"
                else [float(row[key]) for row in history]
            )
            line.set_data(epochs, values)
            base_label = {
                "accuracy": "Accuracy",
                "precision": "Precision",
                "recall": "Recall",
                "f1": "F1",
                "negative_rejection_rate": "Reject coverage",
                "rejected_seed_accuracy": "Rejected-seed accuracy",
                "pr_auc": "PR-AUC",
                "roc_auc": "ROC-AUC",
            }[key]
            line.set_label(
                "%s @ %.2f" % (base_label, threshold)
                if fixed_mode and threshold_dependent
                else base_label
            )

        self.loss_ax.relim()
        self.loss_ax.autoscale_view(scalex=False, scaley=True)
        self.metric_ax.set_xlim(1, max(2, latest_epoch))
        visible_lines = [line for line in self.metric_lines.values() if line.get_visible()]
        self.metric_ax.legend(handles=visible_lines, loc="lower right", frameon=False, ncol=2)
        state = "RUNNING"
        if self.summary_path.is_file():
            try:
                summary = json.loads(self.summary_path.read_text(encoding="utf-8"))
                state = "STOPPED: %s" % summary.get("stop_reason", "completed")
                self.finished = True
            except (OSError, ValueError, json.JSONDecodeError):
                state = "STOPPED"
        common = (
            "%s | Epoch %d | no improvement %d/%d | %.1fs/epoch\n"
            % (
                state,
                latest_epoch,
                int(latest.get("early_stopping_epochs_without_improvement", 0)),
                int(latest.get("early_stopping_patience", 10)),
                float(latest.get("epoch_seconds", 0.0)),
            )
        )
        if fixed_mode:
            self.status_text.set_text(
                "%sFIXED THRESHOLD = %.2f | train %.6f | val %.6f | accuracy %.2f%% | F1 %.2f%%\n"
                "reject coverage %.2f%% | rejected-seed accuracy %.2f%%"
                % (
                    common,
                    threshold,
                    float(latest["train_loss"]),
                    float(latest["val_loss"]),
                    100.0 * float(latest["accuracy"]),
                    100.0 * float(latest["f1"]),
                    100.0 * float(latest["negative_rejection_rate"]),
                    100.0 * rejected_seed_accuracy(latest),
                )
            )
            if self.manager is not None and hasattr(self.manager, "set_window_title"):
                self.manager.set_window_title("VoxRoom Training — Fixed threshold 0.5")
        else:
            self.status_text.set_text(
                "%sLEGACY TARGET-RECALL RUN — not for deployment | train %.6f | val %.6f | PR-AUC %.3f"
                % (
                    common,
                    float(latest["train_loss"]),
                    float(latest["val_loss"]),
                    float(latest["pr_auc"]),
                )
            )
            if self.manager is not None and hasattr(self.manager, "set_window_title"):
                self.manager.set_window_title("VoxRoom Training — Legacy target-recall run")
        self.fig.canvas.draw_idle()
        return not self.finished

    def show(self) -> None:
        self.refresh()
        self.plt.show(block=False)
        while self.plt.fignum_exists(self.fig.number):
            self.refresh()
            self.plt.pause(self.refresh_seconds)

    def save(self, path: str | Path) -> None:
        self.refresh()
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(target, dpi=160, facecolor="white")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live monitor for VoxRoom DoorSeed classifier training.")
    parser.add_argument("--history", required=True, help="Path to training_history.json")
    parser.add_argument("--refresh-seconds", type=float, default=1.0)
    parser.add_argument("--once", action="store_true", help="Render once and exit instead of opening a live window.")
    parser.add_argument("--save-path", default=None, help="PNG path used with --once.")
    args = parser.parse_args(argv)
    monitor = TrainingMonitor(args.history, refresh_seconds=float(args.refresh_seconds))
    if args.once:
        if not args.save_path:
            parser.error("--once requires --save-path")
        monitor.save(args.save_path)
        return 0
    monitor.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
