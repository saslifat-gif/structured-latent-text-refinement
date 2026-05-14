#!/usr/bin/env python3
"""Plot Stage 2 training curves from the text log.

The training log is intentionally verbose and changes over experiments, so this
parser is permissive: it extracts the fields that exist and reports how many
points were found before writing a PNG and CSV.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


TRAIN_RE = re.compile(r"epoch\s+(\d+)\s+step\s+(\d+)/(\d+)\s+\|\s+(.*)")
EPOCH_DONE_RE = re.compile(r"epoch\s+(\d+)\s+done\s+\|\s+avg train loss\s+([-+0-9.eE]+)")
VAL_LOSS_RE = re.compile(r"\s*val loss\s*:\s*([-+0-9.eE]+)")
VAL_SCORE_RE = re.compile(r"\s*val score\s*:\s*([-+0-9.eE]+)")
COSINE_RE = re.compile(r"\s*cosine sim\s*:\s*([-+0-9.eE]+)")
DECODER_CE_RE = re.compile(
    r"\s*decoder CE\s*:\s*real=([-+0-9.eE]+)\s+init=([-+0-9.eE]+)\s+raw=([-+0-9.eE]+)\s+gen=([-+0-9.eE]+)"
)
FUSED_CE_RE = re.compile(r"\s*fused CE\s*:\s*beta=([-+0-9.eE]+)\s+gen=([-+0-9.eE]+)")
METRIC_RE = re.compile(
    r"\s*metric diag\s*:\s*mean=([-+0-9.eE]+)\s+std=([-+0-9.eE]+)\s+min=([-+0-9.eE]+)\s+max=([-+0-9.eE]+)"
)


def _float_after(payload: str, key: str) -> float | None:
    match = re.search(rf"(?:^|\|\s*){re.escape(key)}\s+([-+0-9.eE]+)", payload)
    if match:
        return float(match.group(1))
    return None


def parse_log(path: Path) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    train_rows: list[dict[str, float]] = []
    val_rows: list[dict[str, float]] = []
    pending_val: dict[str, float] | None = None
    current_epoch = 0

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            train_match = TRAIN_RE.search(line)
            if train_match:
                epoch = int(train_match.group(1))
                step = int(train_match.group(2))
                total = max(int(train_match.group(3)), 1)
                payload = train_match.group(4)
                row: dict[str, float] = {
                    "epoch": float(epoch),
                    "step": float(step),
                    "x": epoch + step / total,
                }
                for key in ("rloss", "mloss", "eloss", "x0", "rollout", "rfce", "fce", "sce", "ot", "rnloss"):
                    value = _float_after(payload, key)
                    if value is not None and math.isfinite(value):
                        row[key] = value
                metric_match = re.search(r"\|\s*metric\s+([-+0-9.eE]+)\+/-([-+0-9.eE]+)", payload)
                if metric_match:
                    row["metric_mean"] = float(metric_match.group(1))
                    row["metric_std"] = float(metric_match.group(2))
                train_rows.append(row)
                continue

            done_match = EPOCH_DONE_RE.search(line)
            if done_match:
                current_epoch = int(done_match.group(1))
                continue

            if line.startswith("-- val metrics"):
                pending_val = {"epoch": float(current_epoch), "x": float(current_epoch)}
                continue

            if pending_val is None:
                continue

            if match := VAL_LOSS_RE.search(line):
                pending_val["val_loss"] = float(match.group(1))
            elif match := COSINE_RE.search(line):
                pending_val["cosine"] = float(match.group(1))
            elif match := DECODER_CE_RE.search(line):
                pending_val["real_ce"] = float(match.group(1))
                pending_val["init_ce"] = float(match.group(2))
                pending_val["raw_ce"] = float(match.group(3))
                pending_val["gen_ce"] = float(match.group(4))
            elif match := FUSED_CE_RE.search(line):
                pending_val["fused_beta"] = float(match.group(1))
                pending_val["fused_ce"] = float(match.group(2))
            elif match := METRIC_RE.search(line):
                pending_val["metric_mean"] = float(match.group(1))
                pending_val["metric_std"] = float(match.group(2))
                pending_val["metric_min"] = float(match.group(3))
                pending_val["metric_max"] = float(match.group(4))
            elif match := VAL_SCORE_RE.search(line):
                pending_val["val_score"] = float(match.group(1))
                val_rows.append(pending_val)
                pending_val = None

    return train_rows, val_rows


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    preferred = [
        "kind",
        "epoch",
        "step",
        "x",
        "rloss",
        "mloss",
        "eloss",
        "x0",
        "sce",
        "rfce",
        "fce",
        "rollout",
        "ot",
        "rnloss",
        "metric_mean",
        "metric_std",
        "metric_min",
        "metric_max",
        "val_loss",
        "val_score",
        "real_ce",
        "init_ce",
        "raw_ce",
        "gen_ce",
        "fused_ce",
        "fused_beta",
        "cosine",
    ]
    all_keys = {key for row in rows for key in row.keys()}
    keys = [key for key in preferred if key in all_keys]
    keys.extend(sorted(all_keys.difference(keys)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(train_rows: list[dict[str, float]], val_rows: list[dict[str, float]], out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is not installed, so only CSV files were written") from exc

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), constrained_layout=True)
    axes = axes.ravel()

    def plot_train(ax, keys: list[str], title: str) -> None:
        any_line = False
        for key in keys:
            points = [(row["x"], row[key]) for row in train_rows if key in row]
            if points:
                xs, ys = zip(*points)
                ax.plot(xs, ys, marker=".", linewidth=1.2, label=key)
                any_line = True
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
        if any_line:
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "no parsed points", ha="center", va="center", transform=ax.transAxes)

    def plot_val(ax, keys: list[str], title: str) -> None:
        any_line = False
        for key in keys:
            points = [(row["x"], row[key]) for row in val_rows if key in row]
            if points:
                xs, ys = zip(*points)
                ax.plot(xs, ys, marker="o", linewidth=1.5, label=key)
                any_line = True
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
        if any_line:
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "no parsed points", ha="center", va="center", transform=ax.transAxes)

    plot_train(axes[0], ["rloss", "mloss", "eloss", "x0"], "train losses")
    plot_train(axes[1], ["sce", "rfce", "fce", "rollout"], "token / rollout train losses")
    plot_train(axes[2], ["metric_std", "ot", "rnloss"], "metric / regularizers")
    plot_val(axes[3], ["val_score", "val_loss"], "validation score/loss")
    plot_val(axes[4], ["init_ce", "raw_ce", "gen_ce", "fused_ce", "real_ce"], "validation decoder CE")
    plot_val(axes[5], ["cosine", "metric_std", "metric_min", "metric_max"], "validation geometry")

    fig.suptitle(out_path.stem, fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="Stage 2 log file, e.g. logs_stage2_optionB_...txt")
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path")
    args = parser.parse_args()

    log_path = args.log
    out_path = args.out or log_path.with_suffix("").with_name(log_path.stem + "_curve.png")
    train_rows, val_rows = parse_log(log_path)

    parsed_csv = out_path.with_suffix(".parsed.csv")
    train_csv = out_path.with_suffix(".train.csv")
    val_csv = out_path.with_suffix(".val.csv")
    write_csv(train_csv, [{"kind": "train", **row} for row in train_rows])
    write_csv(val_csv, [{"kind": "val", **row} for row in val_rows])
    write_csv(parsed_csv, [{"kind": "train", **row} for row in train_rows] + [{"kind": "val", **row} for row in val_rows])

    plot_error = None
    try:
        plot_curves(train_rows, val_rows, out_path)
    except RuntimeError as exc:
        plot_error = str(exc)

    print(f"parsed train points: {len(train_rows)}")
    print(f"parsed val points: {len(val_rows)}")
    if plot_error:
        print(plot_error)
    else:
        print(f"wrote {out_path}")
    if train_rows or val_rows:
        print(f"wrote {parsed_csv}")
    if train_rows:
        print(f"wrote {train_csv}")
        print(f"last train row: {train_rows[-1]}")
    if val_rows:
        print(f"wrote {val_csv}")
        print(f"last val row: {val_rows[-1]}")
    if not train_rows and not val_rows:
        print("No points were found. Check that you passed the full tee log, not an empty/new file.")


if __name__ == "__main__":
    main()
