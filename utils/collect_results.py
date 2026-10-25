"""Collect the last logged validation metrics for the main student experiments."""

import argparse
import os
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["ade20k", "coco"], required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--times", type=int, default=int(os.environ.get("TIMES", 5)))
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--work-root", type=Path, default=Path(os.environ.get(
        "WORK_ROOT", str(Path(os.environ.get("MMSEG_ROOT", "../third-party/mmsegmentation-main")) / "work_dirs_distillation")
    )))
    args = parser.parse_args()
    if args.repeats is None:
        args.repeats = int(os.environ.get("REPEATS", 5 if args.dataset == "ade20k" else 3))
    if args.times < 1 or args.repeats < 1:
        parser.error("times/repeats must be positive")
    return args


def last_metrics(log_file):
    for line in reversed(log_file.read_text().splitlines()):
        miou = re.search(r"\bmIoU:\s*([0-9.]+)", line)
        macc = re.search(r"\bmAcc:\s*([0-9.]+)", line)
        if miou and macc:
            return float(miou.group(1)), float(macc.group(1))
    raise ValueError(f"No validation metrics in {log_file}")


def collect(args):
    ratios = ("0.0025", "0.005", "0.01") if args.dataset == "ade20k" else ("0.000625", "0.00125", "0.0025")
    root = args.work_root / args.dataset / f"{args.dataset_name}_segformer"
    for ratio in ratios:
        miou_list, macc_list = [], []
        for time in range(1, args.times + 1):
            result_path = root / f"time_{time}_ratio_{ratio}"
            current = []
            for repeat in range(1, args.repeats + 1):
                logs = sorted((result_path / f"repeat_{repeat}").glob("**/*.log"))
                if len(logs) != 1:
                    print(f"  Missing or ambiguous logs: {result_path}/repeat_{repeat} ({len(logs)} files)")
                    break
                try:
                    current.append(last_metrics(logs[0]))
                except ValueError as error:
                    print(f"  {error}")
                    break
            if len(current) != args.repeats:
                continue
            miou_list.extend(value[0] for value in current)
            macc_list.extend(value[1] for value in current)
            print(f"  time {time}: mIoU {sum(v[0] for v in current) / len(current):.2f}, mAcc {sum(v[1] for v in current) / len(current):.2f}")
        if not miou_list:
            print(f"ratio {ratio}: No results found.")
            continue
        mean_miou = sum(miou_list) / len(miou_list)
        mean_macc = sum(macc_list) / len(macc_list)
        std_miou = (sum((value - mean_miou) ** 2 for value in miou_list) / len(miou_list)) ** 0.5
        std_macc = (sum((value - mean_macc) ** 2 for value in macc_list) / len(macc_list)) ** 0.5
        print(f"ratio {ratio}: mIoU {mean_miou:.2f}±{std_miou:.2f}, mAcc {mean_macc:.2f}±{std_macc:.2f}")


if __name__ == "__main__":
    collect(parse_args())
