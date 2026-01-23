import argparse
import os
import subprocess

try:
    from baselines.common import utils
except ModuleNotFoundError:  # pragma: no cover
    import sys

    _REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    from baselines.common import utils


def _run(cmd: list[str]) -> None:
    subprocess.check_call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--label_csv", default=None)
    ap.add_argument(
        "--output_dir", default=os.path.join(os.path.dirname(__file__), "runs")
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--n_splits", type=int, dest="n_folds", help=argparse.SUPPRESS)
    ap.add_argument(
        "--fold",
        type=int,
        default=None,
        help="If set, run only this 1-based fold index.",
    )
    ap.add_argument(
        "--task",
        choices=["binary_stage1", "binary_stage2", "multiclass"],
        default=None,
        help="Override task for baselines that support it (mfcc_svm, yamnet_lr, se_resnet).",
    )
    ap.add_argument(
        "--spectrogram_root",
        default=None,
        help="Spectrogram root for se_resnet baseline (folder containing class/patient/*.npy).",
    )
    ap.add_argument("--run_name", default=None)
    args = ap.parse_args()

    parent_run = utils.resolve_run_dir(args.output_dir, args.run_name)

    baselines = [
        ("mfcc_svm", "baselines/mfcc_svm/train.py"),
        ("resnet_mel", "baselines/resnet_mel/train.py"),
        ("yamnet_lr", "baselines/yamnet_lr/train.py"),
        ("se_resnet", "baselines/se_resnet/train.py"),
    ]

    for name, script in baselines:
        cmd = [
            "python",
            script,
            "--output_dir",
            parent_run,
            "--seed",
            str(args.seed),
            "--n_folds",
            str(args.n_folds),
            "--run_name",
            name,
        ]
        if args.fold is not None:
            cmd.extend(["--fold", str(args.fold)])
        if bool(args.dry_run):
            cmd.append("--dry_run")
        if args.data_root is not None:
            cmd.extend(["--data_root", args.data_root])
        if args.label_csv is not None:
            cmd.extend(["--label_csv", args.label_csv])
        if args.task is not None and name in ("mfcc_svm", "yamnet_lr"):
            if args.task in ("binary_stage2", "multiclass"):
                cmd.extend(["--task", args.task])
        if args.task is not None and name == "se_resnet":
            cmd.extend(["--task", args.task])
        if args.spectrogram_root is not None and name == "se_resnet":
            cmd.extend(["--spectrogram_root", args.spectrogram_root])

        _run(cmd)


if __name__ == "__main__":
    main()
