#!/usr/bin/env python
import argparse
import os
import shutil
from pathlib import Path
from typing import List, Tuple


def discover_moves(run_dir: Path) -> List[Tuple[Path, Path]]:
    moves: List[Tuple[Path, Path]] = []

    for fold_dir in sorted(run_dir.glob("fold*")):
        if not fold_dir.is_dir():
            continue
        fold_name = fold_dir.name
        if not fold_name.startswith("fold"):
            continue

        for ablation_dir in sorted(fold_dir.iterdir()):
            if not ablation_dir.is_dir():
                continue
            ablation_name = ablation_dir.name

            nested_fold_dir = ablation_dir / fold_name
            if not nested_fold_dir.is_dir():
                continue

            src = nested_fold_dir
            dst = run_dir / ablation_name / fold_name
            moves.append((src, dst))

    return moves


def discover_merge_items(ablation_dir: Path, fold_name: str) -> List[Path]:
    nested_fold_dir = ablation_dir / fold_name
    if not nested_fold_dir.is_dir():
        return []

    items: List[Path] = []
    for item in sorted(ablation_dir.iterdir()):
        if item.name == fold_name:
            continue
        items.append(item)
    return items


def merge_items_into_fold_dir(*, ablation_dir: Path, fold_name: str) -> None:
    nested_fold_dir = ablation_dir / fold_name
    if not nested_fold_dir.is_dir():
        return

    for item in discover_merge_items(ablation_dir, fold_name):
        dest = nested_fold_dir / item.name
        if dest.exists():
            raise FileExistsError(
                f"Refusing to overwrite during merge: {dest} (src={item})"
            )
        shutil.move(str(item), str(dest))


def rmdir_if_empty(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    try:
        next(path.iterdir())
        return False
    except StopIteration:
        path.rmdir()
        return True


def cleanup_empty_scaffold(run_dir: Path) -> None:
    for fold_dir in sorted(run_dir.glob("fold*")):
        if not fold_dir.is_dir():
            continue
        for ablation_dir in sorted(fold_dir.iterdir()):
            if not ablation_dir.is_dir():
                continue
            rmdir_if_empty(ablation_dir)
        rmdir_if_empty(fold_dir)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Reorganize ablation run directories from '<run>/foldK/<abl>/foldK' "
            "to '<run>/<abl>/foldK'."
        )
    )
    ap.add_argument(
        "--run-dir",
        required=True,
        help="Path to the ablation run directory (contains fold1..foldN).",
    )
    ap.add_argument(
        "--execute",
        action="store_true",
        help="Perform moves. If omitted, runs in dry-run mode.",
    )
    ap.add_argument(
        "--cleanup-empty",
        action="store_true",
        help="After moving, remove empty scaffold directories under fold*/.",
    )
    args = ap.parse_args()

    run_dir = Path(os.path.abspath(args.run_dir))
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"run dir not found: {run_dir}")

    moves = discover_moves(run_dir)
    if not moves:
        print(f"No moves discovered under: {run_dir}")
        return

    print(f"Discovered {len(moves)} planned moves under: {run_dir}")

    for src, dst in moves:
        ablation_dir = src.parent
        fold_name = src.name
        merge_items = discover_merge_items(ablation_dir, fold_name)
        for item in merge_items:
            print(f"MERGE: {item} -> {src / item.name}")
        print(f"MOVE: {src} -> {dst}")

    if not args.execute:
        print("\nDry-run only. Re-run with --execute to perform these moves.")
        return

    for src, dst in moves:
        ablation_dir = src.parent
        fold_name = src.name
        merge_items_into_fold_dir(ablation_dir=ablation_dir, fold_name=fold_name)

        dst_parent = dst.parent
        dst_parent.mkdir(parents=True, exist_ok=True)

        if dst.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing destination: {dst} (src={src})"
            )

        shutil.move(str(src), str(dst))

    if args.cleanup_empty:
        cleanup_empty_scaffold(run_dir)

    print("Done")


if __name__ == "__main__":
    main()
