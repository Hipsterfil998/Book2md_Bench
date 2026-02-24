"""
Evaluate predicted Markdown against ground-truth (NED + BLEU).

Usage
-----
Single pair:
    python eval.py <ground_truth.md> <prediction.md>

Batch (matching filenames in two directories):
    python eval.py --ref-dir dataset/italian/book_123/pages \\
                   --pred-dir predictions/italian/book_123
"""

import argparse
import sys
from pathlib import Path

from metrics.ned import NED
from metrics.bleu import BLEU


def eval_pair(ref_path: Path, pred_path: Path) -> dict:
    ref  = ref_path.read_text(encoding="utf-8")
    pred = pred_path.read_text(encoding="utf-8")
    ned  = NED().score(ref, pred)
    return {
        "ned":        ned,
        "similarity": round(1.0 - ned, 6),
        "bleu":       BLEU().score(ref, pred),
    }


def eval_dirs(ref_dir: Path, pred_dir: Path) -> None:
    ref_files = sorted(ref_dir.glob("*.md"))
    if not ref_files:
        print(f"No .md files found in {ref_dir}", file=sys.stderr)
        sys.exit(1)

    scores_list = []
    for ref_path in ref_files:
        pred_path = pred_dir / ref_path.name
        if not pred_path.exists():
            print(f"  [SKIP] {ref_path.name} — prediction not found")
            continue
        scores = eval_pair(ref_path, pred_path)
        scores_list.append(scores)
        print(f"  {ref_path.name}  "
              f"NED={scores['ned']:.4f}  "
              f"similarity={scores['similarity']:.4f}  "
              f"BLEU={scores['bleu']:.2f}")

    if scores_list:
        n        = len(scores_list)
        avg_ned  = round(sum(s["ned"]        for s in scores_list) / n, 4)
        avg_sim  = round(sum(s["similarity"] for s in scores_list) / n, 4)
        avg_bleu = round(sum(s["bleu"]       for s in scores_list) / n, 2)
        print(f"\n{'─'*50}")
        print(f"  Files evaluated : {n}")
        print(f"  Avg NED         : {avg_ned:.4f}")
        print(f"  Avg similarity  : {avg_sim:.4f}")
        print(f"  Avg BLEU        : {avg_bleu:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate predicted Markdown against ground-truth (NED + BLEU)."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("files", nargs="*", metavar="FILE",
                       help="Two files: <ground_truth.md> <prediction.md>")
    group.add_argument("--ref-dir",  type=Path, metavar="DIR")
    parser.add_argument("--pred-dir", type=Path, metavar="DIR")
    args = parser.parse_args()

    if args.ref_dir:
        if not args.pred_dir:
            parser.error("--pred-dir is required with --ref-dir")
        eval_dirs(args.ref_dir, args.pred_dir)
    else:
        if len(args.files) != 2:
            parser.error("Provide exactly two files: <ground_truth.md> <prediction.md>")
        ref_path, pred_path = Path(args.files[0]), Path(args.files[1])
        scores = eval_pair(ref_path, pred_path)
        print(f"NED        : {scores['ned']:.4f}")
        print(f"Similarity : {scores['similarity']:.4f}")
        print(f"BLEU       : {scores['bleu']:.2f}")


if __name__ == "__main__":
    main()
