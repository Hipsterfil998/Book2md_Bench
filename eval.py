"""
Evaluate predicted Markdown against ground-truth.

Metrics
-------
  NED          Normalised Edit Distance        [0, 1]   lower  is better
  BLEU         Word n-gram precision           [0, 100] higher is better
  Structure F1 Markdown structural elements    [0, 1]   higher is better
  BERTScore    Semantic similarity (optional)  [0, 1]   higher is better

Usage
-----
Single pair:
    python eval.py <ground_truth.md> <prediction.md>

Batch (matching filenames in two directories):
    python eval.py --ref-dir dataset/italian/book_123/pages \\
                   --pred-dir predictions/italian/book_123

Add --bert to also compute BERTScore (downloads ~1 GB model on first run).
"""

import argparse
import sys
from pathlib import Path

from metrics import NED, BLEU, MarkdownStructureF1


def eval_pair(ref_path: Path, pred_path: Path, *, bert: bool = False) -> dict:
    ref  = ref_path.read_text(encoding="utf-8")
    pred = pred_path.read_text(encoding="utf-8")
    ned  = NED().score(ref, pred)
    scores = {
        "ned":        ned,
        "similarity": round(1.0 - ned, 6),
        "bleu":       BLEU().score(ref, pred),
        "struct_f1":  MarkdownStructureF1().score(ref, pred),
    }
    if bert:
        from metrics import BERTScore
        scores["bert_f1"] = BERTScore().score(ref, pred)
    return scores


def eval_dirs(ref_dir: Path, pred_dir: Path, *, bert: bool = False) -> None:
    ref_files = sorted(ref_dir.glob("*.md"))
    if not ref_files:
        print(f"No .md files found in {ref_dir}", file=sys.stderr)
        sys.exit(1)

    ned_scorer    = NED()
    bleu_scorer   = BLEU()
    struct_scorer = MarkdownStructureF1()

    scores_list = []
    bert_pairs  = []

    for ref_path in ref_files:
        pred_path = pred_dir / ref_path.name
        if not pred_path.exists():
            print(f"  [SKIP] {ref_path.name} — prediction not found")
            continue

        ref  = ref_path.read_text(encoding="utf-8")
        pred = pred_path.read_text(encoding="utf-8")

        ned = ned_scorer.score(ref, pred)
        scores = {
            "ned":        ned,
            "similarity": round(1.0 - ned, 6),
            "bleu":       bleu_scorer.score(ref, pred),
            "struct_f1":  struct_scorer.score(ref, pred),
        }
        scores_list.append(scores)
        bert_pairs.append((ref, pred))

        print(f"  {ref_path.name}  "
              f"NED={scores['ned']:.4f}  "
              f"sim={scores['similarity']:.4f}  "
              f"BLEU={scores['bleu']:.2f}  "
              f"struct_F1={scores['struct_f1']:.4f}")

    if not scores_list:
        return

    n = len(scores_list)
    print(f"\n{'─'*50}")
    print(f"  Files evaluated : {n}")
    print(f"  Avg NED         : {sum(s['ned']        for s in scores_list) / n:.4f}")
    print(f"  Avg similarity  : {sum(s['similarity'] for s in scores_list) / n:.4f}")
    print(f"  Avg BLEU        : {sum(s['bleu']       for s in scores_list) / n:.2f}")
    print(f"  Avg struct F1   : {sum(s['struct_f1']  for s in scores_list) / n:.4f}")

    if bert:
        from metrics import BERTScore
        print(f"  Avg BERTScore   : {BERTScore().corpus_score(bert_pairs):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate predicted Markdown against ground-truth."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("files", nargs="*", metavar="FILE",
                       help="Two files: <ground_truth.md> <prediction.md>")
    group.add_argument("--ref-dir",  type=Path, metavar="DIR")
    parser.add_argument("--pred-dir", type=Path, metavar="DIR")
    parser.add_argument("--bert", action="store_true",
                        help="Also compute BERTScore (slow, requires ~1 GB model)")
    args = parser.parse_args()

    if args.ref_dir:
        if not args.pred_dir:
            parser.error("--pred-dir is required with --ref-dir")
        eval_dirs(args.ref_dir, args.pred_dir, bert=args.bert)
    else:
        if len(args.files) != 2:
            parser.error("Provide exactly two files: <ground_truth.md> <prediction.md>")
        scores = eval_pair(Path(args.files[0]), Path(args.files[1]), bert=args.bert)
        print(f"NED        : {scores['ned']:.4f}")
        print(f"Similarity : {scores['similarity']:.4f}")
        print(f"BLEU       : {scores['bleu']:.2f}")
        print(f"Struct F1  : {scores['struct_f1']:.4f}")
        if args.bert:
            print(f"BERTScore  : {scores['bert_f1']:.4f}")


if __name__ == "__main__":
    main()
