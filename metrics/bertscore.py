"""
BERTScore — semantic similarity via contextual embeddings.

Uses xlm-roberta-base by default for multilingual support (Italian, German, …).
The model is downloaded automatically on first use (~1 GB) and cached locally.

Range [0, 1] — higher is better.
"""

from bert_score import score as _bert_score

from ._utils import normalise


class BERTScore:
    """BERTScore F1 using xlm-roberta-base (multilingual).

    Range [0, 1] — higher is better.

    Note: the first call downloads the model weights (~1 GB) and may take
    a few minutes.  Subsequent calls use the cached model.
    """

    higher_is_better = True
    _model            = "xlm-roberta-base"

    def score(self, reference: str, hypothesis: str) -> float:
        ref = normalise(reference)
        hyp = normalise(hypothesis)
        if not ref and not hyp:
            return 1.0
        if not ref or not hyp:
            return 0.0
        _, _, f1 = _bert_score(
            [hyp], [ref],
            model_type=self._model,
            verbose=False,
        )
        return round(f1[0].item(), 6)

    def corpus_score(self, pairs: list[tuple[str, str]]) -> float:
        """Batch evaluation — more efficient than calling score() in a loop."""
        if not pairs:
            return 0.0
        refs = [normalise(r) for r, _ in pairs]
        hyps = [normalise(h) for _, h in pairs]
        _, _, f1 = _bert_score(
            hyps, refs,
            model_type=self._model,
            verbose=False,
        )
        return round(float(f1.mean().item()), 4)
