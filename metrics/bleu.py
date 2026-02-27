"""
BLEU score — word n-gram precision via sacrebleu.

Range [0, 100]: 0 = no overlap, 100 = perfect match.
"""

from sacrebleu.metrics import BLEU as _SacreBLEU

from ._utils import normalise


class BLEU:
    """Sentence-level BLEU score (sacrebleu, 13a tokeniser).

    Range [0, 100] — higher is better.
    """

    higher_is_better = True

    def score(self, reference: str, hypothesis: str) -> float:
        ref = normalise(reference)
        hyp = normalise(hypothesis)
        return _SacreBLEU(tokenize="13a", effective_order=True) \
                   .sentence_score(hyp, [ref]).score

    def corpus_score(self, pairs: list[tuple[str, str]]) -> float:
        if not pairs:
            return 0.0
        return round(sum(self.score(r, h) for r, h in pairs) / len(pairs), 2)
