"""
BLEU score — word n-gram precision via sacrebleu.

Range [0, 100]: 0 = no overlap, 100 = perfect match.
"""

import re
import unicodedata

from sacrebleu.metrics import BLEU as _SacreBLEU


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class BLEU:
    """Sentence-level BLEU score (sacrebleu, 13a tokeniser).

    Range [0, 100] — higher is better.
    """

    higher_is_better = True

    def score(self, reference: str, hypothesis: str) -> float:
        ref = _normalise(reference)
        hyp = _normalise(hypothesis)
        return _SacreBLEU(tokenize="13a", effective_order=True) \
                   .sentence_score(hyp, [ref]).score

    def corpus_score(self, pairs: list[tuple[str, str]]) -> float:
        if not pairs:
            return 0.0
        return round(sum(self.score(r, h) for r, h in pairs) / len(pairs), 2)
