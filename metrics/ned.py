"""
Normalised Edit Distance (NED) — character-level similarity metric.

Range [0, 1]: 0 = identical, 1 = completely different.
"""

import re
import unicodedata

from rapidfuzz.distance import Levenshtein


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class NED:
    """Normalised Edit Distance at character level.

    NED = Levenshtein(ref, hyp) / max(len(ref), len(hyp))
    Range [0, 1] — lower is better.
    """

    higher_is_better = False

    def score(self, reference: str, hypothesis: str) -> float:
        ref = _normalise(reference)
        hyp = _normalise(hypothesis)
        if not ref and not hyp:
            return 0.0
        if not ref or not hyp:
            return 1.0
        dist = Levenshtein.distance(ref, hyp)
        return round(dist / max(len(ref), len(hyp)), 6)

    def corpus_score(self, pairs: list[tuple[str, str]]) -> float:
        if not pairs:
            return 0.0
        return round(sum(self.score(r, h) for r, h in pairs) / len(pairs), 4)
