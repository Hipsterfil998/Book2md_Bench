"""Shared text normalisation for evaluation metrics."""

import re
import unicodedata


def normalise(text: str) -> str:
    """Normalise whitespace and Unicode for consistent metric computation."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
