"""
Markdown page splitting and stratified sampling.
"""

import random
import re
from pathlib import Path
from book_mdBench.config import MIN_MD_CHARS, STRATA


class PageSampler:
    """Split a markdown file into chunks and sample pages stratified by zone."""

    # target chunk size when falling back to paragraph-based splitting
    _CHUNK_CHARS = 1500

    def split(self, md_path: Path) -> list[str]:
        """Split markdown into chunks. Uses headings first, falls back to paragraphs."""
        text = md_path.read_text(encoding="utf-8")

        # primary: split on level 1/2 headings
        parts = re.split(r"(?=^#{1,2} )", text, flags=re.MULTILINE)
        chunks = [p for p in parts if len(p.strip()) >= MIN_MD_CHARS]

        # fallback: no headings found — split by paragraph groups
        if len(chunks) < 10:
            chunks = self._split_by_paragraphs(text)

        return chunks

    def _split_by_paragraphs(self, text: str) -> list[str]:
        """Group consecutive paragraphs into chunks of ~_CHUNK_CHARS characters."""
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if len(p.strip()) >= MIN_MD_CHARS]
        chunks, current = [], []
        current_len = 0
        for para in paragraphs:
            current.append(para)
            current_len += len(para)
            if current_len >= self._CHUNK_CHARS:
                chunks.append("\n\n".join(current))
                current, current_len = [], 0
        if current:
            chunks.append("\n\n".join(current))
        return chunks

    def sample(self, n_chunks: int) -> dict[str, list[int]]:
        """
        Sample page indices stratified across front / body / back zones.

        Returns a dict mapping zone name → list of sampled indices.
        """
        front_end = max(1, n_chunks // 5)
        back_start = n_chunks - max(1, n_chunks // 5)

        zones = {
            "front": list(range(0, front_end)),
            "body":  list(range(front_end, back_start)),
            "back":  list(range(back_start, n_chunks)),
        }

        sampled = {}
        for zone, k in STRATA.items():
            pool = zones[zone]
            sampled[zone] = random.sample(pool, min(k, len(pool)))

        return sampled
