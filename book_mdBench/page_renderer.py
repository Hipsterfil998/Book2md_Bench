"""
Chunk renderer — converts markdown chunks to JPEG images via PDF.
"""

import subprocess
import tempfile
from pathlib import Path

from pdf2image import convert_from_path
from pdf2image.exceptions import PDFPageCountError

from book_mdBench.config import IMAGE_DPI


class PageRenderer:
    """Render markdown chunks to JPEG images (md → pdf → jpg)."""

    def render(self, chunks: list[str], indices: list[int], out_dir: Path) -> dict[int, Path]:
        """
        For each index in *indices*, render chunks[index] to a JPEG in *out_dir*.

        Returns a dict mapping chunk index → Path of the saved image.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        rendered = {}

        for idx in indices:
            img_path = out_dir / f"page_{idx:04d}.jpg"
            if img_path.exists():
                rendered[idx] = img_path
                continue

            image = self._chunk_to_image(chunks[idx])
            if image is None:
                continue

            image.save(str(img_path), "JPEG")
            rendered[idx] = img_path

        return rendered

    def _chunk_to_image(self, text: str):
        """Convert a markdown string to a PIL image (first PDF page)."""
        with tempfile.TemporaryDirectory() as tmp:
            md_path  = Path(tmp) / "chunk.md"
            pdf_path = Path(tmp) / "chunk.pdf"

            md_path.write_text(text, encoding="utf-8")

            result = subprocess.run(
                ["pandoc", str(md_path), "-o", str(pdf_path), "--pdf-engine=weasyprint"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                print(f"      ✗ MD→PDF failed: {result.stderr[:150]}")
                return None

            try:
                pages = convert_from_path(str(pdf_path), dpi=IMAGE_DPI)
                return pages[0] if pages else None
            except PDFPageCountError as e:
                print(f"      ✗ PDF→IMG failed: {e}")
                return None
