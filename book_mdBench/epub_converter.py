"""
EPUB conversion utilities.
"""

from pathlib import Path

import pypandoc


class EpubConverter:
    """Convert EPUB files to Markdown."""

    def to_markdown(self, epub_path: Path, md_path: Path) -> bool:
        """Convert EPUB → Markdown via pandoc. Returns True on success."""
        if md_path.exists():
            return True
        try:
            pypandoc.convert_file(
                str(epub_path),
                "markdown",
                outputfile=str(md_path),
                extra_args=["--wrap=none"],
            )
            return True
        except Exception as e:
            print(f"    ✗ EPUB→MD failed: {e}")
            return False
