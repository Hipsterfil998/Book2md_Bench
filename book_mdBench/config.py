"""
Benchmark dataset configuration.
"""

from pathlib import Path


# ── dataset parameters ────────────────────────────────────────────────────────

LANGUAGES: dict[str, str] = {
    "italian": "it",
    "german":  "de",
}

N_BOOKS      = 15
N_PAGES      = 20
STRATA       = {"front": 5, "body": 10, "back": 5}
MIN_MD_CHARS = 150   # min chars in a markdown chunk to be considered non-blank
IMAGE_DPI    = 150   # DPI for page image rendering
OUTPUT_DIR   = Path("./benchmark_data")


# ── external URLs ─────────────────────────────────────────────────────────────

GUTENDEX_URL = "https://gutendex.com/books/"