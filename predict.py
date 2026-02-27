"""
predict.py — Generate Markdown predictions from ground-truth page images.

Runs a vision-language model (via vLLM) on every page_*.jpg in a dataset
pages directory and writes page_*.md predictions ready for evaluation with
eval.py.

Example
-------
    from predict import PageImagePredictor
    from pathlib import Path

    PRED_DIR = Path("predictions")

    for MODEL in [
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "mistralai/Pixtral-12B-2409",
    ]:
        p = PageImagePredictor(model_id=MODEL)
        p.predict_dataset(Path("dataset"), PRED_DIR / p.model_slug)
"""

import base64
import logging
from io import BytesIO
from pathlib import Path

from PIL import Image
from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt — mirrors ground-truth Markdown conventions from epub_converter.py
# ---------------------------------------------------------------------------

_PROMPT = """\
Convert the book page shown in this image to clean Markdown. \
Follow these rules exactly and output ONLY the Markdown — no explanations, \
no commentary, no code fences.

HEADINGS
  Reproduce the heading hierarchy with ATX markers:
    # for top-level title
    ## for chapter / part heading
    ### for section heading
    #### for sub-section heading
    ##### and ###### for lower levels

TABLES
  Render every table as a Markdown pipe table:
    | Col A | Col B |
    |-------|-------|
    | cell  | cell  |

MATH
  Inline formula   →  $formula$
  Block / display  →  $$formula$$

LISTS
  Preserve ordered and unordered lists with their original indentation.

IMAGES / FIGURES
  Replace each figure or illustration with:
    ![image_N](images/image_N.png)
  where N starts at 1 and increments for each image on this page.

BOOK PAGE NUMBERS
  If a page number is printed in the text, header, or margin, write:
    [p. N]

FOOTNOTES
  In-text marker     →  [^N]
  Definition at end  →  [^N]: text of the footnote

INLINE FORMATTING
  Bold           →  **text**
  Italic         →  *text*
  Strikethrough  →  ~~text~~
  Superscript (non-numeric only)  →  ^text^
  Subscript                       →  ~text~
  Numeric superscripts (footnote references) must use [^N], never ^N^.

GENERAL
  Do not emit any HTML tags.
  Do not add YAML front matter.
  Collapse three or more consecutive blank lines to two.\
"""


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _pil_to_data_url(img) -> str:
    """Encode a PIL image as a base64 PNG data URL for vLLM multimodal input."""
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class PageImagePredictor:
    """Convert ground-truth page JPEG images to Markdown via a VLM on vLLM.

    Parameters
    ----------
    model_id:
        HuggingFace model ID, e.g. ``"Qwen/Qwen2.5-VL-7B-Instruct"``.
    max_new_tokens:
        Maximum tokens generated per page.
    batch_size:
        Number of images passed to vLLM in a single call.

    Attributes
    ----------
    model_slug:
        Short name derived from *model_id* (after the last ``/``).
        Use it to build per-model output paths so runs never overwrite each
        other: ``pred_dir / predictor.model_slug``.
    """

    def __init__(
        self,
        model_id:       str,
        max_new_tokens: int = 4096,
        batch_size:     int = 8,
    ) -> None:
        self.model_slug = model_id.split("/")[-1]
        self.batch_size = batch_size
        logger.info("Loading %s …", model_id)
        self.llm        = LLM(model=model_id, dtype="bfloat16")
        self._sampling  = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)

    def predict_dir(self, pages_dir: Path, pred_dir: Path) -> None:
        """Process all page_*.jpg in *pages_dir*; write .md files to *pred_dir*.

        Already-predicted pages are skipped, so the job can be safely resumed.
        """
        jpg_files = sorted(pages_dir.glob("page_*.jpg"))
        if not jpg_files:
            logger.warning("No page_*.jpg found in %s", pages_dir)
            return

        pred_dir.mkdir(parents=True, exist_ok=True)

        pending = [p for p in jpg_files
                   if not (pred_dir / p.with_suffix(".md").name).exists()]
        n_done = len(jpg_files) - len(pending)

        if not pending:
            logger.info("All %d pages already predicted — skipping", len(jpg_files))
            return
        if n_done:
            logger.info("%d / %d pages done; predicting %d remaining",
                        n_done, len(jpg_files), len(pending))

        for i in range(0, len(pending), self.batch_size):
            batch = pending[i : i + self.batch_size]
            logger.info("  Batch %d–%d / %d",
                        i + 1, min(i + self.batch_size, len(pending)), len(pending))
            self._run_batch(batch, pred_dir)

    def predict_dataset(self, dataset_dir: Path, pred_dir: Path) -> None:
        """Process every *lang/book/pages/* sub-tree, mirroring the layout.

        Input:   dataset_dir/<lang>/<book>/pages/page_*.jpg
        Output:  pred_dir/<lang>/<book>/page_*.md
        """
        pages_dirs = sorted(dataset_dir.glob("*/*/pages"))
        if not pages_dirs:
            logger.error("No pages/ directories found under %s", dataset_dir)
            return
        for pages_dir in pages_dirs:
            rel = pages_dir.parent.relative_to(dataset_dir)
            logger.info("Book: %s", rel)
            self.predict_dir(pages_dir, pred_dir / rel)

    # ── internals ─────────────────────────────────────────────────────────────

    def _run_batch(self, jpg_paths: list[Path], pred_dir: Path) -> None:
        messages = [
            [{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": _pil_to_data_url(Image.open(p))}},
                {"type": "text", "text": _PROMPT},
            ]}]
            for p in jpg_paths
        ]
        outputs = self.llm.chat(messages, self._sampling)
        for path, out in zip(jpg_paths, outputs):
            md_name = path.with_suffix(".md").name
            (pred_dir / md_name).write_text(out.outputs[0].text.strip(),
                                            encoding="utf-8")
            logger.info("    ✓ %s", md_name)
