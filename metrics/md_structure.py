"""
Markdown Structure F1 — precision / recall on structural Markdown elements.

Extracted element types
  headings    : (level, normalised text)
  tables      : normalised cell text (position-agnostic)
  math_block  : normalised $$…$$ content
  math_inline : normalised $…$ content
  page_nums   : digit string from [p. N]
  footnotes   : key string from [^key] reference markers
  images      : normalised alt text
  list_items  : normalised list-item text

Standard elements (headings, tables, images, list items) are extracted via
the mistune AST parser.  Domain-specific elements (math, page numbers,
footnotes) are extracted with targeted regular expressions.

Per-type F1 is computed over multisets (Counter) of extracted tokens.
Overall score = macro-average F1 across element types present in the reference.

Range [0, 1] — higher is better.
"""

import re
import unicodedata
from collections import Counter

import mistune


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


# ---------------------------------------------------------------------------
# AST traversal helpers
# ---------------------------------------------------------------------------

def _text_from_nodes(nodes: list) -> str:
    """Recursively extract plain text from a list of AST nodes."""
    parts = []
    for node in nodes:
        t = node.get("type")
        if t == "text":
            parts.append(node.get("raw", ""))
        elif t in ("softline", "linebreak"):
            parts.append(" ")
        else:
            children = node.get("children")
            if children:
                parts.append(_text_from_nodes(children))
    return " ".join(p for p in parts if p)


def _iter_table_rows(table_node: dict):
    """Yield all table_row nodes from a table AST node."""
    for section in table_node.get("children", []):          # table_head / table_body
        for child in section.get("children", []):
            if child.get("type") == "table_row":
                yield child
            else:
                for sub in child.get("children", []):        # extra nesting
                    if sub.get("type") == "table_row":
                        yield sub


def _walk(nodes: list, elements: dict) -> None:
    """Recursively walk AST block nodes and populate element Counters."""
    for node in nodes:
        t = node.get("type")

        if t == "heading":
            level = node.get("attrs", {}).get("level", 1)
            text  = _normalise(_text_from_nodes(node.get("children", [])))
            elements["headings"][(level, text)] += 1

        elif t == "image":
            alt = _normalise(node.get("attrs", {}).get("alt", ""))
            elements["images"][alt] += 1

        elif t == "list_item":
            text = _normalise(_text_from_nodes(node.get("children", [])))
            elements["list_items"][text] += 1

        elif t == "table":
            for row_node in _iter_table_rows(node):
                for cell_node in row_node.get("children", []):
                    cell_text = _normalise(
                        _text_from_nodes(cell_node.get("children", []))
                    )
                    elements["tables"][cell_text] += 1

        # recurse into children (table handled above to avoid double-counting)
        if t != "table":
            children = node.get("children")
            if isinstance(children, list):
                _walk(children, elements)


# ---------------------------------------------------------------------------
# Regex-based extraction for domain-specific elements
# ---------------------------------------------------------------------------

_RE_MATH_BLOCK   = re.compile(r'\$\$(.*?)\$\$', re.DOTALL)
_RE_MATH_INLINE  = re.compile(r'\$([^\$\n]+?)\$')
_RE_PAGE_NUM     = re.compile(r'\[p\.\s*(\d+)\]')
_RE_FOOTNOTE_REF = re.compile(r'\[\^(\w+)\](?!:)')   # markers only, not definitions


def _extract_regex(text: str, elements: dict) -> None:
    for m in _RE_MATH_BLOCK.finditer(text):
        elements["math_block"][_normalise(m.group(1))] += 1

    # remove block math before searching for inline math to avoid re-matching
    text_no_block = _RE_MATH_BLOCK.sub("", text)
    for m in _RE_MATH_INLINE.finditer(text_no_block):
        elements["math_inline"][_normalise(m.group(1))] += 1

    for m in _RE_PAGE_NUM.finditer(text):
        elements["page_nums"][m.group(1)] += 1

    for m in _RE_FOOTNOTE_REF.finditer(text):
        elements["footnotes"][m.group(1)] += 1


# ---------------------------------------------------------------------------
# Main extraction entry point
# ---------------------------------------------------------------------------

_ELEMENT_TYPES = (
    "headings",
    "tables",
    "math_block",
    "math_inline",
    "page_nums",
    "footnotes",
    "images",
    "list_items",
)

_ast_parser = mistune.create_markdown(renderer="ast")


def _extract(text: str) -> dict[str, Counter]:
    elements: dict[str, Counter] = {k: Counter() for k in _ELEMENT_TYPES}
    _extract_regex(text, elements)
    try:
        tokens = _ast_parser(text)
        if isinstance(tokens, list):
            _walk(tokens, elements)
    except Exception:
        pass                          # fall back to regex-only results
    return elements


# ---------------------------------------------------------------------------
# F1 computation over multisets
# ---------------------------------------------------------------------------

def _f1(ref_c: Counter, pred_c: Counter) -> float:
    if not ref_c and not pred_c:
        return 1.0
    if not ref_c or not pred_c:
        return 0.0
    tp        = sum((ref_c & pred_c).values())
    precision = tp / sum(pred_c.values())
    recall    = tp / sum(ref_c.values())
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 6)


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class MarkdownStructureF1:
    """Precision / recall F1 over structural Markdown elements.

    Overall score = macro-average F1 across element types present in the
    reference.  Types absent from the reference are excluded from the average
    (they carry no signal).  If the reference has no structured elements at
    all, the score is 1.0 (nothing to penalise).

    Range [0, 1] — higher is better.
    """

    higher_is_better = True

    def score(self, reference: str, hypothesis: str) -> float:
        ref_elems  = _extract(reference)
        pred_elems = _extract(hypothesis)
        return self._macro_f1(ref_elems, pred_elems)

    def detailed_score(self, reference: str, hypothesis: str) -> dict[str, float]:
        """Return per-element-type F1 scores plus the overall macro-average."""
        ref_elems  = _extract(reference)
        pred_elems = _extract(hypothesis)
        detail = {k: _f1(ref_elems[k], pred_elems[k]) for k in _ELEMENT_TYPES}
        detail["overall"] = self._macro_f1(ref_elems, pred_elems)
        return detail

    @staticmethod
    def _macro_f1(
        ref_elems:  dict[str, Counter],
        pred_elems: dict[str, Counter],
    ) -> float:
        scores = [
            _f1(ref_elems[k], pred_elems[k])
            for k in _ELEMENT_TYPES
            if ref_elems[k]                   # only types present in the reference
        ]
        if not scores:
            return 1.0
        return round(sum(scores) / len(scores), 6)

    def corpus_score(self, pairs: list[tuple[str, str]]) -> float:
        if not pairs:
            return 0.0
        return round(
            sum(self.score(r, h) for r, h in pairs) / len(pairs), 4
        )
