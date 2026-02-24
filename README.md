<div align="center">

# Book2md_Bench

</div>

A benchmark dataset builder for document understanding tasks.
It downloads EPUB books from [Project Gutenberg](https://www.gutenberg.org/), converts each section to clean Markdown, splits it into page-sized chunks, and renders each chunk as a JPEG image via LaTeX.

## Pipeline

1. Search and download EPUBs from Project Gutenberg
2. Parse EPUB spine → extract HTML sections in reading order
3. Convert each HTML section → clean Markdown (ground truth)
4. Split sections into page-sized chunks:
   - primary: split at `[p. N]` markers (original book pagination)
   - fallback: split into ~2800-char blocks at paragraph boundaries
5. Sample chunks stratified across front / body / back zones
6. Render each sampled chunk to JPEG (`chunk.md → PDF (xelatex) → JPEG`)
7. Save aligned `(image, markdown)` pairs with metadata

Since both the JPEG and the Markdown come from the same source text, image and ground truth always represent exactly the same content.

## Markdown Ground Truth

Each `.md` file preserves:

| Element | Format |
|---|---|
| Heading hierarchy | `#` `##` `###` |
| Tables | Markdown pipe tables |
| Math | `$...$` inline, `$$...$$` block |
| Lists and indentation | unchanged |
| Images | `![image_N](images/image_N.png)` |
| Book page numbers | `[p. N]` |
| Footnotes | `[^N]` markers and definitions |

## Dataset Parameters

| Parameter | Value |
|---|---|
| Languages | Italian, German |
| Books per language | 15 |
| Pages per book | 20 |
| Sampling strata | front: 2, body: 10, back: 5 (+ 3 mandatory frontmatter) |
| Image DPI | 200 |
| JPEG quality | 92 |

## Installation

### 1. System dependencies

```bash
sudo apt-get install -y pandoc poppler-utils texlive-xetex texlive-lang-italian texlive-lang-german
```

On Google Colab:

```python
!apt-get update -q
!apt-get install -y pandoc poppler-utils texlive-xetex texlive-lang-italian texlive-lang-german
!pip install -r requirements.txt -q
```

### 2. Python dependencies

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

Output will be saved to `./benchmark_data/`.

## Project Structure

```
book_mdBench/
├── book_mdBench/          # Python package (core logic)
│   ├── __init__.py
│   ├── config.py
│   ├── gutenberg_client.py
│   ├── epub_converter.py  # EPUB spine parsing + HTML → Markdown
│   ├── page_sampler.py    # page-sized chunk splitting + stratified sampling
│   └── page_renderer.py   # Markdown → PDF (xelatex) → JPEG
│
├── main.py                # Entry point
├── requirements.txt
└── README.md
```

## Output Structure

```
benchmark_data/
├── metadata.json
├── italian/
│   ├── metadata.json
│   └── <book_id>_<title>/
│       ├── book.epub
│       ├── book.md        # full book markdown (concatenation of all sections)
│       └── pages/
│           ├── page_0001.md
│           └── page_0001.jpg
└── german/
    └── ...
```
