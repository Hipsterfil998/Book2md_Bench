<div align="center">

# Book2md_Bench

</div>

A **multilingual** benchmark dataset builder for document understanding tasks.
It downloads EPUB books from [Project Gutenberg](https://www.gutenberg.org/) and supports any language available in its catalogue — simply add the language code to `config.py`.
Each section is converted to clean Markdown, split into page-sized chunks, and rendered as a JPEG image via LaTeX.

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
| Books per language | 1 (`N_BOOKS` in `config.py`) |
| Pages per book | 20 |
| Mandatory frontmatter pages | 3 |
| Sampling strata | front: 2, body: 10, back: 5 |
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

### Build the dataset

```bash
python main.py
```

Output will be saved to `./dataset/`.

### Evaluate predictions

Single pair:

```bash
python eval.py ground_truth.md prediction.md
```

Batch (all `.md` files in a directory):

```bash
python eval.py --ref-dir dataset/italian/book_123/pages \
               --pred-dir predictions/italian/book_123
```

Metrics computed: **NED** (Normalised Edit Distance, lower is better) and **BLEU** (higher is better).

## Project Structure

```
book_mdBench/
├── main.py                # Entry point — builds the dataset
├── eval.py                # Evaluation script (NED + BLEU)
├── config.py              # Global parameters
│
├── book2md/               # Core pipeline package
│   ├── gutenberg_client.py  # Project Gutenberg search + EPUB download
│   ├── epub_converter.py    # EPUB spine parsing + HTML → Markdown
│   ├── page_sampler.py      # Chunk splitting + stratified sampling
│   └── page_renderer.py     # Markdown → PDF (xelatex) → JPEG
│
└── metrics/               # Evaluation metrics
    ├── ned.py               # Normalised Edit Distance
    └── bleu.py              # BLEU score (sacrebleu)
```

## Output Structure

```
dataset/
├── metadata.json
├── italian/
│   ├── metadata.json
│   └── <book_id>_<title>/
│       ├── book.epub
│       ├── book.md          # full book Markdown
│       └── pages/
│           ├── page_0001.md
│           └── page_0001.jpg
└── german/
    └── ...
```
