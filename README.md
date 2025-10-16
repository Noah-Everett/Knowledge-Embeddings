arXiv metadata harvester

This repository contains a small arXiv harvester `arxiv_metadata_harvester.py` that uses OAI-PMH (via Sickle) to download arXiv metadata and optionally PDFs.

Output formats

- JSONL (recommended): each line is a JSON object with the following keys:
  - id, title, abstract
  - authors: list of readable author strings (best-effort normalization)
  - categories: list of category strings
  - datestamp: datestamp of the record
  - sets: list of OAI setSpecs, e.g. `physics:hep-ex`
  - doi, comments, journal_ref
  - raw_metadata: trimmed raw metadata dictionary (promoted fields removed). The key `name_parts` contains original name parts when available.

- CSV: flattened view with columns: id, title, abstract, authors (comma-separated), categories (comma-separated), datestamp, sets, doi, comments, journal_ref

Usage examples

Download metadata for hep-ex papers from 2018–2022 into JSONL (recommended):

```bash
python3 ./arxiv_metadata_harvester.py \
  --set hep-ex \
  --from-date 2018-01-01 \
  --until-date 2022-12-31 \
  --output hep-ex-2018-2022.jsonl \
  --chunk-days 30 \
  --batch-delay 0.1
```

Fetch PDFs (be polite and check arXiv policy):

```bash
python3 ./arxiv_metadata_harvester.py --set hep-ex --from-date 2018-01-01 --until-date 2022-12-31 --output hep-ex.jsonl --download-pdfs --pdf-dir hep-ex-pdfs --chunk-days 30 --batch-delay 0.2
```

Notes

- The script will try to resolve short set names (e.g., `hep-ex`) to a full OAI setSpec (e.g., `physics:hep-ex`).
- Use `--max-records` for quick tests.
- Use `--checkpoint path` to persist resumption tokens and resume interrupted runs.

Dependencies

See `requirements.txt`. Install with:

```bash
python3 -m pip install -r requirements.txt
```

Notebooks

- `01-embed.ipynb`: Load JSONL metadata, build SPECTER2 embeddings, and save to `hepex-embeddings.npz`.
- `02-pca.ipynb`: Load `.npz` and visualize a 2D PCA colored by category.
- `03-closest-pairs.ipynb`: Find closest paper pairs in the embedding space.
- `04-clusters-cross-category.ipynb`: Cluster per category and report closest cross-category clusters.
- `05-success-analysis.ipynb`: Find nearest neighbors to a synthetic "success" vector.
