#  CineReads — Movie & Book Search Engine

A from-scratch information-retrieval system that indexes **25,139 documents**
(10,000 movies + 15,139 books) in a single **inverted index** and ranks results
with **TF-IDF**. It supports Boolean **AND / OR** queries, type filtering,
title autocomplete, relevance snippets, and a dark, responsive single-page UI.

> Built for the Information Retrieval / Search Engine lab assignment.
> No search library is used — the index, weighting, and ranking are implemented
> by hand.

---

## Features

- **Inverted index** with positional postings lists (`term → {doc_id → [positions]}`)
- **Preprocessing**: lowercase → punctuation strip → tokenise → stopword removal → Porter stemming
- **TF-IDF ranking**: log-normalised TF `(1 + log tf)` × IDF `log(N/df)`, plus a small rating boost
- **Boolean retrieval**: AND (set intersection) and OR (set union) over postings
- **Type filter**: search *All*, *Movies only*, or *Books only*
- **Smart snippets**: shows the 30-word window of the description with the most query hits
- **Autocomplete** title suggestions as you type
- **Cached index** (`search_index.pkl`) so the app starts instantly after the first build

---

## Project structure

```
cinereads/
├── run.py                 # entry point: build/load index, start the server
├── test_search.py         # CLI smoke test for the search engine
├── requirements.txt
├── search_index.pkl       # pre-built index cache (so no Kaggle download is needed)
├── app/
│   ├── __init__.py
│   ├── inverted_index.py  # the IR engine: preprocessing, index, TF-IDF, search
│   ├── data_loader.py     # dataset download + CSV → document parsing
│   └── server.py          # Flask routes (/, /api/search, /api/stats, /api/suggest)
└── static/
    └── cinereads_ui.html  # single-page front-end
```

The search engine (`app/inverted_index.py`) has **no web dependency** and can be
imported and tested on its own.

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run (the cached index loads automatically — no download needed)
python run.py

# 3. Open the browser (auto-opens) at:
#    http://localhost:5000
```

### Rebuilding the index from scratch (optional)

The repo ships with `search_index.pkl` so it runs out of the box. To rebuild
from the original data, delete the cache and configure a
[Kaggle API token](https://www.kaggle.com/docs/api):

```bash
rm search_index.pkl
python run.py          # downloads datasets, re-indexes, re-caches
```

Datasets used:
- Movies — `rounakbanik/the-movies-dataset` (`movies_metadata.csv`, first 10k rows)
- Books — `mihikaajayjadhav/books-dataset-15k-books-across-100-categories`

---

## Try it from the command line

```bash
python test_search.py "space adventure"          # default: AND, all types
python test_search.py "crime thriller" AND book   # books only
python test_search.py "love war" OR               # OR mode
```

---

## API reference

| Endpoint       | Params                                   | Returns                          |
|----------------|------------------------------------------|----------------------------------|
| `GET /`        | —                                        | the search UI                    |
| `GET /api/search` | `q`, `mode=AND\|OR`, `type=all\|movie\|book`, `k` | ranked results (JSON)   |
| `GET /api/stats`  | —                                     | corpus counts                    |
| `GET /api/suggest`| `q`                                   | up to 8 title suggestions        |

Example:

```
GET /api/search?q=vampire%20romance&mode=AND&type=all&k=10
```

---

## How ranking works

For a query, each candidate document is scored as:

```
score(doc) = Σ  TF(term, doc) · IDF(term)   +   0.3 · (rating / max_rating)
            term ∈ query

TF(t, d)  = 1 + log(count(t, d))        # 0 if the term is absent
IDF(t)    = log(N / df(t))              # N = 25,139 documents
```

The rating boost is intentionally small (≤ 0.3) so that **textual relevance
dominates** and well-rated titles only break ties between similar matches.

---

## Known limitations / future work

- **Director field is empty** for movies (it lives in a separate `credits.csv`,
  not in `movies_metadata.csv`); joining it in would enrich movie search.
- **Many book ratings are 0**, so the rating boost mostly affects movies today.
- No phrase or proximity search yet, even though positions are stored — a
  natural next step given the positional postings already exist.
- No spell-correction; misspelled queries return nothing in AND mode.

---

##  Authors

- **MD Tanveer Mahmood Shanin** — ID: 0432220005101117
- **Joydev Datta** — ID: 0432220005101132
- **Course:** CSE 426 — Information Retrieval, University of Information Technology and Sciences (UITS), Dhaka
- **GitHub:** https://github.com/ShaninX48
