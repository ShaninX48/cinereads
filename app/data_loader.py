"""
data_loader.py
==============
Dataset acquisition and parsing for CineReads.

Two public datasets are merged into a single document collection:

* Movies : "The Movies Dataset" (rounakbanik/the-movies-dataset) — we read
           `movies_metadata.csv` and keep the first 10,000 rows.
* Books  : "Books Dataset - 15k Books across 100 Categories"
           (mihikaajayjadhav/books-dataset-15k-books-across-100-categories).

Each row is normalised into a common document dict so the index does not have
to care whether a document is a movie or a book:

    {id, type, title, rating, year, ...type-specific fields...}

The loaders are defensive: column names differ between dataset versions, so
`find_col` looks for any of several known aliases, and malformed CSV rows are
skipped rather than aborting the whole load.
"""

import ast
import glob

import pandas as pd


# ----------------------------------------------------------------------
# Small parsing helpers
# ----------------------------------------------------------------------
def _clean(val):
    """NaN-safe string cleanup."""
    return '' if pd.isna(val) else str(val).strip()


def _float(val, default=0.0):
    """Parse a float, falling back to a default on bad input."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _year(val):
    """Extract a 4-digit year from a date string like '1994-09-10'."""
    try:
        return int(str(val)[:4])
    except (ValueError, TypeError):
        return 0


def _parse_genres(val):
    """The movies dataset stores genres as a stringified list of dicts:
        "[{'id': 18, 'name': 'Drama'}, {'id': 35, 'name': 'Comedy'}]"
    Convert that into a clean "Drama, Comedy" string. If parsing fails we
    just return the raw cleaned value.
    """
    try:
        parsed = ast.literal_eval(str(val))
        return ', '.join(g['name'] for g in parsed if isinstance(g, dict))
    except (ValueError, SyntaxError):
        return _clean(val)


def find_col(df, variants):
    """Return the first column name from `variants` that exists in `df`."""
    for v in variants:
        if v in df.columns:
            return v
    return None


# ----------------------------------------------------------------------
# Dataset download (via kagglehub)
# ----------------------------------------------------------------------
def download_datasets():
    """Download both datasets from Kaggle and return their CSV paths.

    Requires a configured Kaggle API token. Raises on failure so the caller
    can decide how to degrade.
    """
    import kagglehub

    print('Downloading books dataset...')
    books_path = kagglehub.dataset_download(
        'mihikaajayjadhav/books-dataset-15k-books-across-100-categories')

    print('Downloading movies dataset...')
    movies_path = kagglehub.dataset_download('rounakbanik/the-movies-dataset')

    book_csvs = glob.glob(books_path + '/**/*.csv', recursive=True)
    movie_csvs = glob.glob(movies_path + '/**/movies_metadata.csv', recursive=True)
    if not movie_csvs:
        movie_csvs = glob.glob(movies_path + '/**/*.csv', recursive=True)

    books_csv = book_csvs[0] if book_csvs else None
    movies_csv = next((f for f in movie_csvs if 'movies_metadata' in f),
                      movie_csvs[0] if movie_csvs else None)

    print(f'Books CSV  : {books_csv}')
    print(f'Movies CSV : {movies_csv}')
    return books_csv, movies_csv


# ----------------------------------------------------------------------
# Row -> document conversion
# ----------------------------------------------------------------------
def load_books(csv_path):
    """Load the books CSV into a list of book document dicts."""
    df = pd.read_csv(csv_path, on_bad_lines='skip')
    df.columns = [c.strip() for c in df.columns]

    # Map our canonical field -> the possible column names in the wild.
    cols = {
        'title':       ['Title', 'title', 'Book Title', 'Name'],
        'author':      ['Author', 'author', 'Authors'],
        'rating':      ['Rating', 'rating', 'Average Rating', 'Avg_Rating'],
        'category':    ['Category', 'category', 'Genre', 'Genres'],
        'description': ['Description', 'description', 'Summary', 'About'],
        'num_ratings': ['No_of_Ratings', 'Ratings_Count', 'num_ratings'],
        'year':        ['Year', 'year', 'Published Year', 'Publish_Year'],
    }

    docs = []
    for i, row in df.iterrows():
        title = _clean(row.get(find_col(df, cols['title']), ''))
        if not title:               # skip rows with no title
            continue
        docs.append({
            'id': f'b{i}', 'type': 'book', 'title': title,
            'author':      _clean(row.get(find_col(df, cols['author']), '')),
            'rating':      _float(row.get(find_col(df, cols['rating']), 0)),
            'category':    _clean(row.get(find_col(df, cols['category']), '')),
            'description': _clean(row.get(find_col(df, cols['description']), '')),
            'year':        _year(row.get(find_col(df, cols['year']), '')),
            'num_ratings': _float(row.get(find_col(df, cols['num_ratings']), 0)),
        })
    print(f'Loaded {len(docs)} books')
    return docs


def load_movies(csv_path, max_rows=10000):
    """Load the first `max_rows` of the movies CSV into movie document dicts."""
    df = pd.read_csv(csv_path, on_bad_lines='skip', nrows=max_rows, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    docs = []
    for i, row in df.iterrows():
        title = _clean(row.get('title', ''))
        if not title:
            continue
        docs.append({
            'id': f'm{i}', 'type': 'movie', 'title': title,
            # movies_metadata.csv has no director column (it lives in
            # credits.csv), so we leave it blank; the UI hides empty fields.
            'director': '',
            'genres':   _parse_genres(row.get('genres', '')),
            'rating':   _float(row.get('vote_average', 0)),
            'overview': _clean(row.get('overview', '')),
            'year':     _year(row.get('release_date', '')),
            'language': _clean(row.get('original_language', '')),
        })
    print(f'Loaded {len(docs)} movies')
    return docs
