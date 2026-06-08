"""
run.py
======
CineReads entry point.

Usage:
    pip install -r requirements.txt
    python run.py
    # then open http://localhost:5000

On first run, if no cached index (`search_index.pkl`) exists, it downloads the
datasets from Kaggle, builds the inverted index, and caches it. On every later
run it simply loads the cache, so start-up is fast.
"""

import os
import time
import threading
import webbrowser

from app import InvertedIndex, create_app
from app.data_loader import download_datasets, load_books, load_movies

# Path of the cached, pre-built index.
INDEX_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'search_index.pkl')


def build_index():
    """Return a ready-to-query InvertedIndex (from cache if available)."""
    index = InvertedIndex()

    if os.path.exists(INDEX_CACHE):
        print('Cached index found -> loading...')
        index.load(INDEX_CACHE)
        print(f'Index stats: {index.stats()}')
        return index

    # No cache: build from the raw datasets.
    print('No cache found. Downloading datasets...')
    try:
        books_csv, movies_csv = download_datasets()
    except Exception as e:
        print(f'Kaggle download failed: {e}')
        print('Set up your Kaggle API token and try again.')
        books_csv, movies_csv = None, None

    all_docs = []
    if books_csv:
        all_docs += load_books(books_csv)
    if movies_csv:
        all_docs += load_movies(movies_csv)

    if not all_docs:
        raise SystemExit('No data to index. Configure the Kaggle API and retry.')

    print(f'Building index for {len(all_docs)} documents...')
    index.build_from_list(all_docs, verbose=True)
    index.save(INDEX_CACHE)
    print(f'Index stats: {index.stats()}')
    return index


def main():
    print('=' * 55)
    print('  CineReads - Movie & Book Search Engine')
    print('=' * 55)

    index = build_index()
    app = create_app(index)

    print('\n' + '=' * 55)
    print('Server starting at http://localhost:5000  (Ctrl+C to stop)')
    print('=' * 55 + '\n')

    # Open the browser shortly after the server comes up.
    def _open_browser():
        time.sleep(2)
        webbrowser.open('http://localhost:5000')

    threading.Thread(target=_open_browser, daemon=True).start()

    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)


if __name__ == '__main__':
    main()
