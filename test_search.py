"""
test_search.py
==============
Quick command-line smoke test for the search engine (no web server needed).

Run:
    python test_search.py "space adventure"
    python test_search.py "crime thriller" AND movie

It loads the cached index and prints the ranked results, which is handy for
debugging ranking behaviour and for capturing sample output for the report.
"""

import sys
import os

from app import InvertedIndex

INDEX_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'search_index.pkl')


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else 'space adventure'
    mode = sys.argv[2].upper() if len(sys.argv) > 2 else 'AND'
    doc_type = sys.argv[3].lower() if len(sys.argv) > 3 else 'all'

    index = InvertedIndex()
    index.load(INDEX_CACHE)

    print(f'\nQuery: "{query}"  | mode={mode}  | type={doc_type}')
    print(f'Preprocessed terms: {index.preprocess(query)}\n')

    results = index.search(query, mode=mode, doc_type=doc_type, top_k=10)
    if not results:
        print('No results.')
        return

    for i, r in enumerate(results, 1):
        meta = r.get('author') or r.get('genres') or ''
        print(f'{i:>2}. [{r["type"]:<5}] score={r["score"]:<8} {r["title"][:55]}')
        if meta:
            print(f'      {meta[:70]}')


if __name__ == '__main__':
    main()
