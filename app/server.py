"""
server.py
=========
The Flask web layer for CineReads.

It exposes a tiny JSON API on top of an already-built `InvertedIndex` and
serves the single-page front-end. Keeping the routes in their own module
means the search engine (`inverted_index.py`) stays completely independent of
the web framework.

Endpoints
---------
GET /              -> the search UI (static/cinereads_ui.html)
GET /api/search    -> ?q=&mode=AND|OR&type=all|movie|book&k=10  -> ranked results
GET /api/stats     -> corpus statistics for the header pills
GET /api/suggest   -> ?q=  -> up to 8 title autocomplete suggestions
"""

import os

from flask import Flask, request, jsonify, send_file

# Folder that holds the front-end file.
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'static')
HTML_PATH = os.path.join(STATIC_DIR, 'cinereads_ui.html')


def create_app(index):
    """Application factory: build a Flask app bound to a given index."""
    app = Flask(__name__)

    @app.route('/')
    def home():
        """Serve the single-page UI."""
        return send_file(os.path.abspath(HTML_PATH))

    @app.route('/api/search')
    def search_api():
        """Main search endpoint. Reads query params, delegates to the index."""
        q = request.args.get('q', '').strip()
        mode = request.args.get('mode', 'OR').upper()
        doc_type = request.args.get('type', 'all').lower()
        top_k = int(request.args.get('k', 10))

        if not q:
            return jsonify({'results': [], 'query': '', 'total': 0})

        results = index.search(q, mode=mode, doc_type=doc_type, top_k=top_k)
        return jsonify({
            'query': q, 'mode': mode, 'type': doc_type,
            'total': len(results), 'results': results,
        })

    @app.route('/api/stats')
    def stats_api():
        """Corpus counts shown in the UI header."""
        return jsonify(index.stats())

    @app.route('/api/suggest')
    def suggest_api():
        """Prefix/substring title autocomplete.

        Returns titles that contain the query substring, ordered so that
        earlier matches (e.g. titles that *start* with the query) come first.
        """
        q = request.args.get('q', '').strip().lower()
        if len(q) < 2:
            return jsonify({'suggestions': []})

        titles = [doc['title'] for doc in index.doc_store.values()
                  if q in doc['title'].lower()]
        titles.sort(key=lambda t: t.lower().index(q))
        return jsonify({'suggestions': titles[:8]})

    return app
