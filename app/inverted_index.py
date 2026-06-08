"""
inverted_index.py
=================
Core information-retrieval engine for CineReads.

This module implements a classic **inverted index** with TF-IDF ranking and
Boolean (AND / OR) query processing. It is deliberately framework-free: it has
no dependency on Flask or on any dataset, so it can be unit-tested and reused
on its own.

Pipeline overview
-----------------
1. preprocess(text)      -> normalise, tokenise, remove stopwords, stem
2. add_document(doc)     -> tokenise one document and update the postings lists
3. build_from_list(docs) -> index a whole corpus
4. search(query, ...)    -> Boolean candidate selection + TF-IDF ranking

Index data structures
----------------------
    index[term][doc_id] = [pos1, pos2, ...]   # positional postings list
    term_df[term]       = number of documents containing the term
    doc_store[doc_id]   = the original document dict (for display)
    doc_lengths[doc_id] = token count of the document
    total_docs          = N, used for IDF
"""

import re
import math
import pickle
from collections import defaultdict

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


class InvertedIndex:
    """An in-memory positional inverted index with TF-IDF scoring."""

    def __init__(self, use_stemming=True, remove_stopwords=True):
        # index[term] -> {doc_id -> [positions]}.  defaultdict avoids key checks.
        self.index = defaultdict(lambda: defaultdict(list))
        self.doc_store = {}                  # doc_id -> full document dict
        self.doc_lengths = defaultdict(int)  # doc_id -> number of tokens
        self.term_df = defaultdict(int)      # term   -> document frequency
        self.total_docs = 0                  # N (corpus size)

        # Preprocessing switches (kept configurable for experiments / ablation).
        self.use_stemming = use_stemming
        self.remove_stopwords = remove_stopwords
        self.stemmer = PorterStemmer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            # If the NLTK stopword corpus is missing we degrade gracefully
            # rather than crash; stopword removal simply becomes a no-op.
            self.stop_words = set()

    # ------------------------------------------------------------------
    # Text preprocessing
    # ------------------------------------------------------------------
    def preprocess(self, text):
        """Turn raw text into a clean list of index terms.

        Steps: lowercase -> remove punctuation -> split on whitespace ->
        drop stopwords and 1-char tokens -> Porter stemming. The *same*
        function is applied to documents at index time and to queries at
        search time, which guarantees that query terms and indexed terms
        live in the same vocabulary.
        """
        if not text or not isinstance(text, str):
            return []
        text = text.lower()
        # Replace anything that is not a letter/digit/space with a space.
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        tokens = text.split()
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words and len(t) > 1]
        if self.use_stemming:
            tokens = [self.stemmer.stem(t) for t in tokens]
        return tokens

    def _build_text(self, doc):
        """Concatenate the searchable fields of a document into one string.

        Every field listed here contributes to the document's bag of words.
        Movies and books have different fields, so we simply include whichever
        are present (missing fields are skipped).
        """
        parts = []
        for field in ('title', 'author', 'director',
                      'description', 'overview', 'category', 'genres'):
            value = doc.get(field, '')
            if value:
                parts.append(str(value))
        return ' '.join(parts)

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------
    def add_document(self, doc):
        """Tokenise a single document and fold it into the index."""
        doc_id = doc['id']
        self.doc_store[doc_id] = doc
        self.total_docs += 1

        tokens = self.preprocess(self._build_text(doc))
        self.doc_lengths[doc_id] = len(tokens)

        # `seen` makes sure each term increments document frequency only once
        # per document, even if it occurs many times.
        seen = set()
        for position, term in enumerate(tokens):
            self.index[term][doc_id].append(position)
            if term not in seen:
                self.term_df[term] += 1
                seen.add(term)

    def build_from_list(self, documents, verbose=True):
        """Index a list of document dicts."""
        for i, doc in enumerate(documents):
            self.add_document(doc)
            if verbose and (i + 1) % 2000 == 0:
                print(f'  Indexed {i + 1}/{len(documents)} docs...')
        if verbose:
            print(f'  Done: {self.total_docs} docs, {len(self.index)} unique terms')

    # ------------------------------------------------------------------
    # TF-IDF weighting
    # ------------------------------------------------------------------
    def tf(self, term, doc_id):
        """Log-normalised term frequency: 1 + log(count), or 0 if absent.

        Log normalisation dampens the effect of very frequent terms so that a
        word appearing 100 times does not outweigh everything else linearly.
        """
        count = len(self.index[term].get(doc_id, []))
        return 1 + math.log(count) if count > 0 else 0.0

    def idf(self, term):
        """Inverse document frequency: log(N / df).

        Rare terms (small df) get a high weight; terms that appear in almost
        every document get a weight close to zero.
        """
        df = self.term_df.get(term, 0)
        return math.log(self.total_docs / df) if df > 0 else 0.0

    def tfidf(self, term, doc_id):
        """The classic TF-IDF weight of a term in a document."""
        return self.tf(term, doc_id) * self.idf(term)

    def _score(self, doc_id, query_terms):
        """Rank score = sum of TF-IDF over query terms + small rating boost.

        The rating boost (max 0.3) is a light tie-breaker so that, among
        documents with similar textual relevance, better-rated titles surface
        first. It is intentionally small so that textual relevance dominates.
        """
        score = sum(self.tfidf(t, doc_id) for t in query_terms)
        doc = self.doc_store.get(doc_id, {})
        rating = float(doc.get('rating') or 0)
        max_r = 10.0 if doc.get('type') == 'movie' else 5.0  # different scales
        score += 0.3 * (rating / max_r)
        return score

    # ------------------------------------------------------------------
    # Boolean query processing
    # ------------------------------------------------------------------
    def _postings(self, term):
        """The set of document ids that contain `term`."""
        return set(self.index.get(term, {}).keys())

    def boolean_and(self, terms):
        """Documents containing ALL query terms (set intersection)."""
        if not terms:
            return set()
        result = self._postings(terms[0])
        for t in terms[1:]:
            result &= self._postings(t)
        return result

    def boolean_or(self, terms):
        """Documents containing ANY query term (set union)."""
        result = set()
        for t in terms:
            result |= self._postings(t)
        return result

    # ------------------------------------------------------------------
    # Snippet generation
    # ------------------------------------------------------------------
    def _snippet(self, doc, query_terms, max_words=30):
        """Pick the ~30-word window of the description with the most query hits.

        Instead of always showing the first sentence, we slide a window over
        the text and keep the window that contains the most (stemmed) query
        terms, which produces a far more relevant preview.
        """
        text = doc.get('description') or doc.get('overview') or ''
        if not text:
            return ''
        words = text.split()
        if len(words) <= max_words:
            return text

        best_start, best_hits = 0, 0
        for i in range(len(words) - max_words):
            window = [self.stemmer.stem(w.lower()) for w in words[i:i + max_words]]
            hits = sum(1 for t in query_terms if t in window)
            if hits > best_hits:
                best_hits, best_start = hits, i

        snippet = ' '.join(words[best_start:best_start + max_words])
        return ('...' if best_start > 0 else '') + snippet + '...'

    # ------------------------------------------------------------------
    # Public search API
    # ------------------------------------------------------------------
    def search(self, query, mode='OR', doc_type='all', top_k=10):
        """Run a full query and return ranked, display-ready results.

        1. Preprocess the query into the same term space as the index.
        2. Select candidate documents with Boolean AND or OR.
        3. Optionally filter by type ('movie' / 'book').
        4. Score every candidate with TF-IDF (+ rating boost) and sort.
        5. Attach a relevance snippet and the list of matched terms.
        """
        terms = self.preprocess(query)
        if not terms:
            return []

        # (2) candidate set
        if mode.upper() == 'AND':
            candidates = self.boolean_and(terms)
        else:
            candidates = self.boolean_or(terms)

        # (3) type filter
        if doc_type != 'all':
            candidates = {d for d in candidates
                          if self.doc_store.get(d, {}).get('type') == doc_type}

        # (4) score + sort (descending)
        scored = sorted(
            [(d, self._score(d, terms)) for d in candidates],
            key=lambda x: x[1], reverse=True,
        )

        # (5) build response objects
        results = []
        for doc_id, score in scored[:top_k]:
            doc = dict(self.doc_store[doc_id])         # copy so we don't mutate the store
            doc['score'] = round(score, 4)
            doc['snippet'] = self._snippet(doc, terms)
            doc['matched_terms'] = [t for t in terms if doc_id in self.index.get(t, {})]
            results.append(doc)
        return results

    # ------------------------------------------------------------------
    # Persistence (so we don't re-index 25k docs on every launch)
    # ------------------------------------------------------------------
    def save(self, path):
        """Pickle the index to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'index': dict(self.index),
                'doc_store': self.doc_store,
                'doc_lengths': dict(self.doc_lengths),
                'term_df': dict(self.term_df),
                'total_docs': self.total_docs,
            }, f)
        print(f'Index saved -> {path}')

    def load(self, path):
        """Restore a previously pickled index."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.index = defaultdict(lambda: defaultdict(list), data['index'])
        self.doc_store = data['doc_store']
        self.doc_lengths = defaultdict(int, data['doc_lengths'])
        self.term_df = defaultdict(int, data['term_df'])
        self.total_docs = data['total_docs']
        print(f'Index loaded: {self.total_docs} docs, {len(self.index)} terms')

    def stats(self):
        """Summary counts used by the /api/stats endpoint and the UI."""
        return {
            'total_documents': self.total_docs,
            'unique_terms': len(self.index),
            'movies': sum(1 for d in self.doc_store.values() if d.get('type') == 'movie'),
            'books': sum(1 for d in self.doc_store.values() if d.get('type') == 'book'),
        }
