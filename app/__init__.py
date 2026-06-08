"""CineReads application package.

Exposes the core pieces so they can be imported as:
    from app import InvertedIndex, create_app
"""

from .inverted_index import InvertedIndex
from .server import create_app

__all__ = ['InvertedIndex', 'create_app']
