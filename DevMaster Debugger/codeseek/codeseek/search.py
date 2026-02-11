"""
Search engine combining semantic and text-based search.
"""

import re
from typing import Optional

import numpy as np

from .embeddings import CodeEmbedder
from .models import Query, SearchResult, SymbolType
from .storage import CodeStore


class SearchEngine:
    """Semantic code search engine."""

    def __init__(self, store: CodeStore, embedder: Optional[CodeEmbedder] = None):
        self.store = store
        self.embedder = embedder

    def search(
        self,
        query_str: str,
        top_k: int = 20,
        symbol_type: Optional[SymbolType] = None,
        language: Optional[str] = None,
        use_semantic: bool = True
    ) -> list[SearchResult]:
        """
        Search for code matching the query.

        Args:
            query_str: Search query (natural language or keywords)
            top_k: Number of results to return
            symbol_type: Filter by symbol type (function, class, etc.)
            language: Filter by language
            use_semantic: Use semantic search (requires embeddings)

        Returns:
            List of search results
        """
        # Parse query
        query = self._parse_query(query_str)

        results = []

        # Full-text search
        fts_results = self._text_search(query, limit=top_k * 2)
        for symbol in fts_results:
            results.append(SearchResult(
                symbol=symbol,
                score=0.7,  # Base score for text matches
                match_type="text"
            ))

        # Semantic search (if enabled and embedder available)
        if use_semantic and self.embedder:
            semantic_results = self._semantic_search(query, top_k=top_k * 2)
            for symbol, score in semantic_results:
                # Check if already in results
                existing = next((r for r in results if r.symbol.id == symbol.id), None)
                if existing:
                    # Boost score if found in both
                    existing.score = max(existing.score, score * 1.2)
                    existing.match_type = "hybrid"
                else:
                    results.append(SearchResult(
                        symbol=symbol,
                        score=score,
                        match_type="semantic"
                    ))

        # Apply filters
        if symbol_type:
            results = [r for r in results if r.symbol.symbol_type == symbol_type]

        if language:
            results = [r for r in results if r.symbol.language.value == language.lower()]

        # Sort by score and return top k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def find_similar(
        self,
        symbol_id: int,
        top_k: int = 10
    ) -> list[SearchResult]:
        """Find code symbols similar to a given symbol."""
        if not self.embedder:
            return []

        # Get embedding for the symbol
        embedding = self.store.get_embedding(symbol_id)
        if not embedding:
            return []

        # Find similar embeddings
        similar = self._find_similar_embeddings(embedding.vector, top_k=top_k + 1)

        # Remove the symbol itself and create results
        results = []
        for other_id, score in similar:
            if other_id == symbol_id:
                continue

            symbol = self.store.get_symbol(other_id)
            if symbol:
                results.append(SearchResult(
                    symbol=symbol,
                    score=score,
                    match_type="similarity"
                ))

        return results[:top_k]

    def _parse_query(self, query_str: str) -> Query:
        """Parse search query to extract intent and filters."""
        query = Query(raw_query=query_str, intent="search", semantic_query=query_str)

        # Simple intent detection
        lower = query_str.lower()

        if "function" in lower or "def " in lower or "func " in lower:
            query.filters["symbol_type"] = SymbolType.FUNCTION
            query.intent = "find_functions"

        if "class" in lower:
            query.filters["symbol_type"] = SymbolType.CLASS
            query.intent = "find_classes"

        if "async" in lower:
            query.filters["async"] = True

        # Extract quoted phrases as exact matches
        quotes = re.findall(r'"([^"]+)"', query_str)
        if quotes:
            query.semantic_query = " ".join(quotes)

        return query

    def _text_search(self, query: Query, limit: int = 20) -> list:
        """Full-text search using SQLite FTS."""
        # Build FTS query
        fts_query = query.semantic_query

        # Handle common search terms
        fts_query = fts_query.replace(" or ", " OR ")
        fts_query = fts_query.replace(" and ", " AND ")

        try:
            return self.store.search_symbols_fts(fts_query, limit=limit)
        except Exception as e:
            # Fallback to simple search
            return []

    def _semantic_search(self, query: Query, top_k: int = 20) -> list[tuple]:
        """Semantic search using embeddings."""
        if not self.embedder:
            return []

        # Generate query embedding
        query_vector = self.embedder.embed_query(query.semantic_query)

        # Find similar embeddings
        similar = self._find_similar_embeddings(query_vector, top_k=top_k)

        # Get symbols
        results = []
        for symbol_id, score in similar:
            symbol = self.store.get_symbol(symbol_id)
            if symbol:
                results.append((symbol, score))

        return results

    def _find_similar_embeddings(
        self,
        query_vector: list[float],
        top_k: int = 10
    ) -> list[tuple[int, float]]:
        """Find most similar embeddings using cosine similarity."""
        # Get all embeddings
        all_embeddings = self.store.get_all_embeddings()

        if not all_embeddings:
            return []

        # Convert to numpy arrays
        query_arr = np.array(query_vector)
        query_norm = np.linalg.norm(query_arr)

        if query_norm == 0:
            return []

        similarities = []

        for symbol_id, vector in all_embeddings:
            vec_arr = np.array(vector)
            vec_norm = np.linalg.norm(vec_arr)

            if vec_norm == 0:
                continue

            # Cosine similarity
            similarity = np.dot(query_arr, vec_arr) / (query_norm * vec_norm)
            similarities.append((symbol_id, float(similarity)))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]


class QueryUnderstanding:
    """Advanced query understanding for natural language queries."""

    @staticmethod
    def expand_query(query: str) -> list[str]:
        """Expand query with synonyms and related terms."""
        expansions = {
            "authentication": ["auth", "login", "password", "credential"],
            "validation": ["validate", "check", "verify"],
            "error handling": ["exception", "try catch", "error", "failure"],
            "HTTP": ["request", "fetch", "api", "endpoint"],
            "database": ["db", "sql", "query", "select"],
            "async": ["asynchronous", "await", "promise"],
        }

        query_lower = query.lower()
        expanded_terms = [query]

        for key, synonyms in expansions.items():
            if key in query_lower:
                expanded_terms.extend(synonyms)

        return list(set(expanded_terms))
