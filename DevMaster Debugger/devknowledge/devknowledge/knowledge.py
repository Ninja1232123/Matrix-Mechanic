"""
Main knowledge graph engine that coordinates all components.
"""

from pathlib import Path
from typing import Optional

from .code_parser import UniversalCodeParser, extract_tags_from_content
from .embeddings import EmbeddingEngine, HybridSearch
from .linker import LinkDiscovery
from .models import Document, DocumentType, Language, SearchResult
from .storage import KnowledgeStore


class KnowledgeGraph:
    """Main interface for the knowledge graph system."""

    def __init__(self, db_path: str = "~/.devknowledge/kb.db"):
        self.store = KnowledgeStore(db_path)
        self.embedding_engine = EmbeddingEngine()
        self.code_parser = UniversalCodeParser()
        self.hybrid_search = HybridSearch(self.store, self.embedding_engine)
        self.link_discovery = LinkDiscovery(self.store)

    def add_note(
        self,
        title: str,
        content: str,
        tags: Optional[list[str]] = None
    ) -> Document:
        """Add a new note to the knowledge base."""
        doc = Document(
            title=title,
            content=content,
            doc_type=DocumentType.NOTE,
            language=Language.TEXT,
            tags=tags or []
        )

        # Auto-generate tags
        doc.tags = extract_tags_from_content(doc)

        # Save document
        doc.id = self.store.add_document(doc)

        # Generate and store embedding
        embedding = self.embedding_engine.embed_document(doc)
        self.store.add_embedding(embedding)

        # Discover links
        links = self.link_discovery.discover_links_for_document(doc.id)
        for link in links:
            self.store.add_link(link)

        return doc

    def add_code(
        self,
        title: str,
        content: str,
        language: Optional[Language] = None,
        tags: Optional[list[str]] = None
    ) -> Document:
        """Add a code snippet to the knowledge base."""
        # Detect language if not provided
        if not language:
            language = self.code_parser.detect_language(content)

        doc = Document(
            title=title,
            content=content,
            doc_type=DocumentType.CODE,
            language=language,
            tags=tags or []
        )

        # Parse code to extract symbols and imports
        doc = self.code_parser.parse_document(doc)

        # Auto-generate tags
        doc.tags = extract_tags_from_content(doc)

        # Save document
        doc.id = self.store.add_document(doc)

        # Generate and store embedding
        embedding = self.embedding_engine.embed_document(doc)
        self.store.add_embedding(embedding)

        # Discover links
        links = self.link_discovery.discover_links_for_document(doc.id)
        for link in links:
            self.store.add_link(link)

        return doc

    def add_documentation(
        self,
        title: str,
        content: str,
        url: Optional[str] = None,
        tags: Optional[list[str]] = None
    ) -> Document:
        """Add documentation to the knowledge base."""
        doc = Document(
            title=title,
            content=content,
            doc_type=DocumentType.DOCUMENTATION,
            language=Language.MARKDOWN,
            tags=tags or [],
            metadata={"url": url} if url else {}
        )

        # Auto-generate tags
        doc.tags = extract_tags_from_content(doc)

        # Save document
        doc.id = self.store.add_document(doc)

        # Generate and store embedding
        embedding = self.embedding_engine.embed_document(doc)
        self.store.add_embedding(embedding)

        # Discover links
        links = self.link_discovery.discover_links_for_document(doc.id)
        for link in links:
            self.store.add_link(link)

        return doc

    def search(
        self,
        query: str,
        top_k: int = 10,
        doc_type: Optional[DocumentType] = None
    ) -> list[SearchResult]:
        """Search the knowledge base."""
        results = self.hybrid_search.search(query, top_k=top_k)

        # Filter by document type if specified
        if doc_type:
            results = [r for r in results if r.document.doc_type == doc_type]

        return results

    def find_related(
        self,
        doc_id: int,
        top_k: int = 10
    ) -> list[SearchResult]:
        """Find documents related to a given document."""
        from .embeddings import VectorSearch

        vector_search = VectorSearch(self.store)
        return vector_search.find_similar(doc_id, top_k=top_k)

    def get_document(self, doc_id: int) -> Optional[Document]:
        """Get a document by ID."""
        return self.store.get_document(doc_id)

    def update_document(self, doc: Document):
        """Update a document and recompute embeddings/links."""
        # Update document
        self.store.update_document(doc)

        # Regenerate embedding
        embedding = self.embedding_engine.embed_document(doc)
        self.store.add_embedding(embedding)

        # Rediscover links
        self.store.delete_auto_links(doc.id)
        links = self.link_discovery.discover_links_for_document(doc.id)
        for link in links:
            self.store.add_link(link)

    def delete_document(self, doc_id: int):
        """Delete a document."""
        self.store.delete_document(doc_id)

    def list_documents(
        self,
        doc_type: Optional[DocumentType] = None,
        limit: Optional[int] = None
    ) -> list[Document]:
        """List all documents."""
        return self.store.get_all_documents(doc_type=doc_type, limit=limit)

    def get_links(self, doc_id: int, direction: str = "both") -> dict:
        """Get links for a document."""
        result = {
            "outgoing": [],
            "incoming": []
        }

        if direction in ("outgoing", "both"):
            links = self.store.get_links_from(doc_id)
            for link in links:
                doc = self.store.get_document(link.target_id)
                if doc:
                    result["outgoing"].append({
                        "link": link,
                        "document": doc
                    })

        if direction in ("incoming", "both"):
            links = self.store.get_links_to(doc_id)
            for link in links:
                doc = self.store.get_document(link.source_id)
                if doc:
                    result["incoming"].append({
                        "link": link,
                        "document": doc
                    })

        return result

    def get_graph(self, doc_id: int, depth: int = 2) -> dict:
        """Get a graph representation of links."""
        return self.link_discovery.get_link_graph(doc_id, depth=depth)

    def rebuild_embeddings(self):
        """Rebuild all embeddings."""
        print("Rebuilding embeddings for all documents...")
        docs = self.store.get_all_documents()

        embeddings = self.embedding_engine.embed_documents_batch(docs)

        for embedding in embeddings:
            self.store.add_embedding(embedding)

        print(f"Rebuilt embeddings for {len(docs)} documents")

    def rebuild_links(self):
        """Rebuild all automatic links."""
        print("Rebuilding automatic links...")
        self.link_discovery.discover_all_links(rebuild=True)

    def get_stats(self) -> dict:
        """Get knowledge base statistics."""
        stats = self.store.get_stats()

        # Add tag statistics
        tags = self.store.get_tags()
        stats["total_tags"] = len(tags)
        stats["top_tags"] = tags[:10]

        return stats

    def get_tags(self) -> list[tuple[str, int]]:
        """Get all tags with counts."""
        return self.store.get_tags()

    def search_by_tag(self, tag: str) -> list[Document]:
        """Find all documents with a specific tag."""
        all_docs = self.store.get_all_documents()
        return [doc for doc in all_docs if tag in doc.tags]

    def close(self):
        """Close the knowledge graph."""
        self.store.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
