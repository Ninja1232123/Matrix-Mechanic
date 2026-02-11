"""
Automatic link discovery between documents.
"""

from typing import Optional

from .embeddings import VectorSearch
from .models import Document, Link
from .storage import KnowledgeStore


class LinkDiscovery:
    """Discovers and manages links between documents."""

    def __init__(
        self,
        store: KnowledgeStore,
        similarity_threshold: float = 0.5,
        max_links_per_doc: int = 10
    ):
        self.store = store
        self.vector_search = VectorSearch(store)
        self.similarity_threshold = similarity_threshold
        self.max_links_per_doc = max_links_per_doc

    def discover_links_for_document(self, doc_id: int) -> list[Link]:
        """Discover links for a single document."""
        doc = self.store.get_document(doc_id)
        if not doc:
            return []

        links = []

        # Vector similarity-based links
        similar_docs = self.vector_search.find_similar(
            doc_id,
            top_k=self.max_links_per_doc,
            threshold=self.similarity_threshold
        )

        for result in similar_docs:
            link = Link(
                source_id=doc_id,
                target_id=result.document.id,
                link_type="similar",
                strength=result.score,
                auto_generated=True
            )
            links.append(link)

        # Tag-based links
        tag_links = self._find_tag_based_links(doc)
        links.extend(tag_links)

        # Code dependency links
        if doc.imports:
            dep_links = self._find_dependency_links(doc)
            links.extend(dep_links)

        # Deduplicate and sort by strength
        unique_links = self._deduplicate_links(links)
        unique_links.sort(key=lambda x: x.strength, reverse=True)

        return unique_links[:self.max_links_per_doc]

    def discover_all_links(self, rebuild: bool = False):
        """Discover links for all documents in the knowledge base."""
        if rebuild:
            # Delete existing auto-generated links
            self.store.delete_auto_links()

        # Get all documents
        all_docs = self.store.get_all_documents()

        print(f"Discovering links for {len(all_docs)} documents...")

        for i, doc in enumerate(all_docs):
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(all_docs)} documents")

            links = self.discover_links_for_document(doc.id)

            # Save links to database
            for link in links:
                self.store.add_link(link)

        print(f"Link discovery complete!")

    def _find_tag_based_links(self, doc: Document) -> list[Link]:
        """Find documents with shared tags."""
        if not doc.tags:
            return []

        links = []
        all_docs = self.store.get_all_documents()

        for other_doc in all_docs:
            if other_doc.id == doc.id:
                continue

            if not other_doc.tags:
                continue

            # Calculate tag overlap
            shared_tags = set(doc.tags) & set(other_doc.tags)
            if not shared_tags:
                continue

            # Jaccard similarity for tags
            all_tags = set(doc.tags) | set(other_doc.tags)
            similarity = len(shared_tags) / len(all_tags)

            # Only create link if similarity is significant
            if similarity >= 0.3:
                link = Link(
                    source_id=doc.id,
                    target_id=other_doc.id,
                    link_type="tag_similarity",
                    strength=similarity,
                    auto_generated=True
                )
                links.append(link)

        return links

    def _find_dependency_links(self, doc: Document) -> list[Link]:
        """Find documents based on code dependencies."""
        if not doc.imports:
            return []

        links = []
        all_docs = self.store.get_all_documents()

        for other_doc in all_docs:
            if other_doc.id == doc.id:
                continue

            # Check if other doc defines symbols we import
            if other_doc.symbols:
                shared_symbols = set(doc.imports) & set(other_doc.symbols)
                if shared_symbols:
                    link = Link(
                        source_id=doc.id,
                        target_id=other_doc.id,
                        link_type="imports",
                        strength=0.9,  # High confidence for code dependencies
                        auto_generated=True
                    )
                    links.append(link)

            # Check if we share imports (related implementations)
            if other_doc.imports:
                shared_imports = set(doc.imports) & set(other_doc.imports)
                if len(shared_imports) >= 2:  # At least 2 shared imports
                    similarity = len(shared_imports) / max(len(doc.imports), len(other_doc.imports))
                    link = Link(
                        source_id=doc.id,
                        target_id=other_doc.id,
                        link_type="shared_dependencies",
                        strength=similarity * 0.7,  # Medium confidence
                        auto_generated=True
                    )
                    links.append(link)

        return links

    def _deduplicate_links(self, links: list[Link]) -> list[Link]:
        """Remove duplicate links, keeping the highest strength."""
        link_map = {}

        for link in links:
            key = (link.source_id, link.target_id)

            if key not in link_map or link.strength > link_map[key].strength:
                link_map[key] = link

        return list(link_map.values())

    def get_link_graph(self, doc_id: int, depth: int = 2) -> dict:
        """Get a graph of links starting from a document."""
        visited = set()
        graph = {"nodes": [], "links": []}

        def traverse(current_id: int, current_depth: int):
            if current_depth > depth or current_id in visited:
                return

            visited.add(current_id)

            # Get document
            doc = self.store.get_document(current_id)
            if not doc:
                return

            graph["nodes"].append({
                "id": doc.id,
                "title": doc.title,
                "type": doc.doc_type.value,
                "depth": current_depth
            })

            # Get outgoing links
            links = self.store.get_links_from(current_id)

            for link in links:
                graph["links"].append({
                    "source": link.source_id,
                    "target": link.target_id,
                    "type": link.link_type,
                    "strength": link.strength
                })

                # Traverse to linked documents
                traverse(link.target_id, current_depth + 1)

        traverse(doc_id, 0)
        return graph
