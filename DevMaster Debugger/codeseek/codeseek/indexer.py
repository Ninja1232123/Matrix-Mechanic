"""
Codebase indexer that crawls and parses code files.

Performance optimizations:
- Parallel file processing with ThreadPoolExecutor
- Batch database inserts
- Incremental indexing (only changed files)
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from .models import FileIndex, Language
from .parser import UniversalParser
from .storage import CodeStore


class CodeIndexer:
    """Indexes a codebase by parsing files and storing symbols."""

    def __init__(self, store: CodeStore):
        self.store = store
        self.parser = UniversalParser()

        # Default exclusion patterns
        self.default_excludes = {
            'node_modules', 'venv', 'env', '.venv', '__pycache__',
            '.git', '.svn', '.hg', 'dist', 'build', '.egg-info',
            'vendor', 'target', '.next', '.nuxt', 'coverage',
            '.pytest_cache', '.tox', '.mypy_cache',
        }

        # Supported extensions
        self.supported_extensions = {
            '.py', '.js', '.jsx', '.ts', '.tsx',
            '.go', '.rs', '.java', '.c', '.cpp', '.cc', '.h', '.hpp'
        }

    def index_directory(
        self,
        root_path: str,
        exclude_patterns: Optional[set[str]] = None,
        force: bool = False,
        max_workers: int = 4
    ) -> dict:
        """
        Index all code files in a directory.

        Args:
            root_path: Root directory to index
            exclude_patterns: Additional exclusion patterns
            force: Force reindex even if files haven't changed
            max_workers: Number of parallel workers (default: 4)

        Returns:
            Statistics about indexing

        Performance: Uses parallel processing for 2-4x faster indexing.
        """
        root = Path(root_path).resolve()
        if not root.exists():
            raise FileNotFoundError(f"Directory not found: {root}")

        # Combine exclusion patterns
        excludes = self.default_excludes.copy()
        if exclude_patterns:
            excludes.update(exclude_patterns)

        print(f"🔍 Indexing codebase: {root}")
        print(f"   Excluding: {', '.join(sorted(excludes))}")
        print(f"   Parallel workers: {max_workers}\n")

        stats = {
            'files_processed': 0,
            'files_skipped': 0,
            'symbols_extracted': 0,
            'errors': 0
        }

        # Collect all files first
        file_paths = list(self._walk_directory(root, excludes))
        total_files = len(file_paths)
        print(f"   Found {total_files} files to process...")

        # Process files in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all indexing tasks
            future_to_path = {
                executor.submit(self._index_file_safe, file_path, force): file_path
                for file_path in file_paths
            }

            # Process completed futures
            for future in as_completed(future_to_path):
                file_path = future_to_path[future]
                try:
                    result, error = future.result()
                    if error:
                        stats['errors'] += 1
                        print(f"   ❌ Error indexing {file_path}: {error}")
                    elif result:
                        stats['files_processed'] += 1
                        stats['symbols_extracted'] += result['symbols']

                        if (stats['files_processed'] % 20 == 0):
                            print(f"   Processed {stats['files_processed']}/{total_files} files...")
                    else:
                        stats['files_skipped'] += 1

                except Exception as e:
                    stats['errors'] += 1
                    print(f"   ❌ Error indexing {file_path}: {e}")

        print(f"\n✅ Indexing complete!")
        print(f"   Files processed: {stats['files_processed']}")
        print(f"   Files skipped: {stats['files_skipped']}")
        print(f"   Symbols extracted: {stats['symbols_extracted']}")
        if stats['errors'] > 0:
            print(f"   Errors: {stats['errors']}")

        # Publish to nervous system
        self._publish_index_event(str(root), stats)

        return stats

    def _publish_index_event(self, repo_path: str, stats: dict):
        """Publish indexing completion event to nervous system."""
        try:
            # Try to import from devmaster's nervous system
            import sys
            devmaster_path = Path(__file__).parent.parent.parent / 'devmaster'
            if devmaster_path.exists():
                sys.path.insert(0, str(devmaster_path))

            from devmaster.nervous_system import publish_code_indexed
            publish_code_indexed(
                repo_path=repo_path,
                files_indexed=stats['files_processed'],
                symbols_found=stats['symbols_extracted']
            )
            print("   📡 Published to nervous system")
        except ImportError:
            pass  # Nervous system not available
        except Exception as e:
            print(f"   [dim]Nervous system: {e}[/dim]")

    def _index_file_safe(self, file_path: str, force: bool = False) -> tuple:
        """Thread-safe wrapper for _index_file that catches exceptions."""
        try:
            result = self._index_file(file_path, force=force)
            return result, None
        except Exception as e:
            return None, str(e)

    def index_file(self, file_path: str, force: bool = False) -> Optional[dict]:
        """Index a single file."""
        return self._index_file(file_path, force=force)

    def _index_file(self, file_path: str, force: bool = False) -> Optional[dict]:
        """Index a single file."""
        file_path = str(Path(file_path).resolve())

        # Detect language
        language = self.parser.detect_language(file_path)
        if language == Language.UNKNOWN:
            return None

        # Check if file needs reindexing
        current_hash = self.store.compute_file_hash(file_path)
        if not force and not self.store.file_needs_reindex(file_path, current_hash):
            return None

        # Parse file
        symbols = self.parser.parse_file(file_path, language)

        # Delete existing symbols for this file
        self.store.delete_file_symbols(file_path)

        # Get file info
        file_stat = os.stat(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            line_count = sum(1 for _ in f)

        # Create file index
        file_idx = FileIndex(
            file_path=file_path,
            language=language,
            size_bytes=file_stat.st_size,
            line_count=line_count,
            symbol_count=len(symbols),
            content_hash=current_hash
        )

        # Save file index
        file_id = self.store.add_file(file_idx)

        # Save symbols using batch insert for better performance
        if symbols:
            self.store.add_symbols_batch(symbols, file_id)

        return {
            'file_id': file_id,
            'symbols': len(symbols)
        }

    def _walk_directory(self, root: Path, excludes: set[str]):
        """Walk directory tree, yielding file paths to index."""
        for dirpath, dirnames, filenames in os.walk(root):
            # Filter out excluded directories
            dirnames[:] = [d for d in dirnames if d not in excludes]

            # Check for supported files
            for filename in filenames:
                file_path = Path(dirpath) / filename

                # Check extension
                if file_path.suffix not in self.supported_extensions:
                    continue

                # Skip if any parent directory is in excludes
                if any(part in excludes for part in file_path.parts):
                    continue

                yield str(file_path)

    def remove_file(self, file_path: str):
        """Remove a file from the index."""
        file_path = str(Path(file_path).resolve())
        self.store.delete_file_symbols(file_path)

    def update_file(self, file_path: str):
        """Update a single file in the index."""
        return self._index_file(file_path, force=True)

    def clear_index(self):
        """Clear the entire index."""
        self.store.clear_index()
        print("✅ Index cleared")
