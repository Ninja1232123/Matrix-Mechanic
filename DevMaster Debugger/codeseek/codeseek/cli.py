"""
Command-line interface for CodeSeek.
"""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from . import __version__
from .embeddings import CodeEmbedder
from .indexer import CodeIndexer
from .models import SymbolType
from .search import SearchEngine
from .storage import CodeStore

console = Console()


@click.group()
@click.version_option(version=__version__)
def main():
    """
    CodeSeek - Semantic code search engine.

    Search your codebase using natural language and semantic understanding.
    """
    pass


@main.command()
@click.argument('path', default='.')
@click.option('--exclude', help='Comma-separated exclusion patterns')
@click.option('--force', is_flag=True, help='Force reindex all files')
@click.option('--no-embeddings', is_flag=True, help='Skip generating embeddings')
@click.option('--db-path', default='~/.codeseek/index.db', help='Database path')
def index(path: str, exclude: str, force: bool, no_embeddings: bool, db_path: str):
    """Index a codebase for searching."""
    path = Path(path).resolve()

    if not path.exists():
        console.print(f"❌ Path not found: {path}", style="red")
        sys.exit(1)

    # Parse exclusions
    exclude_patterns = set(exclude.split(',')) if exclude else None

    # Initialize components
    with CodeStore(db_path) as store:
        indexer = CodeIndexer(store)

        # Index files
        stats = indexer.index_directory(str(path), exclude_patterns=exclude_patterns, force=force)

        # Generate embeddings if requested
        if not no_embeddings and stats['symbols_extracted'] > 0:
            console.print("\n🧠 Generating embeddings...")
            console.print("   (This may take a while on first run)\n")

            try:
                embedder = CodeEmbedder()
                symbols = store.get_all_symbols()

                if symbols:
                    embeddings = embedder.embed_symbols_batch(symbols)

                    console.print("\n💾 Saving embeddings...")
                    for embedding in embeddings:
                        store.add_embedding(embedding)

                    console.print(f"✅ Generated {len(embeddings)} embeddings")

            except Exception as e:
                console.print(f"⚠️  Failed to generate embeddings: {e}", style="yellow")
                console.print("   You can still use text-based search", style="dim")

        # Show final stats
        console.print("\n" + "="*60)
        index_stats = store.get_stats()
        _display_stats(index_stats)


@main.command()
@click.argument('query')
@click.option('--type', type=click.Choice(['function', 'class', 'method']), help='Filter by symbol type')
@click.option('--lang', '--language', help='Filter by language')
@click.option('--limit', default=10, help='Number of results')
@click.option('--no-semantic', is_flag=True, help='Disable semantic search')
@click.option('--context', default=3, help='Lines of context to show')
@click.option('--db-path', default='~/.codeseek/index.db')
def find(query: str, type: str, lang: str, limit: int, no_semantic: bool, context: int, db_path: str):
    """Search for code using natural language."""
    db_path_expanded = Path(db_path).expanduser()

    if not db_path_expanded.exists():
        console.print("❌ No index found. Run 'codeseek index' first.", style="red")
        sys.exit(1)

    with CodeStore(db_path) as store:
        # Initialize search engine
        embedder = None
        if not no_semantic:
            try:
                embedder = CodeEmbedder()
            except Exception as e:
                console.print(f"⚠️  Semantic search unavailable: {e}", style="yellow")

        engine = SearchEngine(store, embedder)

        # Parse symbol type
        symbol_type = SymbolType(type) if type else None

        # Search
        console.print(f"\n🔍 Searching for: [bold]{query}[/bold]\n")

        results = engine.search(
            query,
            top_k=limit,
            symbol_type=symbol_type,
            language=lang,
            use_semantic=(embedder is not None and not no_semantic)
        )

        if not results:
            console.print("No results found", style="yellow")
            return

        # Display results
        console.print(f"Found {len(results)} result(s):\n")

        for i, result in enumerate(results, 1):
            _display_result(result, i, context)


@main.command()
@click.argument('symbol_ref')  # file_path:symbol_name
@click.option('--limit', default=10, help='Number of results')
@click.option('--db-path', default='~/.codeseek/index.db')
def similar(symbol_ref: str, limit: int, db_path: str):
    """Find code similar to a given symbol."""
    db_path_expanded = Path(db_path).expanduser()

    if not db_path_expanded.exists():
        console.print("❌ No index found. Run 'codeseek index' first.", style="red")
        sys.exit(1)

    # Parse symbol reference
    if ':' in symbol_ref:
        file_path, symbol_name = symbol_ref.split(':', 1)
    else:
        console.print("❌ Format: file_path:symbol_name", style="red")
        sys.exit(1)

    with CodeStore(db_path) as store:
        # Find the symbol
        symbols = store.get_symbols_by_file(file_path)
        target_symbol = next((s for s in symbols if s.name == symbol_name), None)

        if not target_symbol:
            console.print(f"❌ Symbol '{symbol_name}' not found in {file_path}", style="red")
            sys.exit(1)

        # Initialize search engine
        try:
            embedder = CodeEmbedder()
        except Exception as e:
            console.print(f"❌ Embeddings required for similarity search: {e}", style="red")
            sys.exit(1)

        engine = SearchEngine(store, embedder)

        # Find similar
        console.print(f"\n🔍 Finding code similar to: [bold]{target_symbol.qualified_name}[/bold]\n")

        results = engine.find_similar(target_symbol.id, top_k=limit)

        if not results:
            console.print("No similar code found", style="yellow")
            return

        # Display results
        for i, result in enumerate(results, 1):
            _display_result(result, i, context=2)


@main.command()
@click.option('--db-path', default='~/.codeseek/index.db')
def stats(db_path: str):
    """Show index statistics."""
    db_path_expanded = Path(db_path).expanduser()

    if not db_path_expanded.exists():
        console.print("❌ No index found. Run 'codeseek index' first.", style="red")
        sys.exit(1)

    with CodeStore(db_path) as store:
        index_stats = store.get_stats()
        _display_stats(index_stats)


@main.command()
@click.option('--db-path', default='~/.codeseek/index.db')
@click.confirmation_option(prompt='Are you sure you want to clear the index?')
def clear(db_path: str):
    """Clear the code index."""
    with CodeStore(db_path) as store:
        store.clear_index()
        console.print("✅ Index cleared", style="green")


def _display_result(result, index: int, context: int = 3):
    """Display a single search result."""
    symbol = result.symbol
    score_pct = int(result.score * 100)

    # Title
    title = f"[{index}] {symbol.qualified_name}"
    subtitle = f"{result.match_type} | score: {score_pct}% | {symbol.file_path}:{symbol.start_line}"

    # Code preview
    syntax = Syntax(
        symbol.short_code,
        symbol.language.value,
        theme="monokai",
        line_numbers=True,
        start_line=symbol.start_line
    )

    panel = Panel(
        syntax,
        title=title,
        subtitle=subtitle,
        border_style="cyan" if result.score > 0.7 else "white"
    )

    console.print(panel)
    console.print()


def _display_stats(stats):
    """Display index statistics."""
    console.print("\n📊 [bold]Index Statistics[/bold]\n")

    # Main stats
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    table.add_row("Total Files", str(stats.total_files))
    table.add_row("Total Symbols", str(stats.total_symbols))
    table.add_row("Total Lines", f"{stats.total_lines:,}")
    table.add_row("Index Size", f"{stats.index_size_mb} MB")

    if stats.last_indexed:
        table.add_row("Last Indexed", stats.last_indexed.strftime("%Y-%m-%d %H:%M"))

    console.print(table)

    # Languages
    if stats.languages:
        console.print("\n[bold]Languages:[/bold]")
        for lang, count in sorted(stats.languages.items(), key=lambda x: x[1], reverse=True):
            console.print(f"  {lang}: {count}")

    # Symbol types
    if stats.symbol_types:
        console.print("\n[bold]Symbol Types:[/bold]")
        for sym_type, count in sorted(stats.symbol_types.items(), key=lambda x: x[1], reverse=True):
            console.print(f"  {sym_type}: {count}")


if __name__ == '__main__':
    main()
