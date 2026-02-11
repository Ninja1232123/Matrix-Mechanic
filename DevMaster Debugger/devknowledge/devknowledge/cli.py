"""
Command-line interface for DevKnowledge.
"""

import sys
from pathlib import Path

import click
from rich import print as rprint
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from . import __version__
from .knowledge import KnowledgeGraph
from .models import DocumentType, Language


console = Console()


@click.group()
@click.version_option(version=__version__)
def main():
    """
    DevKnowledge - Local-first knowledge graph for developers.

    Store notes, code snippets, and documentation with automatic semantic linking.
    """
    pass


@main.command()
@click.option('--db-path', default='~/.devknowledge/kb.db', help='Database path')
def init(db_path: str):
    """Initialize a new knowledge base."""
    db_path_expanded = Path(db_path).expanduser()

    if db_path_expanded.exists():
        if not click.confirm(f"Knowledge base already exists at {db_path_expanded}. Reinitialize?"):
            return

    with KnowledgeGraph(db_path) as kg:
        stats = kg.get_stats()

    console.print(f"✅ Knowledge base initialized at {db_path_expanded}", style="green bold")
    console.print(f"   Documents: {stats['total_documents']}", style="dim")


@main.group()
def add():
    """Add content to the knowledge base."""
    pass


@add.command('note')
@click.argument('title')
@click.option('--content', help='Note content')
@click.option('--tags', help='Comma-separated tags')
@click.option('--db-path', default='~/.devknowledge/kb.db')
def add_note(title: str, content: str, tags: str, db_path: str):
    """Add a new note."""
    if not content:
        content = click.edit() or ""

    tag_list = [t.strip() for t in tags.split(',')] if tags else []

    with KnowledgeGraph(db_path) as kg:
        doc = kg.add_note(title, content, tags=tag_list)

    console.print(f"✅ Added note: {doc.title}", style="green bold")
    console.print(f"   ID: {doc.id}", style="dim")
    console.print(f"   Tags: {', '.join(doc.tags)}", style="dim")


@add.command('code')
@click.argument('language')
@click.argument('title')
@click.option('--file', type=click.File('r'), help='Read code from file')
@click.option('--content', help='Code content')
@click.option('--tags', help='Comma-separated tags')
@click.option('--db-path', default='~/.devknowledge/kb.db')
def add_code(language: str, title: str, file, content: str, tags: str, db_path: str):
    """Add a code snippet."""
    if file:
        content = file.read()
    elif not content:
        # Read from stdin
        if not sys.stdin.isatty():
            content = sys.stdin.read()
        else:
            content = click.edit() or ""

    tag_list = [t.strip() for t in tags.split(',')] if tags else []

    # Parse language
    try:
        lang = Language(language.lower())
    except ValueError:
        lang = None

    with KnowledgeGraph(db_path) as kg:
        doc = kg.add_code(title, content, language=lang, tags=tag_list)

    console.print(f"✅ Added code: {doc.title}", style="green bold")
    console.print(f"   ID: {doc.id}", style="dim")
    console.print(f"   Language: {doc.language.value}", style="dim")
    console.print(f"   Symbols: {', '.join(doc.symbols[:5])}", style="dim")
    console.print(f"   Tags: {', '.join(doc.tags)}", style="dim")


@add.command('doc')
@click.argument('title')
@click.option('--content', help='Documentation content')
@click.option('--url', help='Documentation URL')
@click.option('--tags', help='Comma-separated tags')
@click.option('--db-path', default='~/.devknowledge/kb.db')
def add_doc(title: str, content: str, url: str, tags: str, db_path: str):
    """Add documentation."""
    if not content:
        content = click.edit() or ""

    tag_list = [t.strip() for t in tags.split(',')] if tags else []

    with KnowledgeGraph(db_path) as kg:
        doc = kg.add_documentation(title, content, url=url, tags=tag_list)

    console.print(f"✅ Added documentation: {doc.title}", style="green bold")
    console.print(f"   ID: {doc.id}", style="dim")
    if url:
        console.print(f"   URL: {url}", style="dim")


@main.command()
@click.argument('query')
@click.option('--type', type=click.Choice(['note', 'code', 'documentation']), help='Filter by type')
@click.option('--limit', default=10, help='Number of results')
@click.option('--db-path', default='~/.devknowledge/kb.db')
def search(query: str, type: str, limit: int, db_path: str):
    """Search the knowledge base."""
    doc_type = DocumentType(type) if type else None

    with KnowledgeGraph(db_path) as kg:
        results = kg.search(query, top_k=limit, doc_type=doc_type)

    if not results:
        console.print("No results found", style="yellow")
        return

    console.print(f"\n🔍 Found {len(results)} result(s) for: {query}\n", style="bold")

    for i, result in enumerate(results, 1):
        doc = result.document

        panel_content = f"[bold]{doc.title}[/bold]\n"
        panel_content += f"[dim]ID: {doc.id} | Type: {doc.doc_type.value} | Score: {result.score:.3f}[/dim]\n\n"

        # Preview content
        preview = doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
        panel_content += preview

        if doc.tags:
            panel_content += f"\n\n[dim]Tags: {', '.join(doc.tags)}[/dim]"

        console.print(Panel(panel_content, title=f"Result {i}"))


@main.command()
@click.argument('doc_id', type=int)
@click.option('--db-path', default='~/.devknowledge/kb.db')
def show(doc_id: int, db_path: str):
    """Show a document by ID."""
    with KnowledgeGraph(db_path) as kg:
        doc = kg.get_document(doc_id)

    if not doc:
        console.print(f"Document {doc_id} not found", style="red")
        return

    # Display document
    console.print(f"\n[bold cyan]{doc.title}[/bold cyan]")
    console.print(f"[dim]ID: {doc.id} | Type: {doc.doc_type.value} | Language: {doc.language.value}[/dim]\n")

    # Content with syntax highlighting for code
    if doc.doc_type in (DocumentType.CODE, DocumentType.SNIPPET):
        syntax = Syntax(doc.content, doc.language.value, theme="monokai", line_numbers=True)
        console.print(syntax)
    elif doc.language == Language.MARKDOWN:
        md = Markdown(doc.content)
        console.print(md)
    else:
        console.print(doc.content)

    # Metadata
    console.print("\n[bold]Metadata:[/bold]")
    if doc.tags:
        console.print(f"  Tags: {', '.join(doc.tags)}")
    if doc.symbols:
        console.print(f"  Symbols: {', '.join(doc.symbols[:10])}")
    if doc.imports:
        console.print(f"  Imports: {', '.join(doc.imports[:10])}")

    console.print(f"  Created: {doc.created_at.strftime('%Y-%m-%d %H:%M')}")
    console.print(f"  Updated: {doc.updated_at.strftime('%Y-%m-%d %H:%M')}")


@main.command()
@click.argument('doc_id', type=int)
@click.option('--limit', default=10, help='Number of results')
@click.option('--db-path', default='~/.devknowledge/kb.db')
def related(doc_id: int, limit: int, db_path: str):
    """Find documents related to a given document."""
    with KnowledgeGraph(db_path) as kg:
        doc = kg.get_document(doc_id)
        if not doc:
            console.print(f"Document {doc_id} not found", style="red")
            return

        results = kg.find_related(doc_id, top_k=limit)

    console.print(f"\n🔗 Related to: [bold]{doc.title}[/bold]\n")

    for i, result in enumerate(results, 1):
        related_doc = result.document
        console.print(f"{i}. {related_doc.title}")
        console.print(f"   [dim]ID: {related_doc.id} | Score: {result.score:.3f} | Type: {related_doc.doc_type.value}[/dim]")


@main.command()
@click.argument('doc_id', type=int)
@click.option('--db-path', default='~/.devknowledge/kb.db')
def links(doc_id: int, db_path: str):
    """Show incoming and outgoing links for a document."""
    with KnowledgeGraph(db_path) as kg:
        doc = kg.get_document(doc_id)
        if not doc:
            console.print(f"Document {doc_id} not found", style="red")
            return

        link_info = kg.get_links(doc_id)

    console.print(f"\n🔗 Links for: [bold]{doc.title}[/bold]\n")

    # Outgoing links
    console.print("[bold cyan]Outgoing Links:[/bold cyan]")
    if link_info["outgoing"]:
        for item in link_info["outgoing"]:
            link = item["link"]
            target = item["document"]
            console.print(f"  → {target.title}")
            console.print(f"    [dim]{link.link_type} (strength: {link.strength:.3f})[/dim]")
    else:
        console.print("  [dim]None[/dim]")

    # Incoming links
    console.print(f"\n[bold green]Incoming Links:[/bold green]")
    if link_info["incoming"]:
        for item in link_info["incoming"]:
            link = item["link"]
            source = item["document"]
            console.print(f"  ← {source.title}")
            console.print(f"    [dim]{link.link_type} (strength: {link.strength:.3f})[/dim]")
    else:
        console.print("  [dim]None[/dim]")


@main.command()
@click.option('--db-path', default='~/.devknowledge/kb.db')
def stats(db_path: str):
    """Show knowledge base statistics."""
    with KnowledgeGraph(db_path) as kg:
        stats = kg.get_stats()

    console.print("\n📊 [bold]Knowledge Base Statistics[/bold]\n")

    # Main stats
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    table.add_row("Total Documents", str(stats["total_documents"]))
    table.add_row("Total Links", str(stats["total_links"]))
    table.add_row("Total Tags", str(stats["total_tags"]))

    console.print(table)

    # Documents by type
    if stats["documents_by_type"]:
        console.print("\n[bold]Documents by Type:[/bold]")
        for doc_type, count in stats["documents_by_type"].items():
            console.print(f"  {doc_type}: {count}")

    # Top tags
    if stats["top_tags"]:
        console.print("\n[bold]Top Tags:[/bold]")
        for tag, count in stats["top_tags"]:
            console.print(f"  {tag}: {count}")


@main.command()
@click.option('--type', type=click.Choice(['note', 'code', 'documentation']), help='Filter by type')
@click.option('--limit', default=20, help='Number of documents to show')
@click.option('--db-path', default='~/.devknowledge/kb.db')
def list(type: str, limit: int, db_path: str):
    """List all documents."""
    doc_type = DocumentType(type) if type else None

    with KnowledgeGraph(db_path) as kg:
        docs = kg.list_documents(doc_type=doc_type, limit=limit)

    if not docs:
        console.print("No documents found", style="yellow")
        return

    console.print(f"\n📚 [bold]{len(docs)} Document(s)[/bold]\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("ID", justify="right")
    table.add_column("Title")
    table.add_column("Type")
    table.add_column("Tags")
    table.add_column("Updated")

    for doc in docs:
        tags_str = ', '.join(doc.tags[:3])
        if len(doc.tags) > 3:
            tags_str += "..."

        table.add_row(
            str(doc.id),
            doc.title[:40],
            doc.doc_type.value,
            tags_str,
            doc.updated_at.strftime('%Y-%m-%d')
        )

    console.print(table)


@main.command()
@click.option('--db-path', default='~/.devknowledge/kb.db')
def rebuild(db_path: str):
    """Rebuild embeddings and links."""
    with KnowledgeGraph(db_path) as kg:
        console.print("🔄 Rebuilding embeddings...", style="bold")
        kg.rebuild_embeddings()

        console.print("\n🔄 Rebuilding links...", style="bold")
        kg.rebuild_links()

    console.print("\n✅ Rebuild complete!", style="green bold")


if __name__ == '__main__':
    main()
