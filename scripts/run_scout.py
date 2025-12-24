#!/usr/bin/env python3
"""
Safe Scout - Standalone Web Documentation Ingestion.

This script runs the Safe Scout module independently for web documentation
ingestion, URL validation, and knowledge synthesis.

Usage:
    python scripts/run_scout.py [options]

Examples:
    # Process pending topics
    python scripts/run_scout.py

    # Add a topic
    python scripts/run_scout.py --add-topic "Python asyncio" --add-url "https://docs.python.org/3/library/asyncio.html"

    # Run with specific concurrency
    python scripts/run_scout.py --concurrency 10 --pages 20

    # Validate URLs only (no fetching)
    python scripts/run_scout.py --validate-only --url "https://example.com"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List

# Add parent directory to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("scout")

console = Console()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Safe Scout - Web Documentation Ingestion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # General options
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path(os.getenv("NIGHTSHIFT_WORK_DIR", Path.home() / ".jarvis" / "nightshift")),
        help="Working directory",
    )

    # Topic management
    parser.add_argument(
        "--add-topic",
        metavar="NAME",
        help="Add a learning topic to the queue",
    )
    parser.add_argument(
        "--add-url",
        metavar="URL",
        action="append",
        default=[],
        help="Add URL(s) to the topic (can be repeated)",
    )
    parser.add_argument(
        "--priority",
        choices=["critical", "high", "normal", "low", "background"],
        default="normal",
        help="Topic priority",
    )
    parser.add_argument(
        "--category",
        choices=["documentation", "tutorial", "api_reference", "release_notes", "blog", "paper", "other"],
        default="documentation",
        help="Topic category",
    )

    # Queue management
    parser.add_argument(
        "--list-topics",
        action="store_true",
        help="List all topics in queue",
    )
    parser.add_argument(
        "--clear-completed",
        action="store_true",
        help="Clear completed topics from queue",
    )
    parser.add_argument(
        "--reset-failed",
        action="store_true",
        help="Reset failed topics to pending",
    )

    # Processing options
    parser.add_argument(
        "--topics",
        type=int,
        default=int(os.getenv("NIGHTSHIFT_SCOUT_MAX_TOPICS", "10")),
        help="Maximum topics to process",
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=int(os.getenv("NIGHTSHIFT_SCOUT_MAX_PAGES", "5")),
        help="Maximum pages per topic",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.getenv("NIGHTSHIFT_SCOUT_CONCURRENCY", "3")),
        help="Concurrent page fetches",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Page fetch timeout in seconds",
    )

    # Sandbox options
    parser.add_argument(
        "--no-docker",
        action="store_true",
        help="Disable Docker sandbox (use local Playwright)",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Use direct execution (no sandbox - dev only)",
    )

    # Validation options
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate URLs, don't fetch content",
    )
    parser.add_argument(
        "--url",
        metavar="URL",
        action="append",
        default=[],
        help="URL(s) to validate (with --validate-only)",
    )
    parser.add_argument(
        "--check-robots",
        action="store_true",
        default=True,
        help="Check robots.txt (default: True)",
    )
    parser.add_argument(
        "--no-robots",
        action="store_true",
        help="Skip robots.txt checking",
    )

    # Synthesis options
    parser.add_argument(
        "--synthesis-model",
        default=os.getenv("NIGHTSHIFT_SCOUT_MODEL", "gemini-1.5-flash"),
        help="Model for knowledge synthesis",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=5,
        help="Maximum Q&A pairs per page",
    )
    parser.add_argument(
        "--skip-synthesis",
        action="store_true",
        help="Skip knowledge synthesis (extraction only)",
    )

    # Output options
    parser.add_argument(
        "--output-format",
        choices=["json", "jsonl", "chatml"],
        default="json",
        help="Output format for synthesized pairs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Custom output directory for synthesized pairs",
    )

    return parser.parse_args()


async def add_topic(args: argparse.Namespace) -> int:
    """Add a topic to the queue."""
    from reactor_core.scout import (
        TopicQueue,
        TopicQueueConfig,
        LearningTopic,
        TopicPriority,
        TopicCategory,
    )

    queue_config = TopicQueueConfig(db_path=args.work_dir / "scout_queue.db")
    queue = TopicQueue(queue_config)

    priority_map = {
        "critical": TopicPriority.CRITICAL,
        "high": TopicPriority.HIGH,
        "normal": TopicPriority.NORMAL,
        "low": TopicPriority.LOW,
        "background": TopicPriority.BACKGROUND,
    }

    category_map = {
        "documentation": TopicCategory.DOCUMENTATION,
        "tutorial": TopicCategory.TUTORIAL,
        "api_reference": TopicCategory.API_REFERENCE,
        "release_notes": TopicCategory.RELEASE_NOTES,
        "blog": TopicCategory.BLOG,
        "paper": TopicCategory.PAPER,
        "other": TopicCategory.OTHER,
    }

    topic = LearningTopic(
        name=args.add_topic,
        description=f"Topic: {args.add_topic}",
        urls=args.add_url,
        priority=priority_map[args.priority],
        category=category_map[args.category],
    )

    await queue.add_topic(topic)
    await queue.close()

    console.print(f"[green]Added topic:[/green] {args.add_topic}")
    console.print(f"  Priority: {args.priority}")
    console.print(f"  Category: {args.category}")
    console.print(f"  URLs: {len(args.add_url)}")
    for url in args.add_url:
        console.print(f"    - {url}")

    return 0


async def list_topics(args: argparse.Namespace) -> int:
    """List topics in the queue."""
    from reactor_core.scout import TopicQueue, TopicQueueConfig, TopicStatus

    queue_config = TopicQueueConfig(db_path=args.work_dir / "scout_queue.db")
    queue = TopicQueue(queue_config)

    # Get all topics
    pending = await queue.get_pending_topics(limit=1000)
    all_topics = pending  # Would need additional methods for other statuses

    await queue.close()

    if not all_topics:
        console.print("[yellow]No topics in queue[/yellow]")
        return 0

    table = Table(title="Topic Queue")
    table.add_column("ID", style="dim")
    table.add_column("Name", style="cyan")
    table.add_column("Priority", style="yellow")
    table.add_column("Status", style="green")
    table.add_column("URLs", style="blue")

    for topic in all_topics:
        table.add_row(
            topic.topic_id[:8],
            topic.name[:40],
            topic.priority.name,
            topic.status.name,
            str(len(topic.urls)),
        )

    console.print(table)
    console.print(f"\nTotal: {len(all_topics)} topics")

    return 0


async def validate_urls(args: argparse.Namespace) -> int:
    """Validate URLs without fetching."""
    from reactor_core.scout import URLValidator, URLValidatorConfig

    if not args.url:
        console.print("[red]No URLs specified. Use --url to add URLs[/red]")
        return 1

    config = URLValidatorConfig(
        check_robots_txt=not args.no_robots,
        check_safe_browsing=False,
    )
    validator = URLValidator(config)

    table = Table(title="URL Validation Results")
    table.add_column("URL", style="cyan", max_width=60)
    table.add_column("Safe", style="green")
    table.add_column("Level", style="yellow")
    table.add_column("Reason", style="red")

    results = await validator.validate_batch(args.url)

    for url, result in zip(args.url, results):
        table.add_row(
            url[:60],
            "Yes" if result.is_safe else "No",
            result.safety_level.name,
            result.block_reason.name if result.block_reason else "-",
        )

    console.print(table)
    return 0


async def run_scout(args: argparse.Namespace) -> int:
    """Run the Scout ingestion process."""
    from reactor_core.scout import (
        TopicQueue,
        TopicQueueConfig,
        URLValidator,
        URLValidatorConfig,
        ComplianceFilter,
        SandboxExecutor,
        SandboxConfig,
        ExecutionMode,
        ContentExtractor,
        KnowledgeSynthesizer,
    )
    from reactor_core.distillation import create_teacher_client

    console.rule("[bold green]Safe Scout - Web Documentation Ingestion[/bold green]")
    console.print(f"Started: {datetime.now().isoformat()}")
    console.print(f"Work directory: {args.work_dir}")
    console.print()

    args.work_dir.mkdir(parents=True, exist_ok=True)

    # Initialize queue
    queue_config = TopicQueueConfig(db_path=args.work_dir / "scout_queue.db")
    queue = TopicQueue(queue_config)

    # Get pending topics
    topics = await queue.get_pending_topics(limit=args.topics)

    if not topics:
        console.print("[yellow]No pending topics in queue[/yellow]")
        console.print("Add topics with: python scripts/run_scout.py --add-topic 'Topic Name' --add-url 'URL'")
        await queue.close()
        return 0

    console.print(f"Found {len(topics)} pending topics")

    # Initialize components
    validator_config = URLValidatorConfig(
        check_robots_txt=not args.no_robots,
        check_safe_browsing=False,
    )
    validator = URLValidator(validator_config)
    compliance = ComplianceFilter()

    if args.direct:
        exec_mode = ExecutionMode.DIRECT
    elif args.no_docker:
        exec_mode = ExecutionMode.SUBPROCESS
    else:
        exec_mode = ExecutionMode.DOCKER

    sandbox_config = SandboxConfig(
        mode=exec_mode,
        timeout_seconds=args.timeout,
        max_concurrent=args.concurrency,
    )
    sandbox = SandboxExecutor(sandbox_config)
    extractor = ContentExtractor()

    synthesizer = None
    if not args.skip_synthesis:
        try:
            teacher = create_teacher_client(args.synthesis_model)
            synthesizer = KnowledgeSynthesizer(teacher)
        except Exception as e:
            logger.warning(f"Could not initialize synthesizer: {e}")
            console.print(f"[yellow]Synthesis disabled: {e}[/yellow]")

    # Output directory
    output_dir = args.output_dir or (args.work_dir / "scout_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Statistics
    stats = {
        "topics_processed": 0,
        "pages_fetched": 0,
        "pages_blocked": 0,
        "pages_failed": 0,
        "examples_synthesized": 0,
        "start_time": datetime.now(),
    }

    # Process topics with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        overall_task = progress.add_task("Processing topics...", total=len(topics))

        for topic in topics:
            progress.update(overall_task, description=f"[cyan]{topic.name[:30]}[/cyan]")
            await queue.mark_processing(topic.topic_id)
            stats["topics_processed"] += 1

            urls = topic.urls[:args.pages]
            page_task = progress.add_task(f"  Pages", total=len(urls), visible=True)

            for url in urls:
                try:
                    # Validate
                    validation = await validator.validate(url)
                    if not validation.is_safe:
                        stats["pages_blocked"] += 1
                        logger.debug(f"Blocked: {url} - {validation.block_reason}")
                        progress.advance(page_task)
                        continue

                    # Fetch via sandbox
                    result = await sandbox.execute(url)
                    if not result.success:
                        stats["pages_failed"] += 1
                        logger.debug(f"Failed: {url} - {result.error}")
                        progress.advance(page_task)
                        continue

                    # Compliance check
                    comp_result = compliance.check_compliance(result.html_content or "", url)
                    if not comp_result.is_compliant:
                        stats["pages_blocked"] += 1
                        logger.debug(f"Compliance block: {url}")
                        progress.advance(page_task)
                        continue

                    stats["pages_fetched"] += 1

                    # Extract content
                    content = extractor.extract(result.html_content or "", url)

                    if not content.main_content:
                        progress.advance(page_task)
                        continue

                    # Synthesize if enabled
                    if synthesizer and content.main_content:
                        try:
                            syn_result = await synthesizer.synthesize(
                                content=content.main_content,
                                title=content.title or topic.name,
                                code_blocks=content.code_blocks,
                                max_pairs=args.max_pairs,
                            )

                            stats["examples_synthesized"] += len(syn_result.pairs)

                            # Save pairs
                            for pair in syn_result.pairs:
                                pair_file = output_dir / f"{pair.pair_id}.json"
                                with open(pair_file, "w") as f:
                                    json.dump(pair.to_dict(), f, indent=2)

                        except Exception as e:
                            logger.warning(f"Synthesis error for {url}: {e}")

                except Exception as e:
                    stats["pages_failed"] += 1
                    logger.warning(f"Error processing {url}: {e}")

                progress.advance(page_task)

            await queue.mark_completed(topic.topic_id)
            progress.update(page_task, visible=False)
            progress.advance(overall_task)

    # Cleanup
    await sandbox.cleanup()
    await queue.close()

    # Show results
    stats["end_time"] = datetime.now()
    duration = (stats["end_time"] - stats["start_time"]).total_seconds()

    console.print()
    console.rule("[bold]Scout Complete[/bold]")

    results_table = Table(title="Results")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="green")
    results_table.add_row("Topics Processed", str(stats["topics_processed"]))
    results_table.add_row("Pages Fetched", str(stats["pages_fetched"]))
    results_table.add_row("Pages Blocked", str(stats["pages_blocked"]))
    results_table.add_row("Pages Failed", str(stats["pages_failed"]))
    results_table.add_row("Examples Synthesized", str(stats["examples_synthesized"]))
    results_table.add_row("Duration", f"{duration:.1f}s")
    results_table.add_row("Output Directory", str(output_dir))

    console.print(results_table)

    return 0


async def main_async(args: argparse.Namespace) -> int:
    """Async main entry point."""
    args.work_dir.mkdir(parents=True, exist_ok=True)

    if args.add_topic:
        return await add_topic(args)

    if args.list_topics:
        return await list_topics(args)

    if args.validate_only:
        return await validate_urls(args)

    return await run_scout(args)


def main() -> int:
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger("reactor_core").setLevel(logging.DEBUG)
        logging.getLogger("scout").setLevel(logging.DEBUG)

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
