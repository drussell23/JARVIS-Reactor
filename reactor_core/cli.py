"""
Night Shift Training Engine - CLI Entry Points.

Provides:
- nightshift: Main pipeline orchestration
- nightshift-scout: Safe Scout web ingestion
- nightshift-ingest: JARVIS experience ingestion
- nightshift-train: Model training
- nightshift-eval: Model evaluation
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("nightshift")


def setup_logging(verbose: bool = False) -> None:
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger("nightshift").setLevel(level)
    logging.getLogger("reactor_core").setLevel(level)


def main() -> int:
    """Main entry point for the Night Shift CLI."""
    parser = argparse.ArgumentParser(
        prog="nightshift",
        description="Night Shift Training Engine - Autonomous Continuous Learning Pipeline",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version and exit",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Pipeline command
    pipeline_parser = subparsers.add_parser("run", help="Run the full training pipeline")
    pipeline_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous checkpoint",
    )
    pipeline_parser.add_argument(
        "--stop-after",
        choices=["scouting", "ingesting", "formatting", "distilling", "training", "evaluating", "quantizing"],
        help="Stop after this stage",
    )
    pipeline_parser.add_argument(
        "--skip",
        nargs="+",
        choices=["scouting", "ingesting", "formatting", "distilling", "training", "evaluating", "quantizing"],
        help="Skip these stages",
    )
    pipeline_parser.add_argument(
        "--work-dir",
        type=Path,
        help="Working directory for pipeline outputs",
    )
    pipeline_parser.add_argument(
        "--sources",
        nargs="+",
        choices=["scout", "jarvis", "corrections"],
        default=["scout", "jarvis"],
        help="Data sources to enable",
    )

    # Scout command
    scout_parser = subparsers.add_parser("scout", help="Run Safe Scout web ingestion only")
    scout_parser.add_argument(
        "--topics",
        type=int,
        default=10,
        help="Maximum topics to process",
    )
    scout_parser.add_argument(
        "--pages-per-topic",
        type=int,
        default=5,
        help="Maximum pages per topic",
    )
    scout_parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Number of concurrent page fetches",
    )
    scout_parser.add_argument(
        "--no-docker",
        action="store_true",
        help="Disable Docker sandbox (use local Playwright)",
    )
    scout_parser.add_argument(
        "--add-topic",
        metavar="NAME",
        help="Add a topic to the queue",
    )
    scout_parser.add_argument(
        "--add-url",
        metavar="URL",
        help="Add a URL to the topic (requires --add-topic)",
    )

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest JARVIS experience logs")
    ingest_parser.add_argument(
        "--jarvis-path",
        type=Path,
        help="Path to JARVIS-AI-Agent repo",
    )
    ingest_parser.add_argument(
        "--lookback-hours",
        type=int,
        default=168,
        help="Hours of logs to look back (default: 168 = 1 week)",
    )
    ingest_parser.add_argument(
        "--corrections-only",
        action="store_true",
        help="Only ingest correction events",
    )

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--base-model",
        help="Base model to fine-tune",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    train_parser.add_argument(
        "--lora-rank",
        type=int,
        default=64,
        help="LoRA rank",
    )
    train_parser.add_argument(
        "--resume-from",
        type=Path,
        help="Resume training from checkpoint",
    )

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate the model")
    eval_parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to model to evaluate",
    )
    eval_parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["jarvis"],
        help="Benchmarks to run",
    )
    eval_parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Gatekeeper approval threshold",
    )

    # Status command
    status_parser = subparsers.add_parser("status", help="Show pipeline status")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.version:
        from reactor_core import __version__
        print(f"Night Shift Training Engine v{__version__}")
        return 0

    if args.command is None:
        parser.print_help()
        return 0

    # Dispatch to handlers
    if args.command == "run":
        return asyncio.run(_run_pipeline(args))
    elif args.command == "scout":
        return asyncio.run(_run_scout(args))
    elif args.command == "ingest":
        return asyncio.run(_run_ingest(args))
    elif args.command == "train":
        return asyncio.run(_run_train(args))
    elif args.command == "eval":
        return asyncio.run(_run_eval(args))
    elif args.command == "status":
        return _show_status(args)

    return 0


async def _run_pipeline(args: argparse.Namespace) -> int:
    """Run the full training pipeline."""
    from reactor_core.orchestration import (
        NightShiftPipeline,
        PipelineConfig,
        PipelineStage,
        DataSource,
    )
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn

    console = Console()
    console.print("[bold blue]Night Shift Training Pipeline[/bold blue]")
    console.print(f"Started at: {datetime.now().isoformat()}")

    # Build config
    config = PipelineConfig()

    if args.work_dir:
        config.work_dir = args.work_dir

    if args.stop_after:
        config.stop_after = PipelineStage(args.stop_after)

    if args.skip:
        config.skip_stages = [PipelineStage(s) for s in args.skip]

    # Configure sources
    sources = set()
    if "scout" in args.sources:
        sources.add(DataSource.SCOUT)
    if "jarvis" in args.sources:
        sources.add(DataSource.JARVIS_EXPERIENCE)
    if "corrections" in args.sources:
        sources.add(DataSource.JARVIS_CORRECTIONS)
    config.enabled_sources = sources

    # Create and run pipeline
    pipeline = NightShiftPipeline(config)

    def progress_callback(state):
        console.print(f"  Stage: [cyan]{state.stage.value}[/cyan]")

    pipeline.set_progress_callback(progress_callback)

    try:
        result = await pipeline.run(resume=args.resume)

        if result.success:
            console.print("\n[bold green]Pipeline completed successfully![/bold green]")
            console.print(f"Duration: {result.duration_seconds / 60:.1f} minutes")
            console.print("\nArtifacts:")
            for name, path in result.artifacts.items():
                if path:
                    console.print(f"  {name}: {path}")
            return 0
        else:
            console.print(f"\n[bold red]Pipeline failed:[/bold red] {result.error}")
            return 1

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        logger.exception("Pipeline error")
        return 1


async def _run_scout(args: argparse.Namespace) -> int:
    """Run Safe Scout web ingestion."""
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
        create_documentation_topic,
    )
    from reactor_core.distillation import create_teacher_client
    from rich.console import Console
    from rich.progress import Progress

    console = Console()
    console.print("[bold green]Safe Scout - Web Documentation Ingestion[/bold green]")

    work_dir = Path(os.getenv("NIGHTSHIFT_WORK_DIR", Path.home() / ".jarvis" / "nightshift"))
    work_dir.mkdir(parents=True, exist_ok=True)

    # Initialize queue
    queue_config = TopicQueueConfig(db_path=work_dir / "scout_queue.db")
    queue = TopicQueue(queue_config)

    # Add topic if requested
    if args.add_topic:
        topic = create_documentation_topic(
            name=args.add_topic,
            urls=[args.add_url] if args.add_url else [],
        )
        await queue.add_topic(topic)
        console.print(f"Added topic: [cyan]{args.add_topic}[/cyan]")
        if not args.add_url:
            console.print("  (No URLs added - add with --add-url)")
        return 0

    # Get pending topics
    topics = await queue.get_pending_topics(limit=args.topics)
    if not topics:
        console.print("[yellow]No pending topics in queue. Add some with --add-topic[/yellow]")
        return 0

    console.print(f"Processing {len(topics)} topics...")

    # Initialize components
    exec_mode = ExecutionMode.SUBPROCESS if args.no_docker else ExecutionMode.DOCKER
    sandbox_config = SandboxConfig(
        mode=exec_mode,
        timeout_seconds=30,
        max_concurrent=args.concurrency,
    )
    sandbox = SandboxExecutor(sandbox_config)
    validator = URLValidator(URLValidatorConfig())
    compliance = ComplianceFilter()
    extractor = ContentExtractor()

    teacher = create_teacher_client(os.getenv("NIGHTSHIFT_SCOUT_MODEL", "gemini-1.5-flash"))
    synthesizer = KnowledgeSynthesizer(teacher)

    stats = {"pages": 0, "examples": 0, "blocked": 0, "failed": 0}

    with Progress() as progress:
        task = progress.add_task("Processing...", total=len(topics))

        for topic in topics:
            await queue.mark_processing(topic.topic_id)
            urls = topic.urls[:args.pages_per_topic]

            for url in urls:
                # Validate
                validation = await validator.validate(url)
                if not validation.is_safe:
                    stats["blocked"] += 1
                    continue

                try:
                    # Fetch
                    result = await sandbox.execute(url)
                    if not result.success:
                        stats["failed"] += 1
                        continue

                    # Compliance check
                    comp_result = compliance.check_compliance(result.html_content or "", url)
                    if not comp_result.is_compliant:
                        stats["blocked"] += 1
                        continue

                    stats["pages"] += 1

                    # Extract and synthesize
                    content = extractor.extract(result.html_content or "", url)
                    if content.main_content:
                        syn_result = await synthesizer.synthesize(
                            content=content.main_content,
                            title=content.title or topic.name,
                            max_pairs=3,
                        )
                        stats["examples"] += len(syn_result.pairs)

                        # Save pairs
                        output_dir = work_dir / "scout_data"
                        output_dir.mkdir(exist_ok=True)
                        for pair in syn_result.pairs:
                            import json
                            pair_file = output_dir / f"{pair.pair_id}.json"
                            with open(pair_file, "w") as f:
                                json.dump(pair.to_dict(), f, indent=2)

                except Exception as e:
                    stats["failed"] += 1
                    logger.warning(f"Error processing {url}: {e}")

            await queue.mark_completed(topic.topic_id)
            progress.advance(task)

    await sandbox.cleanup()
    await queue.close()

    console.print("\n[bold]Results:[/bold]")
    console.print(f"  Pages fetched: {stats['pages']}")
    console.print(f"  Examples synthesized: {stats['examples']}")
    console.print(f"  Pages blocked: {stats['blocked']}")
    console.print(f"  Pages failed: {stats['failed']}")

    return 0


async def _run_ingest(args: argparse.Namespace) -> int:
    """Run JARVIS experience ingestion."""
    from reactor_core.integration import JARVISConnector, JARVISConnectorConfig
    from rich.console import Console

    console = Console()
    console.print("[bold cyan]JARVIS Experience Ingestion[/bold cyan]")

    jarvis_path = args.jarvis_path or Path(os.getenv(
        "JARVIS_REPO_PATH",
        Path.home() / "Documents" / "repos" / "JARVIS-AI-Agent"
    ))

    if not jarvis_path.exists():
        console.print(f"[red]JARVIS repo not found at: {jarvis_path}[/red]")
        return 1

    config = JARVISConnectorConfig(
        jarvis_repo_path=jarvis_path,
        lookback_hours=args.lookback_hours,
        only_corrections=args.corrections_only,
    )

    connector = JARVISConnector(config)

    if args.corrections_only:
        events = await connector.get_corrections()
        console.print(f"Found {len(events)} correction events")
    else:
        events = await connector.get_events(limit=5000)
        console.print(f"Found {len(events)} total events")

    # Save events
    work_dir = Path(os.getenv("NIGHTSHIFT_WORK_DIR", Path.home() / ".jarvis" / "nightshift"))
    events_dir = work_dir / "jarvis_events"
    events_dir.mkdir(parents=True, exist_ok=True)

    import json
    for event in events:
        event_file = events_dir / f"{event.event_id}.json"
        with open(event_file, "w") as f:
            json.dump(event.to_dict(), f, indent=2)

    console.print(f"Saved events to: {events_dir}")
    return 0


async def _run_train(args: argparse.Namespace) -> int:
    """Run model training."""
    from rich.console import Console

    console = Console()
    console.print("[bold magenta]Night Shift Training[/bold magenta]")
    console.print("Training functionality - coming soon")
    console.print(f"  Base model: {args.base_model or 'default'}")
    console.print(f"  Epochs: {args.epochs}")
    console.print(f"  LoRA rank: {args.lora_rank}")

    # Placeholder - full training implementation would go here
    return 0


async def _run_eval(args: argparse.Namespace) -> int:
    """Run model evaluation."""
    from rich.console import Console

    console = Console()
    console.print("[bold yellow]Model Evaluation[/bold yellow]")
    console.print(f"Model path: {args.model_path}")
    console.print(f"Benchmarks: {', '.join(args.benchmarks)}")
    console.print(f"Threshold: {args.threshold}")

    # Placeholder - full eval implementation would go here
    return 0


def _show_status(args: argparse.Namespace) -> int:
    """Show pipeline status."""
    from rich.console import Console
    from rich.table import Table
    import json

    console = Console()
    console.print("[bold]Night Shift Pipeline Status[/bold]\n")

    work_dir = Path(os.getenv("NIGHTSHIFT_WORK_DIR", Path.home() / ".jarvis" / "nightshift"))
    state_file = work_dir / "pipeline_state.json"

    if not state_file.exists():
        console.print("[yellow]No pipeline state found. Run 'nightshift run' to start.[/yellow]")
        return 0

    with open(state_file) as f:
        state = json.load(f)

    table = Table(title="Pipeline State")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Run ID", state.get("run_id", "N/A"))
    table.add_row("Stage", state.get("stage", "N/A"))
    table.add_row("Started", state.get("started_at", "N/A"))
    table.add_row("Last Updated", state.get("last_updated", "N/A"))
    table.add_row("Scout Topics", str(state.get("scout_topics_processed", 0)))
    table.add_row("Scout Pages", str(state.get("scout_pages_fetched", 0)))
    table.add_row("Scout Examples", str(state.get("scout_examples_synthesized", 0)))
    table.add_row("Ingested", str(state.get("ingestion_count", 0)))
    table.add_row("Formatted", str(state.get("formatted_count", 0)))
    table.add_row("Gatekeeper Passed", str(state.get("gatekeeper_passed", False)))

    if state.get("error"):
        table.add_row("Error", f"[red]{state['error']}[/red]")

    console.print(table)
    return 0


# CLI entry points for pyproject.toml scripts
def ingest() -> int:
    """Entry point for nightshift-ingest."""
    sys.argv = ["nightshift", "ingest"] + sys.argv[1:]
    return main()


def train() -> int:
    """Entry point for nightshift-train."""
    sys.argv = ["nightshift", "train"] + sys.argv[1:]
    return main()


def evaluate() -> int:
    """Entry point for nightshift-eval."""
    sys.argv = ["nightshift", "eval"] + sys.argv[1:]
    return main()


def scout() -> int:
    """Entry point for nightshift-scout."""
    sys.argv = ["nightshift", "scout"] + sys.argv[1:]
    return main()


def run_pipeline() -> int:
    """Entry point for nightshift-pipeline."""
    sys.argv = ["nightshift", "run"] + sys.argv[1:]
    return main()


if __name__ == "__main__":
    sys.exit(main())
