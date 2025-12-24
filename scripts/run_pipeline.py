#!/usr/bin/env python3
"""
Night Shift Training Pipeline - Standalone CLI Script.

This script runs the full Night Shift training pipeline with Safe Scout
and JARVIS experience ingestion.

Usage:
    python scripts/run_pipeline.py [options]

Examples:
    # Run full pipeline
    python scripts/run_pipeline.py

    # Run with Scout only
    python scripts/run_pipeline.py --sources scout

    # Resume from checkpoint
    python scripts/run_pipeline.py --resume

    # Stop after training (skip eval and quantization)
    python scripts/run_pipeline.py --stop-after training
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
)
from rich.table import Table

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("nightshift")

console = Console()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Night Shift Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # General options
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running",
    )

    # Pipeline control
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous checkpoint",
    )
    parser.add_argument(
        "--stop-after",
        choices=["scouting", "ingesting", "formatting", "distilling", "training", "evaluating", "quantizing"],
        help="Stop pipeline after this stage",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        choices=["scouting", "ingesting", "formatting", "distilling", "training", "evaluating", "quantizing"],
        default=[],
        help="Skip these stages",
    )

    # Data sources
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["scout", "jarvis", "corrections", "synthetic"],
        default=["scout", "jarvis"],
        help="Data sources to enable",
    )

    # Directory configuration
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path(os.getenv("NIGHTSHIFT_WORK_DIR", Path.home() / ".jarvis" / "nightshift")),
        help="Working directory for pipeline outputs",
    )
    parser.add_argument(
        "--jarvis-path",
        type=Path,
        default=Path(os.getenv("JARVIS_REPO_PATH", Path.home() / "Documents/repos/JARVIS-AI-Agent")),
        help="Path to JARVIS-AI-Agent repo",
    )

    # Scout configuration
    parser.add_argument(
        "--scout-topics",
        type=int,
        default=int(os.getenv("NIGHTSHIFT_SCOUT_MAX_TOPICS", "50")),
        help="Maximum topics for Scout to process",
    )
    parser.add_argument(
        "--scout-pages",
        type=int,
        default=int(os.getenv("NIGHTSHIFT_SCOUT_MAX_PAGES", "10")),
        help="Maximum pages per topic",
    )
    parser.add_argument(
        "--scout-concurrency",
        type=int,
        default=int(os.getenv("NIGHTSHIFT_SCOUT_CONCURRENCY", "5")),
        help="Scout concurrency level",
    )
    parser.add_argument(
        "--no-docker",
        action="store_true",
        help="Disable Docker sandbox for Scout",
    )

    # Training configuration
    parser.add_argument(
        "--base-model",
        default=os.getenv("NIGHTSHIFT_BASE_MODEL", "meta-llama/Llama-3.2-3B"),
        help="Base model for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=int(os.getenv("NIGHTSHIFT_LORA_RANK", "64")),
        help="LoRA rank for fine-tuning",
    )

    # Evaluation
    parser.add_argument(
        "--eval-threshold",
        type=float,
        default=0.7,
        help="Gatekeeper evaluation threshold",
    )
    parser.add_argument(
        "--skip-gatekeeper",
        action="store_true",
        help="Skip gatekeeper approval",
    )

    # Quantization
    parser.add_argument(
        "--quantize",
        default="q4_k_m",
        choices=["q4_0", "q4_1", "q4_k_m", "q5_0", "q5_1", "q5_k_m", "q8_0"],
        help="Quantization method for GGUF",
    )
    parser.add_argument(
        "--skip-quantization",
        action="store_true",
        help="Skip GGUF quantization",
    )

    # JARVIS Prime
    parser.add_argument(
        "--enable-prime",
        action="store_true",
        help="Enable JARVIS Prime integration",
    )
    parser.add_argument(
        "--prime-host",
        default=os.getenv("JARVIS_PRIME_HOST", "localhost"),
        help="JARVIS Prime host",
    )
    parser.add_argument(
        "--prime-port",
        type=int,
        default=int(os.getenv("JARVIS_PRIME_PORT", "8002")),
        help="JARVIS Prime port",
    )

    return parser.parse_args()


async def run_pipeline(args: argparse.Namespace) -> int:
    """Run the training pipeline."""
    from reactor_core.orchestration import (
        NightShiftPipeline,
        PipelineConfig,
        PipelineStage,
        DataSource,
    )

    console.rule("[bold blue]Night Shift Training Pipeline[/bold blue]")
    console.print(f"Started: {datetime.now().isoformat()}")
    console.print(f"Work directory: {args.work_dir}")
    console.print()

    # Build configuration
    enabled_sources = set()
    if "scout" in args.sources:
        enabled_sources.add(DataSource.SCOUT)
    if "jarvis" in args.sources:
        enabled_sources.add(DataSource.JARVIS_EXPERIENCE)
    if "corrections" in args.sources:
        enabled_sources.add(DataSource.JARVIS_CORRECTIONS)
    if "synthetic" in args.sources:
        enabled_sources.add(DataSource.SYNTHETIC)

    config = PipelineConfig(
        work_dir=args.work_dir,
        enabled_sources=enabled_sources,
        # Scout config
        scout_max_topics=args.scout_topics,
        scout_max_pages_per_topic=args.scout_pages,
        scout_concurrency=args.scout_concurrency,
        scout_use_docker=not args.no_docker,
        # JARVIS config
        jarvis_repo_path=args.jarvis_path,
        # Prime config
        prime_enabled=args.enable_prime,
        prime_host=args.prime_host,
        prime_port=args.prime_port,
        # Training config
        base_model=args.base_model,
        num_epochs=args.epochs,
        lora_rank=args.lora_rank,
        # Evaluation config
        eval_threshold=args.eval_threshold,
        require_gatekeeper=not args.skip_gatekeeper,
        # Quantization config
        quantization_method=args.quantize,
        skip_quantization=args.skip_quantization,
        # Stage control
        skip_stages=[PipelineStage(s) for s in args.skip],
        stop_after=PipelineStage(args.stop_after) if args.stop_after else None,
    )

    # Show configuration
    config_table = Table(title="Configuration", show_header=True)
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="green")
    config_table.add_row("Data Sources", ", ".join(args.sources))
    config_table.add_row("Base Model", args.base_model)
    config_table.add_row("Scout Topics", str(args.scout_topics))
    config_table.add_row("Docker Sandbox", str(not args.no_docker))
    config_table.add_row("JARVIS Prime", str(args.enable_prime))
    config_table.add_row("Quantization", args.quantize if not args.skip_quantization else "Disabled")
    console.print(config_table)
    console.print()

    if args.dry_run:
        console.print("[yellow]Dry run - exiting without running pipeline[/yellow]")
        return 0

    # Create pipeline
    pipeline = NightShiftPipeline(config)

    # Set up progress tracking
    current_stage = ["IDLE"]

    def progress_callback(state):
        current_stage[0] = state.stage.value
        logger.info(f"Stage: {state.stage.value}")

    def error_callback(error, stage):
        console.print(f"[red]Error in {stage.value}: {error}[/red]")

    pipeline.set_progress_callback(progress_callback)
    pipeline.set_error_callback(error_callback)

    # Run pipeline
    console.print("[bold]Running pipeline...[/bold]\n")

    try:
        result = await pipeline.run(resume=args.resume)

        console.print()
        console.rule("[bold]Pipeline Complete[/bold]")

        if result.success:
            console.print("[bold green]SUCCESS[/bold green]")
            console.print(f"Duration: {result.duration_seconds / 60:.1f} minutes")

            # Show metrics
            if result.metrics:
                metrics_table = Table(title="Metrics")
                metrics_table.add_column("Metric", style="cyan")
                metrics_table.add_column("Value", style="green")
                for key, value in result.metrics.items():
                    metrics_table.add_row(key, f"{value:.4f}" if isinstance(value, float) else str(value))
                console.print(metrics_table)

            # Show artifacts
            if result.artifacts:
                console.print("\n[bold]Artifacts:[/bold]")
                for name, path in result.artifacts.items():
                    if path:
                        console.print(f"  {name}: {path}")

            return 0
        else:
            console.print(f"[bold red]FAILED[/bold red]: {result.error}")
            return 1

    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline interrupted by user[/yellow]")
        return 130
    except Exception as e:
        console.print(f"\n[bold red]Fatal error:[/bold red] {e}")
        logger.exception("Pipeline error")
        return 1


def main() -> int:
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger("reactor_core").setLevel(logging.DEBUG)
        logging.getLogger("nightshift").setLevel(logging.DEBUG)

    return asyncio.run(run_pipeline(args))


if __name__ == "__main__":
    sys.exit(main())
