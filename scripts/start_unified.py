#!/usr/bin/env python3
"""
Unified Service Startup Script for JARVIS Ecosystem.

This script coordinates the startup of all JARVIS services across repositories:
- JARVIS-AI-Agent (Main AGI system)
- JARVIS Prime (Cloud deployment)
- Reactor Core / Night Shift (Training Engine)

Features:
- Automatic service discovery
- Health checking with retry logic
- Cross-repo event bridge initialization
- Intelligent dependency resolution
- Graceful shutdown handling

Usage:
    python scripts/start_unified.py [options]

Examples:
    # Start all services
    python scripts/start_unified.py --all

    # Start specific services
    python scripts/start_unified.py --services reactor scout

    # Check service health
    python scripts/start_unified.py --health-check

    # Start with event bridge
    python scripts/start_unified.py --all --enable-bridge
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

# Add parent directory to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
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
from rich.live import Live
from rich.layout import Layout

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
)
logger = logging.getLogger("unified")

console = Console()


class ServiceState(Enum):
    """State of a managed service."""
    UNKNOWN = "unknown"
    STARTING = "starting"
    RUNNING = "running"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class ServiceInfo:
    """Information about a managed service."""
    name: str
    display_name: str
    repo_name: str
    start_command: List[str]
    health_url: Optional[str] = None
    port: int = 0
    cwd: Optional[Path] = None
    env: Dict[str, str] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    state: ServiceState = ServiceState.UNKNOWN
    process: Optional[subprocess.Popen] = None
    last_health_check: Optional[datetime] = None
    startup_timeout: int = 60
    health_retries: int = 5


class UnifiedServiceManager:
    """
    Manages startup and coordination of all JARVIS services.

    Features:
    - Automatic repo discovery
    - Dependency-aware startup ordering
    - Health monitoring
    - Event bridge integration
    - Graceful shutdown
    """

    def __init__(self, repos_base: Optional[Path] = None):
        self.repos_base = repos_base or Path(
            os.getenv("JARVIS_REPOS_BASE", Path.home() / "Documents" / "repos")
        )
        self.services: Dict[str, ServiceInfo] = {}
        self.running_processes: List[subprocess.Popen] = []
        self._shutdown_requested = False
        self._event_bridge = None
        self._bridge_task = None

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Discover services
        self._discover_services()

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        if not self._shutdown_requested:
            self._shutdown_requested = True
            console.print("\n[yellow]Shutdown requested, stopping services...[/yellow]")

    def _discover_services(self) -> None:
        """Discover available services across repos."""
        # JARVIS-AI-Agent
        jarvis_paths = ["JARVIS-AI-Agent", "jarvis-ai-agent"]
        for name in jarvis_paths:
            path = self.repos_base / name
            if path.exists():
                self.services["jarvis"] = ServiceInfo(
                    name="jarvis",
                    display_name="JARVIS AI Agent",
                    repo_name=name,
                    start_command=["python3", "start_system.py", "--backend-only"],
                    health_url="http://localhost:8000/health/ping",
                    port=8000,
                    cwd=path,
                    env={"PYTHONPATH": str(path)},
                )
                break

        # JARVIS Prime
        prime_paths = ["jarvis-prime", "JARVIS-Prime"]
        for name in prime_paths:
            path = self.repos_base / name
            if path.exists():
                self.services["prime"] = ServiceInfo(
                    name="prime",
                    display_name="JARVIS Prime",
                    repo_name=name,
                    start_command=["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"],
                    health_url="http://localhost:8002/health",
                    port=8002,
                    cwd=path,
                    env={"PYTHONPATH": str(path)},
                    depends_on=["jarvis"],
                )
                break

        # Reactor Core / Night Shift
        reactor_paths = ["reactor-core", "REACTOR-CORE"]
        for name in reactor_paths:
            path = self.repos_base / name
            if path.exists():
                self.services["reactor"] = ServiceInfo(
                    name="reactor",
                    display_name="Reactor Core",
                    repo_name=name,
                    start_command=["python3", "-m", "reactor_core.cli", "serve"],
                    health_url="http://localhost:8080/health",
                    port=8080,
                    cwd=path,
                    env={"PYTHONPATH": str(path)},
                )

                # Scout as a sub-service
                self.services["scout"] = ServiceInfo(
                    name="scout",
                    display_name="Safe Scout",
                    repo_name=name,
                    start_command=["python3", "scripts/run_scout.py"],
                    port=0,  # No port, runs as batch
                    cwd=path,
                    env={"PYTHONPATH": str(path)},
                    depends_on=["reactor"],
                )

                # Topic Discovery as a sub-service
                self.services["discovery"] = ServiceInfo(
                    name="discovery",
                    display_name="Topic Discovery",
                    repo_name=name,
                    start_command=["python3", "-c", "from reactor_core.scout import auto_discover_topics; import asyncio; asyncio.run(auto_discover_topics())"],
                    port=0,
                    cwd=path,
                    env={"PYTHONPATH": str(path)},
                    depends_on=["jarvis"],
                )
                break

        # Redis (using docker)
        self.services["redis"] = ServiceInfo(
            name="redis",
            display_name="Redis Cache",
            repo_name="docker",
            start_command=["docker", "run", "-d", "--name", "jarvis-redis", "-p", "6379:6379", "redis:7-alpine"],
            health_url=None,  # Use redis-cli ping
            port=6379,
        )

    async def check_service_health(self, service: ServiceInfo) -> bool:
        """Check if a service is healthy."""
        import aiohttp

        if not service.health_url:
            # For services without HTTP health check, check if process is running
            if service.process and service.process.poll() is None:
                return True
            return False

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    service.health_url,
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as response:
                    return response.status == 200
        except Exception:
            return False

    async def wait_for_healthy(
        self,
        service: ServiceInfo,
        timeout: int = 60,
        interval: float = 2.0,
    ) -> bool:
        """Wait for a service to become healthy."""
        start = datetime.now()

        while (datetime.now() - start).total_seconds() < timeout:
            if self._shutdown_requested:
                return False

            if await self.check_service_health(service):
                service.state = ServiceState.HEALTHY
                service.last_health_check = datetime.now()
                return True

            await asyncio.sleep(interval)

        service.state = ServiceState.UNHEALTHY
        return False

    async def start_service(
        self,
        service_name: str,
        wait_healthy: bool = True,
    ) -> bool:
        """Start a single service."""
        if service_name not in self.services:
            logger.error(f"Unknown service: {service_name}")
            return False

        service = self.services[service_name]

        # Check dependencies
        for dep_name in service.depends_on:
            dep = self.services.get(dep_name)
            if not dep or dep.state not in (ServiceState.RUNNING, ServiceState.HEALTHY):
                logger.warning(f"Dependency {dep_name} not running, starting...")
                if not await self.start_service(dep_name, wait_healthy=True):
                    logger.error(f"Failed to start dependency: {dep_name}")
                    return False

        # Check if already running
        if await self.check_service_health(service):
            logger.info(f"{service.display_name} already running")
            service.state = ServiceState.HEALTHY
            return True

        logger.info(f"Starting {service.display_name}...")
        service.state = ServiceState.STARTING

        # Build environment
        env = os.environ.copy()
        env.update(service.env)

        try:
            # Special handling for Redis (docker)
            if service_name == "redis":
                # Check if container exists
                result = subprocess.run(
                    ["docker", "ps", "-a", "-q", "-f", "name=jarvis-redis"],
                    capture_output=True,
                    text=True,
                )
                if result.stdout.strip():
                    # Container exists, just start it
                    subprocess.run(["docker", "start", "jarvis-redis"], check=True)
                else:
                    # Create new container
                    subprocess.run(service.start_command, check=True)

                service.state = ServiceState.RUNNING
                await asyncio.sleep(2)  # Wait for Redis to be ready
                return True

            # Start process
            process = subprocess.Popen(
                service.start_command,
                cwd=service.cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            service.process = process
            self.running_processes.append(process)
            service.state = ServiceState.RUNNING

            if wait_healthy and service.health_url:
                if await self.wait_for_healthy(service, timeout=service.startup_timeout):
                    logger.info(f"[green]{service.display_name} is healthy[/green]")
                    return True
                else:
                    logger.error(f"{service.display_name} failed health check")
                    return False

            return True

        except Exception as e:
            logger.error(f"Failed to start {service.display_name}: {e}")
            service.state = ServiceState.FAILED
            return False

    async def stop_service(self, service_name: str) -> bool:
        """Stop a single service."""
        if service_name not in self.services:
            return False

        service = self.services[service_name]

        if service_name == "redis":
            try:
                subprocess.run(["docker", "stop", "jarvis-redis"], capture_output=True)
                service.state = ServiceState.STOPPED
                return True
            except Exception:
                return False

        if service.process:
            try:
                service.process.terminate()
                service.process.wait(timeout=10)
                service.state = ServiceState.STOPPED
                return True
            except subprocess.TimeoutExpired:
                service.process.kill()
                service.state = ServiceState.STOPPED
                return True
            except Exception:
                return False

        return True

    async def start_all(
        self,
        services: Optional[List[str]] = None,
        enable_bridge: bool = False,
    ) -> Dict[str, bool]:
        """Start multiple services with dependency ordering."""
        if services is None:
            services = list(self.services.keys())

        results = {}

        # Sort by dependencies (simple topological sort)
        ordered = self._order_by_dependencies(services)

        console.print(Panel(
            f"Starting services: {', '.join(ordered)}",
            title="[bold blue]Unified Service Startup[/bold blue]",
        ))

        for service_name in ordered:
            if self._shutdown_requested:
                break

            success = await self.start_service(service_name)
            results[service_name] = success

            if not success:
                console.print(f"[red]Failed to start {service_name}[/red]")

        # Start event bridge if requested
        if enable_bridge and not self._shutdown_requested:
            await self._start_event_bridge()

        return results

    async def stop_all(self) -> None:
        """Stop all running services."""
        if self._bridge_task:
            self._bridge_task.cancel()
            try:
                await self._bridge_task
            except asyncio.CancelledError:
                pass

        if self._event_bridge:
            await self._event_bridge.stop()

        for service_name in reversed(list(self.services.keys())):
            await self.stop_service(service_name)

    def _order_by_dependencies(self, services: List[str]) -> List[str]:
        """Order services by dependencies (topological sort)."""
        ordered = []
        visited = set()

        def visit(name: str):
            if name in visited:
                return
            visited.add(name)

            service = self.services.get(name)
            if service:
                for dep in service.depends_on:
                    if dep in services or dep in self.services:
                        visit(dep)

            if name in services:
                ordered.append(name)

        for name in services:
            visit(name)

        return ordered

    async def _start_event_bridge(self) -> None:
        """Start the cross-repo event bridge."""
        from reactor_core.integration import (
            EventBridge,
            EventSource,
            FileTransport,
            EventType,
        )

        events_dir = Path(os.getenv(
            "JARVIS_EVENTS_DIR",
            Path.home() / ".jarvis" / "events"
        ))
        events_dir.mkdir(parents=True, exist_ok=True)

        transport = FileTransport(events_dir, EventSource.REACTOR_CORE)
        self._event_bridge = EventBridge(EventSource.REACTOR_CORE, [transport])

        # Register handlers
        @self._event_bridge.on_event(EventType.CORRECTION)
        async def handle_correction(event):
            logger.info(f"Received correction event: {event.event_id}")
            # Could trigger topic discovery here

        @self._event_bridge.on_event(EventType.TRAINING_COMPLETE)
        async def handle_training_complete(event):
            logger.info(f"Training complete: {event.payload}")

        await self._event_bridge.start()
        logger.info("[green]Event bridge started[/green]")

        # Keep bridge running
        self._bridge_task = asyncio.create_task(self._bridge_loop())

    async def _bridge_loop(self) -> None:
        """Keep the event bridge running."""
        while not self._shutdown_requested:
            await asyncio.sleep(1)

    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all services."""
        results = {}

        for name, service in self.services.items():
            healthy = await self.check_service_health(service)
            results[name] = healthy
            service.state = ServiceState.HEALTHY if healthy else ServiceState.UNKNOWN

        return results

    def get_status_table(self) -> Table:
        """Get a Rich table with service status."""
        table = Table(title="Service Status")
        table.add_column("Service", style="cyan")
        table.add_column("State", style="green")
        table.add_column("Port", style="yellow")
        table.add_column("Health URL", style="dim")

        for name, service in self.services.items():
            state_style = {
                ServiceState.HEALTHY: "green",
                ServiceState.RUNNING: "yellow",
                ServiceState.STARTING: "yellow",
                ServiceState.UNHEALTHY: "red",
                ServiceState.FAILED: "red",
                ServiceState.STOPPED: "dim",
                ServiceState.UNKNOWN: "dim",
            }.get(service.state, "white")

            table.add_row(
                service.display_name,
                f"[{state_style}]{service.state.value}[/{state_style}]",
                str(service.port) if service.port else "-",
                service.health_url or "-",
            )

        return table


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified JARVIS Service Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    # Service selection
    parser.add_argument(
        "--all",
        action="store_true",
        help="Start all available services",
    )
    parser.add_argument(
        "--services",
        nargs="+",
        choices=["jarvis", "prime", "reactor", "scout", "redis", "discovery"],
        default=[],
        help="Specific services to start",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=[],
        help="Services to exclude from --all",
    )

    # Control modes
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Only check service health, don't start",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop all services",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Restart all services",
    )

    # Event bridge
    parser.add_argument(
        "--enable-bridge",
        action="store_true",
        help="Enable cross-repo event bridge",
    )

    # Advanced options
    parser.add_argument(
        "--no-deps",
        action="store_true",
        help="Don't automatically start dependencies",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Startup timeout per service in seconds",
    )
    parser.add_argument(
        "--repos-base",
        type=Path,
        help="Base directory for repositories",
    )
    parser.add_argument(
        "--foreground",
        action="store_true",
        help="Run in foreground (keep monitoring)",
    )

    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> int:
    """Async main entry point."""
    manager = UnifiedServiceManager(repos_base=args.repos_base)

    console.print(Panel(
        f"[bold]JARVIS Unified Service Manager[/bold]\n"
        f"Repos base: {manager.repos_base}\n"
        f"Services discovered: {len(manager.services)}",
        title="[blue]Night Shift[/blue]",
    ))

    if args.health_check:
        console.print("\n[bold]Checking service health...[/bold]\n")
        results = await manager.health_check_all()
        console.print(manager.get_status_table())

        all_healthy = all(results.values())
        console.print(f"\nOverall: {'[green]All healthy[/green]' if all_healthy else '[red]Some unhealthy[/red]'}")
        return 0 if all_healthy else 1

    if args.stop:
        console.print("\n[bold]Stopping all services...[/bold]\n")
        await manager.stop_all()
        console.print("[green]All services stopped[/green]")
        return 0

    if args.restart:
        console.print("\n[bold]Restarting all services...[/bold]\n")
        await manager.stop_all()
        await asyncio.sleep(2)

    # Determine which services to start
    services_to_start = []
    if args.all:
        services_to_start = [s for s in manager.services.keys() if s not in args.exclude]
    elif args.services:
        services_to_start = args.services
    else:
        # Default: just show status
        console.print(manager.get_status_table())
        console.print("\nUse --all or --services to start services")
        return 0

    # Start services
    console.print(f"\n[bold]Starting services: {', '.join(services_to_start)}[/bold]\n")

    results = await manager.start_all(
        services=services_to_start,
        enable_bridge=args.enable_bridge,
    )

    console.print()
    console.print(manager.get_status_table())

    success_count = sum(1 for v in results.values() if v)
    console.print(f"\n[bold]Started {success_count}/{len(results)} services[/bold]")

    if args.foreground and not manager._shutdown_requested:
        console.print("\n[dim]Running in foreground. Press Ctrl+C to stop.[/dim]\n")

        try:
            while not manager._shutdown_requested:
                await asyncio.sleep(5)
                # Periodic health check
                await manager.health_check_all()
        except KeyboardInterrupt:
            pass
        finally:
            console.print("\n[yellow]Shutting down...[/yellow]")
            await manager.stop_all()

    return 0 if all(results.values()) else 1


def main() -> int:
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger("unified").setLevel(logging.DEBUG)

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
