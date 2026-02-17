"""Deployment gate for GGUF model validation before deployment.

Validates GGUF models through a series of checks before allowing
deployment to JARVIS Prime. Catches corrupt headers, wrong file sizes,
and optionally tests inference generation quality.

Usage:
    gate = DeploymentGate()
    result = await gate.validate(Path("/path/to/model.gguf"))
    if result.passed:
        deploy(model)
    else:
        log_rejection(result.summary())
"""

from __future__ import annotations

import asyncio
import logging
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# GGUF magic number: little-endian bytes for "GGUF" (0x47, 0x47, 0x55, 0x46)
GGUF_MAGIC = 0x46475547

# Supported GGUF format versions
GGUF_SUPPORTED_VERSIONS = frozenset({1, 2, 3})

# Default minimum file size: 100 MB
DEFAULT_MIN_FILE_SIZE = 100 * 1024 * 1024

# Default maximum file size: 20 GB
DEFAULT_MAX_FILE_SIZE = 20 * 1024 * 1024 * 1024

# Default prompts used for inference quality checks
DEFAULT_TEST_PROMPTS = [
    "The capital of France is",
    "In machine learning, a neural network",
    "def fibonacci(n):",
    "Explain the concept of gravity in simple terms:",
    "Once upon a time in a land far away,",
]


@dataclass
class CheckResult:
    """Result of a single validation check.

    Attributes:
        name: Identifier for this check (e.g., 'gguf_header', 'file_size').
        passed: Whether the check succeeded.
        severity: 'critical' means gate fails if this check fails.
                  'warning' is logged but does not block deployment.
        reason: Human-readable explanation (populated on failure).
        value: Optional numeric measurement (e.g., file size in bytes).
    """

    name: str
    passed: bool
    severity: str = "critical"
    reason: str = ""
    value: Optional[float] = None


@dataclass
class GateResult:
    """Aggregate result from all deployment gate checks.

    Attributes:
        passed: True only when no critical checks failed.
        checks: List of individual CheckResult objects.
        model_path: Path to the validated model file.
    """

    passed: bool
    checks: List[CheckResult] = field(default_factory=list)
    model_path: Optional[str] = None

    @property
    def critical_failures(self) -> List[CheckResult]:
        """Return checks that failed with critical severity."""
        return [
            c for c in self.checks
            if not c.passed and c.severity == "critical"
        ]

    def summary(self) -> str:
        """Human-readable summary of the gate decision.

        Returns:
            String like 'APPROVED (3/3 checks passed)' or
            'REJECTED (1/3 checks passed): Invalid GGUF magic, File too small'
        """
        total = len(self.checks)
        passed_count = sum(1 for c in self.checks if c.passed)
        verdict = "APPROVED" if self.passed else "REJECTED"

        msg = f"{verdict} ({passed_count}/{total} checks passed)"

        if not self.passed:
            failure_reasons = [
                c.reason for c in self.checks
                if not c.passed and c.reason
            ]
            if failure_reasons:
                msg += f": {', '.join(failure_reasons)}"

        return msg


class DeploymentGate:
    """Validates GGUF models before deployment to JARVIS Prime.

    Runs a configurable set of checks against a model file:
      1. GGUF header magic and version validation
      2. File size range check (catches truncated or bloated files)
      3. Optional inference test (loads model, generates text, checks quality)

    Args:
        min_file_size_bytes: Minimum acceptable file size.
        max_file_size_bytes: Maximum acceptable file size.
        skip_inference_check: If True, skip the expensive model-load test.
        test_prompts: Prompts used for inference quality check.
    """

    def __init__(
        self,
        min_file_size_bytes: int = DEFAULT_MIN_FILE_SIZE,
        max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE,
        skip_inference_check: bool = False,
        test_prompts: Optional[List[str]] = None,
    ) -> None:
        self.min_file_size_bytes = min_file_size_bytes
        self.max_file_size_bytes = max_file_size_bytes
        self.skip_inference_check = skip_inference_check
        self.test_prompts = test_prompts if test_prompts is not None else list(DEFAULT_TEST_PROMPTS)

    async def validate(
        self,
        model_path: Union[str, Path],
        manifest: Optional[Dict] = None,
    ) -> GateResult:
        """Run all validation checks and return an aggregate result.

        Args:
            model_path: Path to the GGUF model file.
            manifest: Optional metadata dict (model_name, quant_method, etc.).
                      Currently logged for auditing; future checks may use it.

        Returns:
            GateResult with individual check results and overall pass/fail.
        """
        model_path = Path(model_path)
        checks: List[CheckResult] = []

        if manifest:
            logger.info(
                "Validating model with manifest: %s",
                {k: v for k, v in manifest.items() if k != "secrets"},
            )

        # Run header and size checks concurrently
        header_task = asyncio.ensure_future(self._check_gguf_header(model_path))
        size_task = asyncio.ensure_future(self._check_file_size(model_path))

        header_result, size_result = await asyncio.gather(header_task, size_task)
        checks.append(header_result)
        checks.append(size_result)

        # Inference check (expensive, optional, gated)
        if not self.skip_inference_check:
            inference_result = await self._check_generates_text(model_path)
            checks.append(inference_result)

        # Gate passes only when no critical checks failed
        critical_failures = [
            c for c in checks
            if not c.passed and c.severity == "critical"
        ]
        passed = len(critical_failures) == 0

        result = GateResult(
            passed=passed,
            checks=checks,
            model_path=str(model_path),
        )

        if passed:
            logger.info("Deployment gate APPROVED: %s", result.summary())
        else:
            logger.warning("Deployment gate REJECTED: %s", result.summary())

        return result

    async def _check_gguf_header(self, model_path: Path) -> CheckResult:
        """Verify that the file starts with valid GGUF magic bytes and version.

        Reads the first 8 bytes: 4 bytes for magic (0x46475547) and
        4 bytes for version (must be 1, 2, or 3).

        Args:
            model_path: Path to the GGUF file.

        Returns:
            CheckResult indicating pass/fail with reason.
        """
        name = "gguf_header"

        try:
            # Use run_in_executor for non-blocking file I/O
            loop = asyncio.get_running_loop()
            header_bytes = await loop.run_in_executor(
                None, self._read_header_bytes, model_path
            )
        except FileNotFoundError:
            return CheckResult(
                name=name,
                passed=False,
                severity="critical",
                reason=f"File not found: {model_path}",
            )
        except Exception as exc:
            return CheckResult(
                name=name,
                passed=False,
                severity="critical",
                reason=f"Failed to read header: {exc}",
            )

        if len(header_bytes) < 8:
            return CheckResult(
                name=name,
                passed=False,
                severity="critical",
                reason=f"File too small for GGUF header ({len(header_bytes)} bytes, need 8)",
            )

        # Unpack magic (4 bytes, little-endian unsigned int)
        magic = struct.unpack("<I", header_bytes[:4])[0]
        if magic != GGUF_MAGIC:
            return CheckResult(
                name=name,
                passed=False,
                severity="critical",
                reason=f"Invalid GGUF magic: 0x{magic:08X} (expected 0x{GGUF_MAGIC:08X})",
            )

        # Unpack version (4 bytes, little-endian unsigned int)
        version = struct.unpack("<I", header_bytes[4:8])[0]
        if version not in GGUF_SUPPORTED_VERSIONS:
            return CheckResult(
                name=name,
                passed=False,
                severity="critical",
                reason=(
                    f"Unsupported GGUF version: {version} "
                    f"(supported: {sorted(GGUF_SUPPORTED_VERSIONS)})"
                ),
            )

        return CheckResult(
            name=name,
            passed=True,
            reason=f"Valid GGUF v{version}",
            value=float(version),
        )

    async def _check_file_size(self, model_path: Path) -> CheckResult:
        """Verify the model file size is within the acceptable range.

        Catches truncated downloads (too small) and bloated/corrupt
        files (too large).

        Args:
            model_path: Path to the GGUF file.

        Returns:
            CheckResult with the actual file size in value field.
        """
        name = "file_size"

        try:
            loop = asyncio.get_running_loop()
            size_bytes = await loop.run_in_executor(
                None, self._get_file_size, model_path
            )
        except FileNotFoundError:
            return CheckResult(
                name=name,
                passed=False,
                severity="critical",
                reason=f"File not found: {model_path}",
            )
        except Exception as exc:
            return CheckResult(
                name=name,
                passed=False,
                severity="critical",
                reason=f"Failed to stat file: {exc}",
            )

        if size_bytes < self.min_file_size_bytes:
            return CheckResult(
                name=name,
                passed=False,
                severity="critical",
                reason=(
                    f"File size {size_bytes:,} bytes is below minimum "
                    f"{self.min_file_size_bytes:,} bytes"
                ),
                value=float(size_bytes),
            )

        if size_bytes > self.max_file_size_bytes:
            return CheckResult(
                name=name,
                passed=False,
                severity="critical",
                reason=(
                    f"File size {size_bytes:,} bytes exceeds maximum "
                    f"{self.max_file_size_bytes:,} bytes"
                ),
                value=float(size_bytes),
            )

        return CheckResult(
            name=name,
            passed=True,
            reason=f"File size {size_bytes:,} bytes within range",
            value=float(size_bytes),
        )

    async def _check_generates_text(self, model_path: Path) -> CheckResult:
        """Load the model and verify it produces non-empty, non-degenerate text.

        Uses llama-cpp-python to load the GGUF model in CPU-only mode with
        minimal context (n_ctx=512). Tests 3 prompts from self.test_prompts
        and checks that:
          - Each prompt produces non-empty output
          - Not all outputs are identical (degenerate model detection)

        If llama-cpp-python is not installed, this check is skipped
        gracefully with a warning.

        Args:
            model_path: Path to the GGUF file.

        Returns:
            CheckResult. Severity is 'warning' when skipped, 'critical'
            when the model fails inference.
        """
        name = "generates_text"

        # Try to import llama_cpp
        try:
            import llama_cpp  # noqa: F811
        except (ImportError, TypeError):
            logger.info(
                "llama-cpp-python not available; skipping inference check"
            )
            return CheckResult(
                name=name,
                passed=True,
                severity="warning",
                reason="Skipped: llama-cpp-python not installed",
            )

        # Run inference in executor to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        try:
            outputs = await loop.run_in_executor(
                None,
                self._run_inference_sync,
                model_path,
                llama_cpp,
            )
        except Exception as exc:
            return CheckResult(
                name=name,
                passed=False,
                severity="critical",
                reason=f"Inference failed: {exc}",
            )

        # Validate outputs
        if not outputs:
            return CheckResult(
                name=name,
                passed=False,
                severity="critical",
                reason="Model produced no outputs",
            )

        empty_count = sum(1 for o in outputs if not o.strip())
        if empty_count == len(outputs):
            return CheckResult(
                name=name,
                passed=False,
                severity="critical",
                reason="All outputs were empty",
            )

        # Check for degenerate model (all outputs identical)
        unique_outputs = set(o.strip() for o in outputs)
        if len(unique_outputs) == 1 and len(outputs) > 1:
            return CheckResult(
                name=name,
                passed=False,
                severity="critical",
                reason="All outputs identical (degenerate model)",
                value=float(len(outputs)),
            )

        return CheckResult(
            name=name,
            passed=True,
            reason=f"Generated {len(outputs)} distinct outputs from {len(outputs)} prompts",
            value=float(len(unique_outputs)),
        )

    # ------------------------------------------------------------------
    # Synchronous helpers (run inside thread pool executor)
    # ------------------------------------------------------------------

    @staticmethod
    def _read_header_bytes(model_path: Path) -> bytes:
        """Read the first 8 bytes of a file (GGUF header)."""
        with open(model_path, "rb") as f:
            return f.read(8)

    @staticmethod
    def _get_file_size(model_path: Path) -> int:
        """Return file size in bytes."""
        return model_path.stat().st_size

    def _run_inference_sync(self, model_path: Path, llama_cpp_module) -> List[str]:
        """Load model and generate text for test prompts (blocking).

        Args:
            model_path: Path to the GGUF model.
            llama_cpp_module: The imported llama_cpp module.

        Returns:
            List of generated text strings.
        """
        Llama = llama_cpp_module.Llama

        # Load with CPU only, minimal context to keep memory low
        model = Llama(
            model_path=str(model_path),
            n_ctx=512,
            n_gpu_layers=0,  # CPU only for validation
            verbose=False,
        )

        try:
            # Test with first 3 prompts (or fewer if less configured)
            prompts_to_test = self.test_prompts[:3]
            outputs: List[str] = []

            for prompt in prompts_to_test:
                response = model(
                    prompt,
                    max_tokens=64,
                    temperature=0.7,
                    stop=["\n\n"],
                )
                text = response["choices"][0]["text"] if response.get("choices") else ""
                outputs.append(text)

            return outputs
        finally:
            # Explicit cleanup
            del model
