"""Tests for GGUF model deployment gate."""

import pytest
import struct
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.fixture
def tmp_model(tmp_path):
    """Create a fake GGUF file for testing."""
    model_file = tmp_path / "test_model.gguf"
    header = struct.pack("<I", 0x46475547)  # GGUF magic
    header += struct.pack("<I", 3)  # version 3
    model_file.write_bytes(header + b"\x00" * 1024)
    return model_file


@pytest.fixture
def tmp_model_v2(tmp_path):
    """Create a fake GGUF v2 file for testing."""
    model_file = tmp_path / "test_model_v2.gguf"
    header = struct.pack("<I", 0x46475547)  # GGUF magic
    header += struct.pack("<I", 2)  # version 2
    model_file.write_bytes(header + b"\x00" * 1024)
    return model_file


@pytest.fixture
def tmp_model_v1(tmp_path):
    """Create a fake GGUF v1 file for testing."""
    model_file = tmp_path / "test_model_v1.gguf"
    header = struct.pack("<I", 0x46475547)  # GGUF magic
    header += struct.pack("<I", 1)  # version 1
    model_file.write_bytes(header + b"\x00" * 1024)
    return model_file


@pytest.fixture
def small_model(tmp_path):
    """Create a suspiciously small GGUF file."""
    model_file = tmp_path / "tiny_model.gguf"
    model_file.write_bytes(b"\x00" * 100)
    return model_file


@pytest.fixture
def oversized_model(tmp_path):
    """Create a model file that exceeds maximum size (simulated via low max_file_size_bytes)."""
    model_file = tmp_path / "big_model.gguf"
    header = struct.pack("<I", 0x46475547)  # GGUF magic
    header += struct.pack("<I", 3)  # version 3
    model_file.write_bytes(header + b"\x00" * 2048)
    return model_file


class TestCheckResult:
    """Verify CheckResult dataclass behavior."""

    def test_check_result_defaults(self):
        from reactor_core.deployment.gate import CheckResult

        result = CheckResult(name="test_check", passed=True)
        assert result.name == "test_check"
        assert result.passed is True
        assert result.severity == "critical"
        assert result.reason == ""
        assert result.value is None

    def test_check_result_custom_fields(self):
        from reactor_core.deployment.gate import CheckResult

        result = CheckResult(
            name="size_check",
            passed=False,
            severity="warning",
            reason="File too small",
            value=42.0,
        )
        assert result.passed is False
        assert result.severity == "warning"
        assert result.reason == "File too small"
        assert result.value == 42.0


class TestGateResult:
    """Verify GateResult aggregation behavior."""

    def test_gate_result_passed(self):
        from reactor_core.deployment.gate import CheckResult, GateResult

        checks = [
            CheckResult(name="header", passed=True),
            CheckResult(name="size", passed=True),
        ]
        result = GateResult(passed=True, checks=checks, model_path="/tmp/model.gguf")
        assert result.passed is True
        assert len(result.checks) == 2
        assert result.model_path == "/tmp/model.gguf"

    def test_critical_failures_property(self):
        from reactor_core.deployment.gate import CheckResult, GateResult

        checks = [
            CheckResult(name="header", passed=False, severity="critical", reason="Bad magic"),
            CheckResult(name="size", passed=True, severity="critical"),
            CheckResult(name="inference", passed=False, severity="warning", reason="Slow"),
        ]
        result = GateResult(passed=False, checks=checks)
        critical = result.critical_failures
        assert len(critical) == 1
        assert critical[0].name == "header"

    def test_summary_approved(self):
        from reactor_core.deployment.gate import CheckResult, GateResult

        checks = [
            CheckResult(name="header", passed=True),
            CheckResult(name="size", passed=True),
        ]
        result = GateResult(passed=True, checks=checks)
        summary = result.summary()
        assert "APPROVED" in summary
        assert "2/2" in summary

    def test_summary_rejected(self):
        from reactor_core.deployment.gate import CheckResult, GateResult

        checks = [
            CheckResult(name="header", passed=False, severity="critical", reason="Invalid magic"),
            CheckResult(name="size", passed=True),
        ]
        result = GateResult(passed=False, checks=checks)
        summary = result.summary()
        assert "REJECTED" in summary
        assert "1/2" in summary
        assert "Invalid magic" in summary


class TestDeploymentGate:
    """Verify deployment gate catches bad models."""

    @pytest.mark.asyncio
    async def test_valid_gguf_header_passes(self, tmp_model):
        from reactor_core.deployment.gate import DeploymentGate

        gate = DeploymentGate(min_file_size_bytes=512)
        result = await gate._check_gguf_header(tmp_model)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_valid_gguf_v2_header_passes(self, tmp_model_v2):
        from reactor_core.deployment.gate import DeploymentGate

        gate = DeploymentGate(min_file_size_bytes=512)
        result = await gate._check_gguf_header(tmp_model_v2)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_valid_gguf_v1_header_passes(self, tmp_model_v1):
        from reactor_core.deployment.gate import DeploymentGate

        gate = DeploymentGate(min_file_size_bytes=512)
        result = await gate._check_gguf_header(tmp_model_v1)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_invalid_gguf_header_fails(self, tmp_path):
        from reactor_core.deployment.gate import DeploymentGate

        bad_file = tmp_path / "bad.gguf"
        bad_file.write_bytes(b"NOT_GGUF_DATA" + b"\x00" * 1024)
        gate = DeploymentGate(min_file_size_bytes=512)
        result = await gate._check_gguf_header(bad_file)
        assert result.passed is False
        assert "magic" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_invalid_gguf_version_fails(self, tmp_path):
        from reactor_core.deployment.gate import DeploymentGate

        bad_version_file = tmp_path / "bad_version.gguf"
        header = struct.pack("<I", 0x46475547)  # valid magic
        header += struct.pack("<I", 99)  # invalid version
        bad_version_file.write_bytes(header + b"\x00" * 1024)
        gate = DeploymentGate(min_file_size_bytes=512)
        result = await gate._check_gguf_header(bad_version_file)
        assert result.passed is False
        assert "version" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_truncated_header_fails(self, tmp_path):
        from reactor_core.deployment.gate import DeploymentGate

        truncated = tmp_path / "truncated.gguf"
        truncated.write_bytes(b"\x47\x47")  # only 2 bytes
        gate = DeploymentGate(min_file_size_bytes=0)
        result = await gate._check_gguf_header(truncated)
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_nonexistent_file_fails(self, tmp_path):
        from reactor_core.deployment.gate import DeploymentGate

        missing = tmp_path / "does_not_exist.gguf"
        gate = DeploymentGate(min_file_size_bytes=0)
        result = await gate._check_gguf_header(missing)
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_small_file_fails(self, small_model):
        from reactor_core.deployment.gate import DeploymentGate

        gate = DeploymentGate(min_file_size_bytes=500)
        result = await gate._check_file_size(small_model)
        assert result.passed is False
        assert result.severity == "critical"

    @pytest.mark.asyncio
    async def test_file_size_within_range_passes(self, tmp_model):
        from reactor_core.deployment.gate import DeploymentGate

        gate = DeploymentGate(min_file_size_bytes=512, max_file_size_bytes=10_000)
        result = await gate._check_file_size(tmp_model)
        assert result.passed is True
        assert result.value is not None  # should record actual size

    @pytest.mark.asyncio
    async def test_file_exceeds_max_fails(self, oversized_model):
        from reactor_core.deployment.gate import DeploymentGate

        gate = DeploymentGate(min_file_size_bytes=0, max_file_size_bytes=1024)
        result = await gate._check_file_size(oversized_model)
        assert result.passed is False
        assert "exceeds" in result.reason.lower() or "max" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_validate_aggregates_checks(self, tmp_model):
        from reactor_core.deployment.gate import DeploymentGate

        gate = DeploymentGate(min_file_size_bytes=512, skip_inference_check=True)
        result = await gate.validate(tmp_model)
        assert result.passed is True
        assert len(result.checks) >= 2  # At least header + size checks
        assert result.model_path is not None

    @pytest.mark.asyncio
    async def test_validate_fails_on_bad_header(self, tmp_path):
        from reactor_core.deployment.gate import DeploymentGate

        bad_file = tmp_path / "bad.gguf"
        bad_file.write_bytes(b"NOPE" + b"\x00" * 2048)
        gate = DeploymentGate(min_file_size_bytes=512, skip_inference_check=True)
        result = await gate.validate(bad_file)
        assert result.passed is False
        assert len(result.critical_failures) >= 1

    @pytest.mark.asyncio
    async def test_validate_with_manifest(self, tmp_model):
        from reactor_core.deployment.gate import DeploymentGate

        gate = DeploymentGate(min_file_size_bytes=512, skip_inference_check=True)
        manifest = {"model_name": "test-q4", "quant_method": "q4_k_m"}
        result = await gate.validate(tmp_model, manifest=manifest)
        # Should still pass even with manifest metadata
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_validate_records_model_path(self, tmp_model):
        from reactor_core.deployment.gate import DeploymentGate

        gate = DeploymentGate(min_file_size_bytes=512, skip_inference_check=True)
        result = await gate.validate(tmp_model)
        assert str(tmp_model) in str(result.model_path)

    @pytest.mark.asyncio
    async def test_inference_check_skipped_gracefully(self, tmp_model):
        """When skip_inference_check=True, no inference check should appear in results."""
        from reactor_core.deployment.gate import DeploymentGate

        gate = DeploymentGate(min_file_size_bytes=512, skip_inference_check=True)
        result = await gate.validate(tmp_model)
        check_names = [c.name for c in result.checks]
        assert "generates_text" not in check_names

    @pytest.mark.asyncio
    async def test_inference_check_without_llama_cpp(self, tmp_model):
        """When llama-cpp-python is not installed, inference check should skip gracefully."""
        from reactor_core.deployment.gate import DeploymentGate

        gate = DeploymentGate(min_file_size_bytes=512, skip_inference_check=False)
        # Mock llama_cpp as unavailable
        with patch.dict("sys.modules", {"llama_cpp": None}):
            result = await gate._check_generates_text(tmp_model)
            # Should not crash, should return a warning-level skip
            assert result.passed is True or result.severity == "warning"

    @pytest.mark.asyncio
    async def test_default_test_prompts(self):
        """DeploymentGate should have sensible default test prompts."""
        from reactor_core.deployment.gate import DeploymentGate

        gate = DeploymentGate()
        assert len(gate.test_prompts) >= 3

    @pytest.mark.asyncio
    async def test_custom_test_prompts(self):
        """Custom test prompts should override defaults."""
        from reactor_core.deployment.gate import DeploymentGate

        custom = ["Hello, world!", "What is AI?"]
        gate = DeploymentGate(test_prompts=custom)
        assert gate.test_prompts == custom
