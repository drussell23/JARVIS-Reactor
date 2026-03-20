"""Tests for VisionCalibrator — learns from vision action outcomes."""
import pytest
from reactor_core.training.vision_calibrator import VisionCalibrator


@pytest.fixture
def calibrator():
    return VisionCalibrator()


class TestConfidenceTracking:
    def test_record_action_outcome(self, calibrator):
        calibrator.record_outcome(
            task_type="ui_element_detection",
            confidence=0.85,
            success=True,
            model_used="llava_7b",
        )
        stats = calibrator.get_stats("ui_element_detection")
        assert stats["total"] == 1
        assert stats["successes"] == 1

    def test_multiple_outcomes_tracked(self, calibrator):
        for conf, success in [(0.9, True), (0.8, True), (0.7, False), (0.6, False)]:
            calibrator.record_outcome("click", conf, success, "llava_7b")
        stats = calibrator.get_stats("click")
        assert stats["total"] == 4
        assert stats["successes"] == 2
        assert stats["failures"] == 2

    def test_success_rate_by_task(self, calibrator):
        for _ in range(8):
            calibrator.record_outcome("click", 0.9, True, "llava_7b")
        for _ in range(2):
            calibrator.record_outcome("click", 0.5, False, "llava_7b")
        rate = calibrator.success_rate("click")
        assert rate == pytest.approx(0.8, abs=0.01)


class TestThresholdCalibration:
    def test_calibrate_produces_thresholds(self, calibrator):
        # Feed data: high confidence succeeds, low fails
        for _ in range(50):
            calibrator.record_outcome("click", 0.85, True, "llava_7b")
        for _ in range(50):
            calibrator.record_outcome("click", 0.50, False, "llava_7b")
        thresholds = calibrator.calibrate()
        assert "click" in thresholds
        # Threshold should be between 0.50 and 0.85
        assert 0.50 <= thresholds["click"] <= 0.85

    def test_calibrate_empty_returns_defaults(self, calibrator):
        thresholds = calibrator.calibrate()
        assert isinstance(thresholds, dict)


class TestModelPerformance:
    def test_track_by_model(self, calibrator):
        calibrator.record_outcome("click", 0.9, True, "llava_7b")
        calibrator.record_outcome("click", 0.8, False, "qwen2_vl_7b")
        model_stats = calibrator.get_model_stats()
        assert "llava_7b" in model_stats
        assert "qwen2_vl_7b" in model_stats
        assert model_stats["llava_7b"]["successes"] == 1
        assert model_stats["qwen2_vl_7b"]["failures"] == 1


class TestExport:
    def test_export_thresholds_json(self, calibrator):
        for _ in range(20):
            calibrator.record_outcome("click", 0.85, True, "llava_7b")
        for _ in range(20):
            calibrator.record_outcome("click", 0.45, False, "llava_7b")
        exported = calibrator.export_thresholds_json()
        assert isinstance(exported, str)
        import json
        parsed = json.loads(exported)
        assert "thresholds" in parsed
