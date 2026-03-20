"""
VisionCalibrator — learns optimal confidence thresholds from vision action outcomes.

Receives vision_action_outcome events from JARVIS, tracks per-task-type
confidence vs actual success rate, produces adjusted_thresholds.json
for PRECHECK gate calibration.

Spec: Section 8 of realtime-vision-action-loop-design.md
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default confidence threshold used when no calibration data is available.
_DEFAULT_THRESHOLD = 0.75

# Target success rate — the calibrator seeks the lowest confidence value that
# still achieves at least this success rate from outcomes seen so far.
_TARGET_SUCCESS_RATE = 0.80


class VisionCalibrator:
    """
    Learns optimal per-task-type confidence thresholds from vision action outcomes.

    Call record_outcome() each time a vision action completes, then call
    calibrate() to obtain updated thresholds. The thresholds can be exported
    as a JSON string for consumption by the PRECHECK gate.

    All state is in-process (pure Python collections, no external deps).
    """

    def __init__(self, target_success_rate: float = _TARGET_SUCCESS_RATE) -> None:
        """
        Args:
            target_success_rate: Minimum success rate a threshold must achieve.
                                 Defaults to 0.80 (80 %).
        """
        self._target_success_rate = target_success_rate

        # task_type -> [(confidence, success)]
        self._outcomes: Dict[str, List[Tuple[float, bool]]] = defaultdict(list)

        # model_name -> {"successes": int, "failures": int, "total": int}
        self._model_outcomes: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"successes": 0, "failures": 0, "total": 0}
        )

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        task_type: str,
        confidence: float,
        success: bool,
        model_used: str,
    ) -> None:
        """Record the outcome of a single vision action.

        Args:
            task_type:  Logical task category (e.g. "click", "ui_element_detection").
            confidence: The confidence score produced by the model (0.0–1.0).
            success:    Whether the action actually succeeded.
            model_used: Name/ID of the vision model that produced the prediction.
        """
        self._outcomes[task_type].append((confidence, success))

        bucket = self._model_outcomes[model_used]
        bucket["total"] += 1
        if success:
            bucket["successes"] += 1
        else:
            bucket["failures"] += 1

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self, task_type: str) -> Dict[str, int]:
        """Return aggregate counts for a task type.

        Returns:
            {"total": int, "successes": int, "failures": int}
        """
        outcomes = self._outcomes.get(task_type, [])
        successes = sum(1 for _, s in outcomes if s)
        return {
            "total": len(outcomes),
            "successes": successes,
            "failures": len(outcomes) - successes,
        }

    def success_rate(self, task_type: str) -> float:
        """Return the overall success rate for a task type (0.0–1.0).

        Returns 0.0 if no outcomes have been recorded.
        """
        stats = self.get_stats(task_type)
        if stats["total"] == 0:
            return 0.0
        return stats["successes"] / stats["total"]

    def get_model_stats(self) -> Dict[str, Dict[str, int]]:
        """Return per-model success/failure/total counts.

        Returns:
            {model_name: {"successes": int, "failures": int, "total": int}, ...}
        """
        # Return plain dicts (not defaultdict) to keep the public API stable.
        return {name: dict(counts) for name, counts in self._model_outcomes.items()}

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self) -> Dict[str, float]:
        """Compute per-task-type confidence thresholds.

        Algorithm (simple sorted-cutoff, no external ML):
        1. Sort outcomes for the task type by confidence descending.
        2. Walk from the highest confidence down, accumulating a running
           success rate over the subset seen so far.
        3. Stop at the first confidence value where the running success rate
           drops below _target_success_rate; the threshold is the confidence
           of that last still-acceptable sample.
        4. If all samples exceed the target (or there is only one distinct
           confidence band), return the minimum confidence in the dataset so
           the gate is as permissive as the data supports.

        Returns:
            Dict mapping task_type -> threshold float.  Empty if no data.
        """
        thresholds: Dict[str, float] = {}

        for task_type, outcomes in self._outcomes.items():
            if not outcomes:
                continue

            # Sort by confidence descending so we process highest first.
            sorted_outcomes = sorted(outcomes, key=lambda x: x[0], reverse=True)

            best_threshold = sorted_outcomes[-1][0]  # default: use minimum
            cumulative_successes = 0

            for i, (confidence, success) in enumerate(sorted_outcomes):
                if success:
                    cumulative_successes += 1
                total_so_far = i + 1
                running_rate = cumulative_successes / total_so_far

                if running_rate >= self._target_success_rate:
                    # This confidence level still meets the target — record it
                    # as a candidate threshold.
                    best_threshold = confidence
                # If it drops below the target, we stop updating best_threshold
                # (earlier iterations already captured the last good value).

            thresholds[task_type] = best_threshold

        return thresholds

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_thresholds_json(self) -> str:
        """Serialise calibrated thresholds to a JSON string.

        Schema:
        {
            "thresholds":    {task_type: float, ...},
            "model_routing": {model_name: {"successes": int, "failures": int,
                                           "total": int, "success_rate": float}, ...},
            "calibrated_at": "<ISO-8601 UTC timestamp>"
        }
        """
        thresholds = self.calibrate()

        model_routing: Dict[str, Dict] = {}
        for model_name, counts in self.get_model_stats().items():
            total = counts["total"]
            rate = counts["successes"] / total if total > 0 else 0.0
            model_routing[model_name] = {**counts, "success_rate": round(rate, 4)}

        payload = {
            "thresholds": thresholds,
            "model_routing": model_routing,
            "calibrated_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        return json.dumps(payload, indent=2)
