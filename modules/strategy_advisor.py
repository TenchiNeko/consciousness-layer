"""
Strategy Advisor — Experience-Based Approach Selection.

v2.0: Consciousness-inspired strategy layer.

Before each task, the advisor queries the performance journal to:
  1. Estimate task complexity from goal description
  2. Look up historically best approaches for that complexity level
  3. Estimate expected token budget based on past similar tasks
  4. Generate a strategy context that gets injected into the planner prompt

This is the system's "intuition" — accumulated experience compressed
into approach preferences that bias future decisions. It doesn't
override the planner, it influences it, the same way a human's gut
feeling shapes their approach before they start reasoning explicitly.

The advisor also generates a post-task reflection that the scorer
uses to calibrate future recommendations. Over time, the advisor
gets better at predicting what will work for different task types.
"""

import json
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from performance_scorer import PerformanceScorer, estimate_complexity

logger = logging.getLogger(__name__)


@dataclass
class StrategyRecommendation:
    """Pre-task recommendation from the advisor."""
    complexity_estimate: float
    complexity_bucket: str
    recommended_approach: str
    confidence: float
    expected_tokens: int
    expected_quality: float
    max_iterations_suggestion: int
    reasoning: str
    historical_context: str  # Injected into planner prompt

    def to_context_string(self) -> str:
        """Format as injectable context for the planner agent."""
        parts = [
            "## Strategy Advisor Context (from performance history)",
            "",
            f"Estimated complexity: {self.complexity_bucket} ({self.complexity_estimate:.1f}/10)",
        ]

        if self.recommended_approach:
            parts.append(
                f"Recommended approach: {self.recommended_approach} "
                f"(confidence: {self.confidence:.0%})"
            )

        if self.expected_tokens > 0:
            parts.append(
                f"Expected effort: ~{self.expected_tokens:,} tokens "
                f"(based on {self.complexity_bucket} task history)"
            )

        if self.reasoning:
            parts.append(f"Advisory note: {self.reasoning}")

        if self.historical_context:
            parts.append("")
            parts.append(self.historical_context)

        parts.append("")
        return "\n".join(parts)


class StrategyAdvisor:
    """
    Pre-task strategy advisor powered by the performance journal.

    Queries historical performance data to recommend approaches,
    estimate effort, and inject strategic context into planning.
    """

    def __init__(self, scorer: PerformanceScorer):
        self.scorer = scorer

    def advise(
        self,
        goal: str,
        source_file_count: int = 0,
        task_level: int = 0,
        max_iterations_default: int = 3,
    ) -> StrategyRecommendation:
        """
        Generate a strategy recommendation for an upcoming task.

        This runs BEFORE the planner and its output gets injected
        into the planner's context as advisory information.
        """
        # 1. Estimate complexity
        complexity = estimate_complexity(goal, source_file_count, task_level)
        bucket = PerformanceScorer._complexity_bucket(complexity)

        # 2. Look up preferred approach
        approach_data = self.scorer.get_preferred_approach(complexity)
        recommended_approach = ""
        approach_confidence = 0.0
        if approach_data:
            recommended_approach = approach_data["recommended"]
            approach_confidence = approach_data["confidence"]

        # 3. Get effort estimate
        effort_data = self.scorer.get_effort_estimate(complexity)
        expected_tokens = 0
        expected_quality = 0.0
        if effort_data:
            expected_tokens = effort_data["expected_tokens"]
            expected_quality = effort_data["expected_quality"]

        # 4. Suggest max iterations based on complexity
        if task_level > 0:
            # Benchmark task — use level-based suggestion
            if task_level <= 3:
                suggested_iters = 3
            elif task_level <= 5:
                suggested_iters = 5
            else:
                suggested_iters = max_iterations_default
        elif complexity <= 3.0:
            suggested_iters = 2
        elif complexity <= 6.0:
            suggested_iters = 3
        else:
            suggested_iters = max(max_iterations_default, 4)

        # 5. Generate reasoning
        reasoning = self._generate_reasoning(
            complexity, bucket, approach_data, effort_data
        )

        # 6. Build historical context string
        historical = self._build_historical_context(complexity, bucket)

        recommendation = StrategyRecommendation(
            complexity_estimate=complexity,
            complexity_bucket=bucket,
            recommended_approach=recommended_approach,
            confidence=approach_confidence,
            expected_tokens=expected_tokens,
            expected_quality=expected_quality,
            max_iterations_suggestion=suggested_iters,
            reasoning=reasoning,
            historical_context=historical,
        )

        logger.info(
            f"🧭 Strategy: complexity={bucket} ({complexity:.1f}), "
            f"approach={recommended_approach or 'no preference'}, "
            f"est_tokens={expected_tokens:,}"
        )

        return recommendation

    def _generate_reasoning(
        self,
        complexity: float,
        bucket: str,
        approach_data: Optional[Dict],
        effort_data: Optional[Dict],
    ) -> str:
        """Generate human-readable reasoning for the recommendation."""
        parts = []

        # Performance trend
        trend = self.scorer.get_performance_trend()
        if trend.get("trend") == "improving":
            parts.append(
                f"System performance trending upward "
                f"(+{trend.get('improvement_pct', 0):.0f}% over recent tasks)."
            )
        elif trend.get("trend") == "declining":
            parts.append(
                "System performance declining — consider more conservative approach."
            )

        # Approach confidence
        if approach_data and approach_data["confidence"] > 0.5:
            reinforcement = approach_data["reinforcement"]
            if reinforcement > 0.3:
                parts.append(
                    f"'{approach_data['recommended']}' has strong positive history "
                    f"for {bucket} tasks (signal: {reinforcement:+.2f})."
                )
            elif reinforcement < -0.3:
                # Recommend AGAINST the top approach if it's negative
                parts.append(
                    f"Warning: best-known approach for {bucket} tasks still has "
                    f"negative signal ({reinforcement:+.2f}). Consider experimenting."
                )

        # Effort calibration
        if effort_data and effort_data["confidence"] > 0.3:
            parts.append(
                f"Historical {bucket} tasks average {effort_data['expected_tokens']:,} tokens "
                f"at {effort_data['expected_quality']:.0%} quality."
            )

        return " ".join(parts) if parts else ""

    def _build_historical_context(self, complexity: float, bucket: str) -> str:
        """
        Build a context string with relevant lessons from the journal.

        This gets injected into the planner prompt so historical
        performance directly influences future planning.
        """
        parts = []

        # Get approach preferences for this complexity level
        approach_data = self.scorer.get_preferred_approach(complexity)
        if approach_data and approach_data.get("alternatives"):
            parts.append("### Approach History (this complexity level)")
            parts.append(f"- Best: {approach_data['recommended']} "
                        f"(reinforcement: {approach_data['reinforcement']:+.2f})")
            for alt in approach_data["alternatives"]:
                parts.append(f"- Alt: {alt['approach']} "
                           f"(reinforcement: {alt['reinforcement']:+.2f}, "
                           f"n={alt['samples']})")

        # Get effort curves
        effort_data = self.scorer.get_effort_estimate(complexity)
        if effort_data:
            parts.append(f"\n### Expected Effort Profile")
            parts.append(f"- Tokens: ~{effort_data['expected_tokens']:,}")
            parts.append(f"- Quality: {effort_data['expected_quality']:.0%}")
            parts.append(f"- Efficiency ratio: {effort_data['expected_ratio']:.2f}")

        return "\n".join(parts) if parts else ""

    def post_task_reflect(
        self,
        recommendation: StrategyRecommendation,
        actual_tokens: int,
        actual_quality: float,
        actual_iterations: int,
        success: bool,
    ) -> Dict[str, Any]:
        """
        Post-task reflection: compare prediction vs reality.

        This helps calibrate future recommendations.
        """
        reflection = {
            "predicted_complexity": recommendation.complexity_bucket,
            "predicted_tokens": recommendation.expected_tokens,
            "actual_tokens": actual_tokens,
            "predicted_quality": recommendation.expected_quality,
            "actual_quality": actual_quality,
            "token_prediction_error": (
                abs(actual_tokens - recommendation.expected_tokens)
                / max(recommendation.expected_tokens, 1)
                if recommendation.expected_tokens > 0 else None
            ),
            "quality_prediction_error": (
                abs(actual_quality - recommendation.expected_quality)
                if recommendation.expected_quality > 0 else None
            ),
            "iterations_predicted": recommendation.max_iterations_suggestion,
            "iterations_actual": actual_iterations,
            "success": success,
        }

        if reflection["token_prediction_error"] is not None:
            accuracy = 1.0 - min(reflection["token_prediction_error"], 1.0)
            logger.info(
                f"🔮 Prediction accuracy: tokens={accuracy:.0%}, "
                f"quality_delta={reflection.get('quality_prediction_error', 'N/A')}"
            )

        return reflection
