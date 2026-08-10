"""Governed ensemble policy composition for execution-time decisions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class Policy(Protocol):
    def predict(self, observation: object) -> int: ...


@dataclass(frozen=True)
class EnsembleDecision:
    action: int
    confidence: float
    votes: tuple[int, ...]
    abstained: bool


class WeightedPolicyEnsemble:
    """Combine independently trained policies with abstention and vote limits."""

    def __init__(
        self,
        policies: tuple[Policy, ...],
        *,
        weights: tuple[float, ...] | None = None,
        min_confidence: float = 0.60,
        neutral_action: int = 0,
    ) -> None:
        if not policies:
            raise ValueError("at least one policy is required")
        self.policies = policies
        self.weights = weights or tuple(1.0 for _ in policies)
        if len(self.weights) != len(policies) or any(weight <= 0 for weight in self.weights):
            raise ValueError("weights must be positive and match policies")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        self.min_confidence = min_confidence
        self.neutral_action = neutral_action

    def predict_with_confidence(self, observation: object) -> EnsembleDecision:
        votes = tuple(int(policy.predict(observation)) for policy in self.policies)
        totals: dict[int, float] = {}
        for vote, weight in zip(votes, self.weights, strict=True):
            totals[vote] = totals.get(vote, 0.0) + weight
        action, winning_weight = max(totals.items(), key=lambda item: (item[1], -item[0]))
        confidence = winning_weight / sum(self.weights)
        abstained = confidence < self.min_confidence
        return EnsembleDecision(
            action=self.neutral_action if abstained else action,
            confidence=confidence,
            votes=votes,
            abstained=abstained,
        )

    def predict(self, observation: object) -> int:
        return self.predict_with_confidence(observation).action
