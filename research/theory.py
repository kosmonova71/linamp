from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


@dataclass
class TheoryRule:
    """An emergent if-then rule discovered through experiment analysis."""

    condition: dict[str, Any]
    action: dict[str, Any]
    confidence: float = 0.0
    evidence_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": dict(self.condition),
            "action": dict(self.action),
            "confidence": self.confidence,
            "evidence_count": self.evidence_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TheoryRule:
        return cls(
            condition=dict(data.get("condition", {})),
            action=dict(data.get("action", {})),
            confidence=float(data.get("confidence", 0.0)),
            evidence_count=int(data.get("evidence_count", 0)),
        )


class Theory:
    """A synthesized body of emergent visual laws.

    Theories are built from observed correlations across many experiments.
    They can influence future evolution by recommending mutation parameters.
    """

    def __init__(self, name: str):
        self.name = name
        self.rules: list[TheoryRule] = []

    def add_rule(
        self,
        condition: dict[str, Any],
        action: dict[str, Any],
        confidence: float = 0.5,
    ) -> None:
        existing = [r for r in self.rules if r.condition == condition and r.action == action]
        if existing:
            rule = existing[0]
            rule.confidence = min(1.0, rule.confidence + 0.05)
            rule.evidence_count += 1
        else:
            self.rules.append(
                TheoryRule(
                    condition=condition,
                    action=action,
                    confidence=min(1.0, confidence),
                    evidence_count=1,
                )
            )

    def update_confidence(self, condition: dict[str, Any], action: dict[str, Any], reward: float) -> None:
        for rule in self.rules:
            if rule.condition == condition and rule.action == action:
                rule.confidence = max(0.0, min(1.0, rule.confidence + reward * 0.02))
                rule.evidence_count += 1
                return
        self.add_rule(condition, action, confidence=max(0.0, min(1.0, reward * 0.05)))

    def recommend(self, audio: dict[str, float], genome_params: dict[str, float]) -> dict[str, Any] | None:
        """Given current audio + genome params, return recommended mutation params.

        Returns a dict of recommended changes, or None if no rule matches.
        """
        best_rule = None
        best_score = 0.0
        bass = float(audio.get("bass", 0.0))
        treble = float(audio.get("treble", 0.0))

        for rule in self.rules:
            if rule.confidence <= 0.0:
                continue
            cond = rule.condition
            match = True
            for key, threshold in cond.items():
                if key == "bass_gt" and not (bass > float(threshold)):
                    match = False
                    break
                if key == "treble_lt" and not (treble < float(threshold)):
                    match = False
                    break
                if key == "treble_gt" and not (treble > float(threshold)):
                    match = False
                    break
                if key == "bass_lt" and not (bass < float(threshold)):
                    match = False
                    break
                if key == "complexity_lt" and not (len(genome_params) < int(threshold)):
                    match = False
                    break
                if key == "complexity_gt" and not (len(genome_params) > int(threshold)):
                    match = False
                    break
            if match:
                score = rule.confidence * rule.evidence_count
                if score > best_score:
                    best_score = score
                    best_rule = rule

        if best_rule is None:
            return None
        return dict(best_rule.action)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rules": [r.to_dict() for r in self.rules],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Theory:
        theory = cls(name=str(data.get("name", "")))
        for rule_data in data.get("rules", []):
            theory.rules.append(TheoryRule.from_dict(rule_data))
        return theory

    @property
    def stable_id(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
