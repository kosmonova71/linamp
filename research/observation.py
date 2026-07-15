from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Observation:
    """Single data point from an experiment run.

    Captures the audio context, genome identity, and fitness delta so the
    researcher can later detect patterns or build theories.
    """

    mutation_id: str
    before_score: float
    after_score: float
    delta: float
    audio: dict[str, float]
    genome_modules: list[str]
    genome_params: dict[str, float]
    hypothesis_change: str = ""
    timestamp: float = 0.0
    approved: bool = False

    @property
    def success(self) -> bool:
        return self.delta > 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mutation_id": self.mutation_id,
            "before_score": self.before_score,
            "after_score": self.after_score,
            "delta": self.delta,
            "audio": dict(self.audio),
            "genome_modules": list(self.genome_modules),
            "genome_params": dict(self.genome_params),
            "hypothesis_change": self.hypothesis_change,
            "timestamp": self.timestamp,
            "approved": self.approved,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Observation:
        return cls(
            mutation_id=str(data.get("mutation_id", "")),
            before_score=float(data.get("before_score", 0.0)),
            after_score=float(data.get("after_score", 0.0)),
            delta=float(data.get("delta", 0.0)),
            audio=dict(data.get("audio", {})),
            genome_modules=list(data.get("genome_modules", [])),
            genome_params=dict(data.get("genome_params", {})),
            hypothesis_change=str(data.get("hypothesis_change", "")),
            timestamp=float(data.get("timestamp", 0.0)),
            approved=bool(data.get("approved", False)),
        )
