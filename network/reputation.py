from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Reputation:
    score: float
    fitness: float
    novelty: float
    stability: float
    audio_response: float
    runtime_survival: float
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": float(self.score),
            "fitness": float(self.fitness),
            "novelty": float(self.novelty),
            "stability": float(self.stability),
            "audio_response": float(self.audio_response),
            "runtime_survival": float(self.runtime_survival),
            "status": str(self.status),
        }


class ReputationScorer:
    """Reputation gate for which genetics should spread."""

    def __init__(
        self,
        *,
        weights: dict[str, float] | None = None,
        global_threshold: float = 80.0,
    ):
        self.weights = weights or {
            "fitness": 0.35,
            "novelty": 0.25,
            "stability": 0.20,
            "audio_response": 0.10,
            "runtime_survival": 0.10,
        }
        self.global_threshold = float(global_threshold)

    @staticmethod
    def _clamp01(x: float) -> float:
        x = float(x)
        return max(0.0, min(1.0, x))

    def score(self, *, fitness: float, novelty: float, stability: float, audio_response: float, runtime_survival: float) -> Reputation:
        fitness_n = self._clamp01(fitness / 100.0)  # project fitness is ~0..100 in examples
        novelty_n = self._clamp01(novelty)
        stability_n = self._clamp01(stability)
        audio_n = self._clamp01(audio_response)
        survival_n = self._clamp01(runtime_survival)

        w = self.weights
        total_w = float(sum(w.values())) or 1.0

        raw = (
            fitness_n * float(w.get("fitness", 0.0))
            + novelty_n * float(w.get("novelty", 0.0))
            + stability_n * float(w.get("stability", 0.0))
            + audio_n * float(w.get("audio_response", 0.0))
            + survival_n * float(w.get("runtime_survival", 0.0))
        ) / total_w

        # Convert to 0..100 scale
        score_100 = float(raw) * 100.0

        status = "GLOBAL SPECIES" if score_100 >= self.global_threshold else "LOCAL SPECIES"

        return Reputation(
            score=score_100,
            fitness=float(fitness),
            novelty=float(novelty),
            stability=float(stability),
            audio_response=float(audio_response),
            runtime_survival=float(runtime_survival),
            status=status,
        )

