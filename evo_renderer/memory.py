from __future__ import annotations


class EvolutionMemory:
    def __init__(self):
        self.champions: list[dict] = []

    def save(self, genome, score: float) -> None:
        self.champions.append({"genome": genome, "score": float(score)})
        self.champions.sort(key=lambda x: x["score"], reverse=True)
        self.champions = self.champions[:100]

