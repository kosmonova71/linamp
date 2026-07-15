"""Long-term goals with competing weights.

The system balances multiple objectives. Goal weights shift based on
whether the goal is being satisfied or frustrated.
"""

from __future__ import annotations

from identity.mood import Mood


class Goals:
    """Balanced objective set.

    Dimensions:
      discover_new_species:   seek visual novelty
      maintain_stability:     avoid crashing / bad visuals
      increase_complexity:    add modules, depth, detail
      save_energy:            keep GPU cost low
    """

    DEFAULT = {
        "discover_new_species": 0.8,
        "maintain_stability": 0.6,
        "increase_complexity": 0.7,
        "save_energy": 0.4,
    }

    def __init__(self, state: dict[str, float] | None = None):
        self.weights: dict[str, float] = dict(self.DEFAULT)
        if isinstance(state, dict):
            for k, v in state.items():
                if k in self.DEFAULT:
                    try:
                        self.weights[k] = max(0.0, min(1.0, float(v)))
                    except (TypeError, ValueError):
                        pass

    def get(self, key: str) -> float:
        return float(self.weights.get(key, 0.0))

    def set(self, key: str, value: float) -> None:
        if key in self.DEFAULT:
            self.weights[key] = max(0.0, min(1.0, float(value)))

    def shift(self, key: str, delta: float) -> None:
        self.set(key, self.get(key) + float(delta))

    def clip(self) -> None:
        for k in self.weights:
            self.weights[k] = max(0.0, min(1.0, float(self.weights[k])))

    def to_dict(self) -> dict[str, float]:
        return dict(self.weights)

    def dominant(self) -> str | None:
        if not self.weights:
            return None
        return max(self.weights.keys(), key=lambda k: self.weights[k])

    def balanced_score(self) -> float:
        if not self.weights:
            return 0.0
        vals = list(self.weights.values())
        mean = sum(vals) / len(vals)
        variance = sum((v - mean) ** 2 for v in vals) / len(vals)
        return float(mean - 0.3 * variance)

    def reward(self, key: str, improvement: float) -> None:
        if key in self.weights:
            self.shift(key, improvement * 0.02)
            self.clip()

    def penalize(self, key: str, amount: float) -> None:
        if key in self.weights:
            self.shift(key, -amount * 0.02)
            self.clip()

    def evolve(self, mood: "Mood", recent_delta: float) -> None:
        if recent_delta > 0.0:
            self.reward("discover_new_species", recent_delta)
            if mood.get("energy") > 0.7:
                self.shift("increase_complexity", 0.02)
        else:
            self.reward("maintain_stability", abs(recent_delta))
            self.shift("save_energy", 0.01)
        self.clip()

    def __repr__(self) -> str:
        return f"Goals({self.weights})"
