"""Internal mood model.
Mood is a control signal, not human consciousness.
It influences mutation rates, exploration, and evolution behaviour.
"""

from __future__ import annotations

class Mood:
    """Dynamic mood state that acts as a control signal for evolution.
    Mood dimensions:
      energy:       willingness to mutate actively
      curiosity:    openness to unknown shaders / new physics
      stability:    preference for proven, successful organisms
      aggression:   tendency toward disruptive / chaotic mutations

    Mood shifts based on recent experience (fitness deltas, silence, user feedback).
    """

    DEFAULT = {
        "energy": 0.5,
        "curiosity": 0.6,
        "stability": 0.5,
        "aggression": 0.2,
    }

    def __init__(self, state: dict[str, float] | None = None):
        self.state: dict[str, float] = dict(self.DEFAULT)
        if isinstance(state, dict):
            for k, v in state.items():
                if k in self.DEFAULT:
                    try:
                        self.state[k] = max(0.0, min(1.0, float(v)))
                    except (TypeError, ValueError):
                        pass

    def get(self, key: str) -> float:
        return float(self.state.get(key, 0.0))

    def set(self, key: str, value: float) -> None:
        if key in self.DEFAULT:
            self.state[key] = max(0.0, min(1.0, float(value)))

    def shift(self, key: str, delta: float) -> None:
        self.set(key, self.get(key) + float(delta))

    def clip(self) -> None:
        for k in self.state:
            self.state[k] = max(0.0, min(1.0, float(self.state[k])))

    def to_dict(self) -> dict[str, float]:
        return dict(self.state)

    def __repr__(self) -> str:
        return f"Mood({self.state})"

    def apply_feedback(self, delta: float) -> None:
        if delta > 0.0:
            self.shift("energy", 0.05)
            self.shift("curiosity", 0.03)
        else:
            self.shift("stability", 0.04)
            self.shift("aggression", abs(delta) * 0.5)
        self.clip()

    def apply_silence(self, duration_secs: float) -> None:
        self.shift("energy", -0.02 * min(duration_secs, 10.0))
        self.shift("curiosity", 0.01 * min(duration_secs, 20.0))
        self.clip()

    def apply_human_sentiment(self, sentiment: str) -> None:
        if sentiment == "more_alive":
            self.shift("energy", 0.15)
            self.shift("curiosity", 0.10)
            self.shift("aggression", -0.10)
        elif sentiment == "calmer":
            self.shift("stability", 0.20)
            self.shift("energy", -0.10)
            self.shift("aggression", -0.15)
        elif sentiment == "more_chaos":
            self.shift("aggression", 0.25)
            self.shift("curiosity", 0.10)
            self.shift("stability", -0.10)
        self.clip()

    def mutation_strength(self) -> float:
        energy = self.get("energy")
        curiosity = self.get("curiosity")
        stability = self.get("stability")
        base = 0.08
        return base + 0.12 * energy + 0.08 * curiosity - 0.06 * stability

    def exploration_probability(self) -> float:
        return max(0.0, min(1.0, self.get("curiosity") * 0.7 + 0.1))

    def prefer_known_species(self) -> bool:
        return self.get("stability") > 0.65
