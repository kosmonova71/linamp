"""Learned preferences from past experiences.

Preferences are simple like/dislike records for visual features and mutation types.
They influence future mutation choices and goal weighting.
"""

from __future__ import annotations

import json
import os


class Preferences:
    """Accumulated likes and dislikes for visual features, modules, and mutations."""

    def __init__(self, state_path: str | None = None):
        self.likes: dict[str, float] = {}
        self.dislikes: dict[str, float] = {}
        self.state_path = state_path or os.path.join(
            os.path.dirname(__file__), "preferences.json"
        )
        self._load()

    def record_like(self, feature: str, amount: float = 0.1) -> None:
        self.likes[feature] = float(self.likes.get(feature, 0.0)) + float(amount)

    def record_dislike(self, feature: str, amount: float = 0.1) -> None:
        self.dislikes[feature] = float(self.dislikes.get(feature, 0.0)) + float(amount)

    def opinion(self, feature: str) -> float:
        like = float(self.likes.get(feature, 0.0))
        dislike = float(self.dislikes.get(feature, 0.0))
        return like - dislike

    def liked(self, feature: str, threshold: float = 0.15) -> bool:
        return self.opinion(feature) > threshold

    def disliked(self, feature: str, threshold: float = 0.15) -> bool:
        return self.opinion(feature) < -threshold

    def update_from_mutation(self, mutation_id: str, delta: float) -> None:
        if delta > 0.0:
            self.record_like(mutation_id, delta * 0.05)
        else:
            self.record_dislike(mutation_id, abs(delta) * 0.05)

    def update_from_modules(self, modules: list[str], delta: float) -> None:
        for mod in modules:
            if delta > 0.0:
                self.record_like(mod, delta * 0.03)
            else:
                self.record_dislike(mod, abs(delta) * 0.03)

    def bias_weight(self, mutation_id: str, base: float = 0.5) -> float:
        opinion = self.opinion(mutation_id)
        return base + opinion

    def save(self) -> None:
        try:
            payload = {
                "likes": self.likes,
                "dislikes": self.dislikes,
            }
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            tmp = self.state_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, self.state_path)
        except Exception:
            pass

    def _load(self) -> None:
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    likes = data.get("likes", {})
                    dislikes = data.get("dislikes", {})
                    if isinstance(likes, dict):
                        self.likes = {str(k): float(v) for k, v in likes.items()}
                    if isinstance(dislikes, dict):
                        self.dislikes = {str(k): float(v) for k, v in dislikes.items()}
        except Exception:
            pass

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {
            "likes": dict(self.likes),
            "dislikes": dict(self.dislikes),
        }

    def __repr__(self) -> str:
        return f"Preferences(likes={len(self.likes)}, dislikes={len(self.dislikes)})"
