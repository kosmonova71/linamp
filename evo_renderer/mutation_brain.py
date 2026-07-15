from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


@dataclass
class AudioContext:
    bass: float = 0.0
    mid: float = 0.0
    treble: float = 0.0
    tempo: float = 120.0

    @staticmethod
    def from_audio_state(audio: Any) -> "AudioContext":
        # Current repo AudioState only has bass/mid/treble.
        # Tempo is not available yet, so we default.
        bass = float(getattr(audio, "bass", 0.0))
        mid = float(getattr(audio, "mid", 0.0))
        treble = float(getattr(audio, "treble", 0.0))
        return AudioContext(bass=bass, mid=mid, treble=treble, tempo=120.0)


class MutationBrain:
    """A tiny learning layer (no huge model).

    - Maintains per-mutation weights.
    - Chooses best operator given audio context.
    - Learns via reward() = update weight by improvement.

    Note: Mutation operators are mapped to the existing VisualGenome mutation
    behaviours.
    """

    DEFAULT_WEIGHTS: dict[str, float] = {
        "add_noise": 0.5,
        "add_fractal": 0.5,
        "increase_feedback": 0.5,
        "change_colour": 0.5,
        "warp_module": 0.5,
        "add_particles": 0.5,
        "tune_params": 0.5,
    }

    def __init__(self, state_path: str | None = None):
        self.state_path = state_path or os.path.join(
            os.path.dirname(__file__), "brain_state.json"
        )
        self.weights: dict[str, float] = dict(self.DEFAULT_WEIGHTS)
        self._load()

    def _load(self) -> None:
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    w = data.get("weights", {})
                    if isinstance(w, dict):
                        for k, v in w.items():
                            try:
                                self.weights[k] = float(v)
                            except Exception:
                                pass
        except Exception:
            # Corrupt state: ignore.
            pass

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        payload = {"weights": self.weights}
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def choose(self, audio: Any) -> str:
        ctx = AudioContext.from_audio_state(audio)

        # Apply hard audio priors per your spec (tempo currently unknown -> default 120)
        score_overrides: dict[str, float] = {}

        # Bass low -> prefer fractals / slow feedback
        if ctx.bass < 0.3:
            score_overrides["add_fractal"] = self.weights.get("add_fractal", 0.5) + 0.25
            score_overrides["increase_feedback"] = self.weights.get(
                "increase_feedback", 0.5
            ) - 0.05
            score_overrides["tune_params"] = self.weights.get("tune_params", 0.5) + 0.05

        # Bass high -> prefer particles / rapid distortion / colour mutation
        if ctx.bass > 0.8:
            score_overrides["add_particles"] = self.weights.get("add_particles", 0.5) + 0.25
            score_overrides["change_colour"] = self.weights.get("change_colour", 0.5) + 0.20
            score_overrides["tune_params"] = self.weights.get("tune_params", 0.5) + 0.10

        # Treble high -> more noise/warp
        if ctx.treble > 0.7:
            score_overrides["add_noise"] = self.weights.get("add_noise", 0.5) + 0.20
            score_overrides["warp_module"] = self.weights.get("warp_module", 0.5) + 0.15

        # Tempo rules are placeholders (tempo not provided yet)
        if ctx.tempo < 70:
            score_overrides["add_fractal"] = self.weights.get("add_fractal", 0.5) + 0.25
        if ctx.tempo > 140:
            score_overrides["add_particles"] = self.weights.get("add_particles", 0.5) + 0.20

        # Pick argmax of (weights possibly overridden)
        best_mut = None
        best_val = float("-inf")
        for mut, base_w in self.weights.items():
            val = float(score_overrides.get(mut, base_w))
            if val > best_val:
                best_val = val
                best_mut = mut

        return best_mut or "tune_params"

    def reward(self, mutation_id: str, improvement: float) -> None:
        # improvement is delta fitness.
        # scale down strongly to avoid runaway.
        w = self.weights.get(mutation_id, 0.5)
        w += float(improvement) * 0.01
        # clamp to stable range
        w = max(0.0, min(2.5, w))
        self.weights[mutation_id] = w

    def __repr__(self) -> str:
        return f"MutationBrain(weights={self.weights})"

