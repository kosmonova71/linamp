from __future__ import annotations

import time
from typing import Any

from .hypothesis import Hypothesis
from .observation import Observation


class Experiment:
    """Runs a controlled test of a hypothesis against the current visualizer state.

    Each experiment applies the change described by the hypothesis, evaluates,
    and records the result as an Observation. It does NOT directly draw.
    """

    def __init__(
        self,
        renderer: Any,
        evolution_engine: Any,
        hypothesis: Hypothesis,
        max_runs: int = 50,
    ):
        self.renderer = renderer
        self.evolution = evolution_engine
        self.hypothesis = hypothesis
        self.max_runs = int(max_runs)
        self.observations: list[Observation] = []

    def run(self, audio: Any) -> Observation | None:
        """Execute one controlled trial of the hypothesis.

        Returns the Observation, or None if the trial could not be performed.
        """
        if self.hypothesis.attempts >= self.max_runs:
            return None

        _evolve = getattr(self.evolution, "_evolve_once", None)
        _score = getattr(self.evolution, "_score_current", None)
        _snapshot = getattr(self.evolution, "_snapshot_genome", None)
        
        guardian = getattr(self.evolution, "guardian", None)
        _apply = getattr(self.evolution, "_apply_brain_mutation", None)

        if _score is None or _snapshot is None:
            return None

        parent = _snapshot(self.renderer.genome)
        before = _score(audio)

        child = _snapshot(parent)
        self._apply_change(child, self.hypothesis.change)

        self.renderer.genome = child
        try:
            self.renderer.evolve_shader()
        except Exception:
            self.renderer.genome = parent
            try:
                self.renderer.evolve_shader()
            except Exception:
                pass
            self.hypothesis.attempts += 1
            return None

        after = _score(audio)
        delta = float(after) - float(before)

        approved = True
        if guardian is not None:
            decision = guardian.approve(
                {"score": after, "delta": delta},
                parent_score=before,
            )
            approved = decision.approved
            if not approved:
                self.renderer.genome = parent
                try:
                    self.renderer.evolve_shader()
                except Exception:
                    pass

        audio_dict: dict[str, float] = {}
        if audio is not None:
            audio_dict = {
                "bass": float(getattr(audio, "bass", 0.0)),
                "mid": float(getattr(audio, "mid", 0.0)),
                "treble": float(getattr(audio, "treble", 0.0)),
            }

        observation = Observation(
            mutation_id=str(self.hypothesis.change),
            before_score=float(before),
            after_score=float(after),
            delta=float(delta),
            audio=audio_dict,
            genome_modules=list(getattr(child, "modules", [])),
            genome_params=dict(getattr(child, "params", {})),
            hypothesis_change=str(self.hypothesis.change),
            timestamp=time.time(),
            approved=approved,
        )
        self.observations.append(observation)
        self.hypothesis.attempts += 1
        if approved:
            self.hypothesis.successes += 1

        return observation

    def _apply_change(self, genome: Any, change: str) -> None:
        """Map a hypothesis change name to a genome mutation.

        This delegates to the evolution engine's mutation dispatcher when
        available, falling back to a direct genome mutate otherwise.
        """
        apply_fn = getattr(self.evolution, "_apply_brain_mutation", None)
        if apply_fn is not None:
            try:
                apply_fn(genome, change)
                return
            except Exception:
                pass

        modules = getattr(genome, "modules", None)
        params = getattr(genome, "params", None)
        mutate_fn = getattr(genome, "mutate", None)

        if change == "increase_feedback":
            if params is not None:
                params["feedback"] = max(
                    0.0,
                    min(
                        0.99,
                        float(params.get("feedback", 0.85)) + 0.03,
                    ),
                )
            if mutate_fn is not None:
                mutate_fn(strength=0.08)
        elif change == "add_fractal":
            if modules is not None and "fractal" not in modules:
                modules.append("fractal")
            if mutate_fn is not None:
                mutate_fn(strength=0.10)
        elif change == "add_particles":
            if modules is not None and "particles" not in modules:
                modules.append("particles")
            if mutate_fn is not None:
                mutate_fn(strength=0.10)
        elif change == "add_noise":
            if modules is not None and "noise" not in modules:
                modules.append("noise")
            if mutate_fn is not None:
                mutate_fn(strength=0.10)
        elif change == "warp_module":
            if modules is not None and "warp" not in modules:
                modules.append("warp")
            if mutate_fn is not None:
                mutate_fn(strength=0.10)
        else:
            if mutate_fn is not None:
                mutate_fn(strength=0.12)

        if modules is not None and len(modules) > 6:
            genome.modules = modules[-6:]

    @property
    def mean_delta(self) -> float:
        if not self.observations:
            return 0.0
        return float(sum(o.delta for o in self.observations)) / float(len(self.observations))

    @property
    def success_rate(self) -> float:
        if not self.observations:
            return 0.0
        count = sum(1 for o in self.observations if o.approved and o.delta > 0.0)
        return float(count) / float(len(self.observations))
