"""Identity engine — glues VisualIdentity into the renderer/evolution pipeline.
This is the entry point for Stage 12 integration. The EvolutionEngine creates
an IdentityEngine and ticks it each frame. The identity engine then feeds
behavioural parameters back into the evolution loop.
"""

from __future__ import annotations
from typing import Any
from .state import VisualIdentity, IdentityGenome


class IdentityEngine:
    """Bridge between the renderer's evolution loop and the persistent identity.
    Responsibilities:
      - Tick identity (age, mood, goals, memory) each frame.
      - Translate identity mood/goals into mutation parameters.
      - Run dream simulations during silence/idle.
      - Expose lineage and preference summaries.
      - Persist identity state periodically.
    """

    def __init__(
        self,
        renderer: Any,
        state_path: str | None = None,
        max_memory_events: int = 10000,
        dream_interval_secs: float = 8.0,
        dream_max_steps: int = 20,
        persist_interval_secs: float = 30.0,
    ):
        self.renderer = renderer
        self.identity = VisualIdentity(
            state_path=state_path,
            max_memory_events=max_memory_events,
        )
        self.dream_interval_secs = float(dream_interval_secs)
        self.dream_max_steps = int(dream_max_steps)
        self.persist_interval_secs = float(persist_interval_secs)
        self._dream_timer = 0.0
        self._persist_timer = 0.0
        self._silence_accum = 0.0
        self.silence_threshold = 0.08

    def update(self, dt: float, audio: Any | None = None) -> dict[str, Any]:
        self.identity.tick(dt, audio=audio)

        if audio is not None:
            avg_energy = (
                float(getattr(audio, "bass", 0.0))
                + float(getattr(audio, "mid", 0.0))
                + float(getattr(audio, "treble", 0.0))
            ) / 3.0
        else:
            avg_energy = 0.0

        if avg_energy < self.silence_threshold:
            self._silence_accum += float(dt)
        else:
            if self._silence_accum > 0.0:
                self.identity.apply_silence(self._silence_accum)
            self._silence_accum = 0.0

        if self._silence_accum >= self.dream_interval_secs:
            self._run_dream()

        self._dream_timer += float(dt)
        self._persist_timer += float(dt)
        if self._persist_timer >= self.persist_interval_secs:
            self._persist_timer = 0.0
            self.identity.save()

        return self.identity.summary()

    def record_mutation(self, mutation_id: str, before: float, after: float) -> None:
        self.identity.record_experience(mutation_id, before, after)

    def record_champion(self, genome: Any, score: float, parent: Any | None = None) -> None:
        self.identity.record_champion(genome, score, parent)

    def record_modules(self, modules: list[str], delta: float) -> None:
        self.identity.record_modules(modules, delta)

    def human_feedback(self, sentiment: str) -> None:
        self.identity.apply_human_feedback(sentiment)

    def identity_mutation_strength(self) -> float:
        return self.identity.identity_mutation_strength()

    def visual_mutation_strength(self) -> float:
        return self.identity.mutation_strength()

    def exploration_probability(self) -> float:
        return self.identity.exploration_probability()

    def prefer_known_species(self) -> bool:
        return self.identity.prefer_known_species()

    def should_explore_unknown(self) -> bool:
        return self.identity.should_explore_unknown()

    def identity_genome(self) -> IdentityGenome:
        return self.identity.identity_genome

    def mutate_identity_genome(self) -> None:
        self.identity.identity_genome.mutate(
            strength=self.identity.identity_mutation_strength()
        )

    def consume_dreams(self) -> list[dict[str, Any]]:
        return self.identity.consume_dreams()

    def lineage_summary(self, top_n: int = 10) -> list[dict[str, Any]]:
        return self.identity.lineage.top(n=top_n)

    def summary(self) -> dict[str, Any]:
        return self.identity.summary()

    def save(self) -> None:
        self.identity.save()

    def _run_dream(self) -> None:
        try:
            self.identity.dream(self.renderer, max_steps=self.dream_max_steps)
        except Exception:
            pass
        finally:
            self._silence_accum = 0.0
