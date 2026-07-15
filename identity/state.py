"""Visual Identity — persistent state model.

This is NOT human consciousness. It is a software model of persistence,
preference, memory, and adaptive behaviour.

The identity genome controls HOW the visual genome evolves, separate from
what the visual genome controls (how it looks).
"""

from __future__ import annotations

import copy
import json
import os
import random
import time
from typing import Any

from .mood import Mood
from .goals import Goals
from .preferences import Preferences
from .memory import Memory


class IdentityGenome:
    """Controls how the visual genome evolves.

    Unlike the visual genome (warp, noise, feedback values), the identity
    genome encodes behavioural parameters:

      - curiosity:      openness to unknown mutations
      - stability:      preference for proven organisms
      - complexity:     drive toward more modules / depth
      - aggression:     tendency toward disruptive changes
      - energy:         mutation frequency multiplier
      - conservatism:   resistance to major anatomical changes
    """

    def __init__(self, state: dict[str, float] | None = None):
        defaults = {
            "curiosity": 0.6,
            "stability": 0.5,
            "complexity": 0.5,
            "aggression": 0.2,
            "energy": 0.5,
            "conservatism": 0.3,
        }
        self.params: dict[str, float] = dict(defaults)
        if isinstance(state, dict):
            for k, v in state.items():
                if k in defaults:
                    try:
                        self.params[k] = max(0.0, min(1.0, float(v)))
                    except (TypeError, ValueError):
                        pass

    def mutate(self, strength: float = 0.08) -> None:
        key = random.choice(list(self.params.keys()))
        self.params[key] = float(self.params[key] + random.uniform(-strength, strength))
        self.clamp()

    def clamp(self) -> None:
        for k in self.params:
            self.params[k] = max(0.0, min(1.0, float(self.params[k])))

    def to_dict(self) -> dict[str, float]:
        return dict(self.params)

    def __repr__(self) -> str:
        return f"IdentityGenome({self.params})"


class SpeciesLineage:
    """Simple lineage tracker for archived species.

    Each species entry records the genome snapshot, score, parent,
    and descendant count.
    """

    def __init__(self, state_path: str | None = None):
        self.state_path = state_path or os.path.join(
            os.path.dirname(__file__), "lineages.json"
        )
        self.species: list[dict[str, Any]] = []
        self._load()

    def record(self, genome: Any, score: float, parent: Any | None = None) -> None:
        entry = {
            "ts": time.time(),
            "score": float(score),
            "genome": self._genome_snapshot(genome),
            "parent": self._genome_snapshot(parent) if parent is not None else None,
            "descendants": 0,
        }
        self.species.append(entry)
        self._prune()
        self._save()

    def _genome_snapshot(self, genome: Any) -> dict[str, Any]:
        return {
            "modules": list(getattr(genome, "modules", [])),
            "params": dict(getattr(genome, "params", {})),
        }

    def _prune(self, max_entries: int = 500) -> None:
        if len(self.species) > max_entries:
            self.species = self.species[-max_entries:]

    def top(self, n: int = 10) -> list[dict[str, Any]]:
        return sorted(self.species, key=lambda s: s["score"], reverse=True)[:n]

    def lineage_by_modules(self, modules: list[str]) -> list[dict[str, Any]]:
        mod_set = set(modules)
        return [
            s
            for s in self.species
            if set(s["genome"].get("modules", [])) & mod_set
        ]

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            tmp = self.state_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.species, f, indent=2, default=str)
            os.replace(tmp, self.state_path)
        except Exception:
            pass

    def _load(self) -> None:
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self.species = data
        except Exception:
            pass


class VisualIdentity:
    """Core persistent identity model.

    Holds:
      - age:                     number of evolution cycles survived
      - mood:                    dynamic control signal
      - identity_genome:         behavioural genome (how it evolves)
      - goals:                   long-term competing objectives
      - preferences:             learned likes/dislikes
      - memory:                  timestamped event recall
      - lineage:                 self-curated archive of successful species
    """

    def __init__(
        self,
        state_path: str | None = None,
        max_memory_events: int = 10000,
    ):
        self.state_path = state_path or os.path.join(
            os.path.dirname(__file__), "identity_state.json"
        )
        self.age: int = 0
        self.mood = Mood()
        self.identity_genome = IdentityGenome()
        self.goals = Goals()
        self.preferences = Preferences()
        self.memory = Memory(max_events=max_memory_events)
        self.lineage = SpeciesLineage()
        self.birth_ts = time.time()
        self.season: str = "birth"
        self._dream_pending: list[dict[str, Any]] = []
        self._load()

    def tick(self, delta: float, audio: Any | None = None) -> None:
        self.age += 1
        self._advance_season()
        self.memory.remember(
            "tick",
            {
                "age": self.age,
                "season": self.season,
                "mood": self.mood.to_dict(),
            },
        )

    def record_experience(self, mutation_id: str, before: float, after: float) -> None:
        delta = float(after) - float(before)
        self.memory.remember(
            "mutation",
            {
                "mutation": mutation_id,
                "before": before,
                "after": after,
                "delta": delta,
            },
        )
        self.mood.apply_feedback(delta)
        self.goals.evolve(self.mood, delta)
        self.preferences.update_from_mutation(mutation_id, delta)
        self._update_dream_pending(delta)

    def record_modules(self, modules: list[str], delta: float) -> None:
        self.preferences.update_from_modules(modules, delta)

    def apply_human_feedback(self, sentiment: str) -> None:
        self.mood.apply_human_sentiment(sentiment)
        self.memory.remember("human_feedback", {"sentiment": sentiment})

    def apply_silence(self, duration_secs: float) -> None:
        self.mood.apply_silence(duration_secs)
        self.memory.remember("silence", {"duration": duration_secs})

    def mutation_strength(self) -> float:
        return self.mood.mutation_strength()

    def exploration_probability(self) -> float:
        return self.mood.exploration_probability()

    def prefer_known_species(self) -> bool:
        return self.mood.prefer_known_species()

    def identity_mutation_strength(self) -> float:
        return 0.04 + 0.06 * self.mood.get("curiosity")

    def should_explore_unknown(self) -> bool:
        return random.random() < self.exploration_probability()

    def score_candidate(self, genome: Any, audio: Any | None = None) -> float:
        components: dict[str, float] = {}
        components["discovery"] = self.goals.get("discover_new_species") * 0.3
        components["stability"] = self.goals.get("maintain_stability") * 0.25
        components["complexity"] = self.goals.get("increase_complexity") * 0.25
        components["energy"] = self.goals.get("save_energy") * 0.2

        modules = getattr(genome, "modules", [])
        if modules:
            complexity_bonus = min(1.0, len(modules) / 6.0)
            components["complexity"] += 0.1 * complexity_bonus * self.goals.get("increase_complexity")

        for mod in modules:
            op = float(self.preferences.opinion(mod))
            components["preference"] = components.get("preference", 0.0) + op * 0.05

        return sum(components.values())

    def dream(self, renderer: Any, max_steps: int = 20) -> list[dict[str, Any]]:
        dreams: list[dict[str, Any]] = []
        try:
            base_genome = getattr(renderer, "genome", None)
            if base_genome is None:
                return dreams

            for _ in range(max_steps):
                candidate = copy.deepcopy(base_genome)
                candidate.mutate(strength=self.mutation_strength() * 1.5)
                score = self.score_candidate(candidate)
                dreams.append(
                    {
                        "genome": candidate,
                        "score": score,
                    }
                )

            dreams.sort(key=lambda d: d["score"], reverse=True)
            promising = dreams[:3]
            self._dream_pending.extend(promising)
            self.memory.remember(
                "dream",
                {
                    "simulated": len(dreams),
                    "promising_score": promising[0]["score"] if promising else 0.0,
                },
            )
        except Exception:
            pass
        return dreams

    def consume_dreams(self) -> list[dict[str, Any]]:
        pending = list(self._dream_pending)
        self._dream_pending.clear()
        return pending

    def record_champion(self, genome: Any, score: float, parent: Any | None = None) -> None:
        self.lineage.record(genome, score, parent)
        self.memory.remember(
            "champion",
            {
                "score": score,
                "modules": getattr(genome, "modules", []),
            },
        )
        self.goals.reward("discover_new_species", score)
        if score > 2.0:
            self.goals.reward("increase_complexity", score * 0.5)
        self.goals.clip()

    def _advance_season(self) -> None:
        age = self.age
        if age < 100:
            self.season = "birth"
        elif age < 500:
            self.season = "growth"
        elif age < 2000:
            self.season = "maturity"
        else:
            self.season = "archive"

    def _update_dream_pending(self, delta: float) -> None:
        if delta < -0.1 and len(self._dream_pending) < 5:
            try:
                self._dream_pending.append({"reason": "recovery_dream", "delta": delta})
            except Exception:
                pass

    def save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            payload = {
                "age": self.age,
                "birth_ts": self.birth_ts,
                "season": self.season,
                "mood": self.mood.to_dict(),
                "identity_genome": self.identity_genome.to_dict(),
                "goals": self.goals.to_dict(),
            }
            tmp = self.state_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, self.state_path)
            self.preferences.save()
        except Exception:
            pass

    def _load(self) -> None:
        try:
            if not os.path.exists(self.state_path):
                return
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return
            self.age = int(data.get("age", 0))
            self.birth_ts = float(data.get("birth_ts", time.time()))
            self.season = str(data.get("season", "birth"))
            mood = data.get("mood")
            if isinstance(mood, dict):
                self.mood = Mood(state=mood)
            ig = data.get("identity_genome")
            if isinstance(ig, dict):
                self.identity_genome = IdentityGenome(state=ig)
            goals = data.get("goals")
            if isinstance(goals, dict):
                self.goals = Goals(state=goals)
        except Exception:
            pass

    def summary(self) -> dict[str, Any]:
        return {
            "age": self.age,
            "season": self.season,
            "mood": self.mood.to_dict(),
            "identity_genome": self.identity_genome.to_dict(),
            "goals": self.goals.to_dict(),
            "preferences": self.preferences.to_dict(),
            "memory": self.memory.summary(),
        }

    def __repr__(self) -> str:
        return f"VisualIdentity(age={self.age}, season={self.season}, mood={self.mood})"
