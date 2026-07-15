from __future__ import annotations

import json
import os
import random
import time
from typing import Any

from .hypothesis import Hypothesis
from .experiments import Experiment
from .theory import Theory
from .observation import Observation


class Researcher:
    """Stage 11 autonomous visual researcher.

    Sits above the mutation/render/score loop and runs a scientific method:
      observe -> form hypothesis -> create mutation -> run experiment ->
      measure -> update knowledge -> generate new hypothesis

    Does not directly draw. It asks "What changes produce interesting behaviour?"
    and uses controlled experiments to answer.
    """

    def __init__(self, renderer: Any, interval_secs: float = 30.0):
        self.renderer = renderer
        self.interval_secs = float(interval_secs)
        self.timer = 0.0

        self.evolution = getattr(renderer, "evolution", None)

        self.hypotheses: list[Hypothesis] = []
        self.active_experiments: dict[str, Experiment] = {}
        self.theory = Theory(name="root")
        self.observations: list[Observation] = []
        self.failed_hypotheses: list[dict[str, Any]] = []

        knowledge_dir = os.path.join(os.path.dirname(__file__), "..", "knowledge")
        self.knowledge_dir = os.path.abspath(knowledge_dir)
        os.makedirs(self.knowledge_dir, exist_ok=True)

        self.discoveries_path = os.path.join(self.knowledge_dir, "discoveries.json")
        self.laws_path = os.path.join(self.knowledge_dir, "laws.json")
        self.failed_path = os.path.join(self.knowledge_dir, "failed_experiments.json")

        self.curiosity: float = 1.0
        self.cycles: int = 0
        self.max_observations: int = 5000

    def update(self, dt: float, audio: Any | None = None) -> None:
        self.timer += float(dt)
        if self.timer < self.interval_secs:
            self._decay_curiosity(dt)
            return

        self.timer = 0.0
        self.cycles += 1
        self._trim_observations()

        if audio is None:
            audio = getattr(self.renderer, "last_audio_state", None)
        if audio is None:
            return

        self._observe(audio)
        self._form_hypotheses()
        self._schedule_experiments(audio)
        self._run_experiments(audio)
        self._synthesize_theories()
        self._persist()

    def _observe(self, audio: Any) -> None:
        pass

    def _form_hypotheses(self) -> None:
        if not self.observations and not self.hypotheses:
            self._seed_initial_hypotheses()

        if self.hypotheses:
            return

        if len(self.observations) < 20:
            return

        audio_groups: dict[str, list[Observation]] = {}
        for obs in self.observations:
            bass = float(obs.audio.get("bass", 0.0))
            treble = float(obs.audio.get("treble", 0.0))
            if bass > 0.7:
                key = "high_bass"
            elif bass < 0.2:
                key = "low_bass"
            elif treble > 0.7:
                key = "high_treble"
            elif treble < 0.2:
                key = "low_treble"
            else:
                key = "mid"
            audio_groups.setdefault(key, []).append(obs)

        for group_key, group_obs in audio_groups.items():
            if len(group_obs) < 5:
                continue
            successes = [o for o in group_obs if o.success]
            if len(successes) < 3:
                continue
            mutation_counts: dict[str, int] = {}
            for o in successes:
                mutation_counts[o.mutation_id] = mutation_counts.get(o.mutation_id, 0) + 1
            best_mut = max(mutation_counts, key=mutation_counts.get)
            confidence = min(1.0, len(successes) / max(1, len(group_obs)))

            pred = f"{group_key.replace('_', ' ')} correlates with successful {best_mut}"
            hyp = Hypothesis(
                change=best_mut,
                prediction=pred,
                target_parameter=group_key,
                min_experiments=30,
                priority=confidence,
            )
            self.hypotheses.append(hyp)

    def _seed_initial_hypotheses(self) -> None:
        seeds = [
            Hypothesis(change="increase_feedback", prediction="feedback recursion stabilises motion", priority=0.3),
            Hypothesis(change="add_fractal", prediction="fractal module adds complexity", priority=0.3),
            Hypothesis(change="add_noise", prediction="noise module breaks repetition", priority=0.3),
            Hypothesis(change="add_particles", prediction="particle module introduces emergent motion", priority=0.3),
            Hypothesis(change="change_colour", prediction="colour variation improves audio sync", priority=0.3),
            Hypothesis(change="warp_module", prediction="warp module increases visual flow", priority=0.3),
        ]
        self.hypotheses.extend(seeds)

    def _schedule_experiments(self, audio: Any) -> None:
        available_slots = max(1, int(self.curiosity * 5))

        while len(self.active_experiments) < available_slots and self.hypotheses:
            def priority(h: Hypothesis) -> float:
                potential_gain = float(h.successes + 1) / max(1, h.attempts + 1)
                novelty = 1.0 - (float(h.attempts) / max(1, h.min_experiments))
                uncertainty = 1.0 - min(1.0, float(h.attempts) / 10.0)
                return float(potential_gain + novelty + uncertainty + h.priority)

            self.hypotheses.sort(key=priority, reverse=True)

            candidate = self.hypotheses.pop(0)
            if candidate.attempts >= candidate.min_experiments:
                if candidate.successes / max(1, candidate.attempts) >= 0.6:
                    self._promote_to_law(candidate)
                else:
                    self._record_failure(candidate)
                continue

            exp = Experiment(
                renderer=self.renderer,
                evolution_engine=self.evolution,
                hypothesis=candidate,
                max_runs=min(candidate.min_experiments, 50),
            )
            self.active_experiments[candidate.change] = exp

    def _run_experiments(self, audio: Any) -> None:
        if not self.active_experiments:
            return

        runs_this_tick = max(1, int(self.curiosity * 3))
        done: list[str] = []

        shuffled_keys = list(self.active_experiments.keys())
        random.shuffle(shuffled_keys)

        for key in shuffled_keys:
            if runs_this_tick <= 0:
                break
            exp = self.active_experiments[key]
            obs = exp.run(audio)
            runs_this_tick -= 1
            if obs is not None:
                self.observations.append(obs)

            if exp.hypothesis.attempts >= exp.hypothesis.min_experiments:
                done.append(key)

        for key in done:
            exp = self.active_experiments.pop(key)
            self._process_completed_experiment(exp)

    def _process_completed_experiment(self, exp: Experiment) -> None:
        hyp = exp.hypothesis
        rate = hyp.successes / max(1, hyp.attempts)
        if rate >= 0.55 and hyp.attempts >= 10:
            self._promote_to_law(hyp)
        elif hyp.attempts >= hyp.min_experiments:
            self._record_failure(hyp)

    def _promote_to_law(self, hyp: Hypothesis) -> None:
        law = {
            "law": self._format_law(hyp),
            "confidence": round(hyp.successes / max(1, hyp.attempts), 4),
            "evidence_count": hyp.attempts,
            "first_seen": time.time(),
            "last_confirmed": time.time(),
            "change": hyp.change,
        }
        laws = self._load_json(self.laws_path, [])
        for existing in laws:
            if existing.get("change") == hyp.change:
                existing["confidence"] = law["confidence"]
                existing["evidence_count"] = law["evidence_count"]
                existing["last_confirmed"] = law["last_confirmed"]
                break
        else:
            laws.append(law)
        self._save_json(self.laws_path, laws)

    def _record_failure(self, hyp: Hypothesis) -> None:
        record = {
            "change": hyp.change,
            "prediction": hyp.prediction,
            "attempts": hyp.attempts,
            "successes": hyp.successes,
            "avg_delta": round(
                sum(o.delta for o in self.observations if o.hypothesis_change == hyp.change)
                / max(1, hyp.attempts),
                4,
            ),
            "failed_at": time.time(),
        }
        failed = self._load_json(self.failed_path, [])
        failed.append(record)
        self._save_json(self.failed_path, failed)
        self.failed_hypotheses.append(record)

    def _synthesize_theories(self) -> None:
        if len(self.observations) < 50:
            return

        groups: dict[str, list[Observation]] = {}
        for obs in self.observations:
            if not obs.success:
                continue
            bass = float(obs.audio.get("bass", 0.0))
            if bass > 0.8:
                key = "bass_gt_0.8"
            elif bass < 0.2:
                key = "bass_lt_0.2"
            else:
                key = "mid_bass"
            groups.setdefault(key, []).append(obs)

        for group_key, group_obs in groups.items():
            if len(group_obs) < 5:
                continue
            actions = {}
            for o in group_obs:
                actions[o.mutation_id] = actions.get(o.mutation_id, 0) + 1

            best_action = max(actions, key=actions.get)
            confidence = min(1.0, float(len(group_obs)) / 100.0 + 0.4)

            condition = {}
            if "bass" in group_key:
                if "gt" in group_key:
                    condition["bass_gt"] = 0.8 if "0.8" in group_key else 0.5
                else:
                    condition["bass_lt"] = 0.2
            if "treble" in group_key and "lt" in group_key:
                condition["treble_lt"] = 0.2

            action_map: dict[str, Any] = {}
            if "particle" in best_action:
                action_map["increase_particle_birth_rate"] = True
            elif "feedback" in best_action:
                action_map["increase_feedback_recursion"] = True
            elif "colour" in best_action:
                action_map["shift_towards_warm"] = True
            elif "fractal" in best_action:
                action_map["increase_complexity"] = True
            elif "warp" in best_action:
                action_map["increase_flow"] = True
            elif "noise" in best_action:
                action_map["increase_disruption"] = True

            if action_map:
                self.theory.add_rule(condition=condition, action=action_map, confidence=confidence)

            discovery = {
                "id": f"disc_{int(time.time() * 1000)}",
                "pattern": f"when {group_key} then {best_action}",
                "confidence": round(confidence, 4),
                "first_seen": time.time(),
                "last_seen": time.time(),
                "occurrences": len(group_obs),
            }
            discoveries = self._load_json(self.discoveries_path, [])
            discoveries.append(discovery)
            self._save_json(self.discoveries_path, discoveries)

    def apply_theory(self, audio: Any, genome_params: dict[str, float]) -> dict[str, Any] | None:
        """Use the built theory to suggest mutation parameters.

        Intended to be called by the evolution engine or renderer to bias
        future mutations toward visually successful patterns.
        """
        audio_dict: dict[str, float] = {}
        if audio is not None:
            audio_dict = {
                "bass": float(getattr(audio, "bass", 0.0)),
                "mid": float(getattr(audio, "mid", 0.0)),
                "treble": float(getattr(audio, "treble", 0.0)),
            }
        return self.theory.recommend(audio_dict, genome_params)

    def human_feedback(self, sentiment: str) -> dict[str, float]:
        """Receive optional human feedback and return audio-conditioned nudges.

        sentiment values:
          "more_alive" -> boost growth, motion, complexity
          "calmer"     -> boost slow evolution, reduce chaos
          "more_chaos" -> boost noise, particles, disruption
        """
        nudges: dict[str, float] = {}
        if sentiment == "more_alive":
            nudges["growth"] = 1.3
            nudges["motion"] = 1.3
            nudges["complexity"] = 1.4
            hyp = Hypothesis(
                change="add_particles",
                prediction="human prefers more alive visuals",
                target_parameter="human_feedback",
                priority=2.0,
            )
            self.hypotheses.insert(0, hyp)
        elif sentiment == "calmer":
            nudges["slow_evolution"] = 1.4
            nudges["reduce_chaos"] = 0.6
            hyp = Hypothesis(
                change="increase_feedback",
                prediction="human prefers calmer visuals",
                target_parameter="human_feedback",
                priority=2.0,
            )
            self.hypotheses.insert(0, hyp)
        elif sentiment == "more_chaos":
            nudges["noise"] = 1.5
            nudges["particles"] = 1.3
            nudges["disruption"] = 1.4
            hyp = Hypothesis(
                change="add_noise",
                prediction="human prefers chaotic visuals",
                target_parameter="human_feedback",
                priority=2.0,
            )
            self.hypotheses.insert(0, hyp)
        return nudges

    def curiosity_decay(self) -> None:
        self.curiosity = max(0.1, self.curiosity * 0.99)

    def curiosity_boost(self) -> None:
        self.curiosity = min(1.0, self.curiosity + 0.2)

    def _decay_curiosity(self, dt: float) -> None:
        decay = max(0.05, 0.02 * float(dt) / max(0.1, self.interval_secs))
        self.curiosity = max(0.1, self.curiosity - decay)

    def _format_law(self, hyp: Hypothesis) -> str:
        return f"{hyp.target_parameter}: {hyp.prediction}" if hyp.target_parameter else hyp.prediction

    def _trim_observations(self) -> None:
        if len(self.observations) > self.max_observations:
            self.observations = self.observations[-self.max_observations // 2 :]

    def _persist(self) -> None:
        if self.cycles % 5 != 0:
            return
        try:
            theory_path = os.path.join(self.knowledge_dir, "theory.json")
            self._save_json(theory_path, self.theory.to_dict())
        except Exception:
            pass

    @staticmethod
    def _load_json(path: str, default: Any) -> Any:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default

    @staticmethod
    def _save_json(path: str, data: Any) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
