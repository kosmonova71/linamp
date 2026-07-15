from __future__ import annotations

import os
import random
import time
from typing import Any


from .mutation_brain import MutationBrain
from .evolution_memory import EvolutionMemory, MutationResult

from brain.structure_agent import StructureAgent
from brain.motion_agent import MotionAgent
from brain.colour_agent import ColourAgent
from brain.guardian_agent import GuardianAgent

from network.species_exchange import SpeciesExchange

from identity.engine import IdentityEngine



class EvolutionEngine:
    """Stage 5 neural evolution layer + Stage 12 identity system.

    - Uses MutationBrain to choose mutation strategy based on audio context.
    - Evaluates fitness (proxy) before/after mutation.
    - Records MutationResult and rewards brain.
    - In silence (dream mode), stores champions without blocking visuals.
    - Identity engine provides persistent mood, goals, memory, lineage, dreams.
    """

    def __init__(
        self,
        renderer,
        interval_secs: float = 15.0,
        population_size: int = 3,
    ):
        self.renderer = renderer
        self.timer = 0.0
        self.interval_secs = float(interval_secs)

        # Stage 8 (Option 1): novelty via concept archaeology.
        from creativity.archive import ConceptArchaeology
        from .fitness import Fitness

        concept_db = os.path.join(os.path.dirname(__file__), "mutations.db")
        concept_archive = ConceptArchaeology(db_path=concept_db)
        self.fitness = Fitness(concept_archive=concept_archive)

        self.brain = MutationBrain()  # kept for backwards compatibility
        self.memory = EvolutionMemory()

        # Stage 7: multi-agent hive (structure/motion/colour) + guardian.
        self.structure_agent = StructureAgent()
        self.motion_agent = MotionAgent()
        self.colour_agent = ColourAgent()
        self.guardian = GuardianAgent(min_score=0.0, min_delta=0.0)

        # Stage 9: collective evolution network (file-based exchange).
        try:
            self.species_exchange = SpeciesExchange(
                migration_interval_secs=max(5.0, self.interval_secs * 2.0)
            )
        except Exception:
            self.species_exchange = None

        # Stage 11: autonomous visual researcher
        try:
            from research.researcher import Researcher

            self.researcher = Researcher(renderer=self.renderer, interval_secs=self.interval_secs * 1.5)
        except Exception:
            self.researcher = None

        self.population_size = int(max(1, population_size))

        # Dream mode
        self._silent_accum = 0.0
        self.silence_threshold = 0.08  # average of bass/mid/treble
        self.silence_hold_secs = 5.0
        self.dream_evolutions_per_tick = 2

        # Stage 12: persistent identity system
        try:
            self.identity = IdentityEngine(
                renderer=self.renderer,
                state_path=os.path.join(os.path.dirname(__file__), "..", "identity", "identity_state.json"),
            )
        except Exception:
            self.identity = None

        self._was_silent = False

        # internal frame counter for fitness proxy
        self._frame_time = time.time()

    def update(self, dt: float) -> list[dict[str, Any]]:
        audio = self.renderer.last_audio_state
        if audio is None:
            return []

        # Silence detection
        avg_energy = (float(getattr(audio, "bass", 0.0)) + float(getattr(audio, "mid", 0.0)) + float(getattr(audio, "treble", 0.0))) / 3.0
        if avg_energy < self.silence_threshold:
            self._silent_accum += float(dt)
        else:
            self._silent_accum = 0.0

        is_dream = self._silent_accum >= self.silence_hold_secs
        just_woke = self._was_silent and not is_dream
        self._was_silent = is_dream

        # Stage 12: tick identity engine
        dreams_applied: list[dict[str, Any]] = []
        if self.identity is not None:
            try:
                self.identity.update(dt, audio=audio)
            except Exception:
                pass
            if just_woke:
                try:
                    dreams = self.identity.consume_dreams()
                    dreams_applied = self._apply_dreams(dreams)
                except Exception:
                    pass

        # Evolve at regular cadence; during dream we do more steps.
        self.timer += float(dt)
        if self.timer <= self.interval_secs and not is_dream:
            return dreams_applied

        # In dream, allow periodic evolutions even if interval hasn't hit
        if is_dream and self.timer < self.interval_secs:
            # still evolve, but capped
            if random.random() < 0.3:
                self._evolve_once(audio=audio, dream=True)
            return dreams_applied

        # Otherwise normal interval evolution (or dream after interval)
        self.timer = 0.0
        if is_dream:
            for _ in range(self.dream_evolutions_per_tick):
                self._evolve_once(audio=audio, dream=True)
        else:
            self._evolve_once(audio=audio, dream=False)

        # Stage 9: periodically receive/migrate and attempt local adaptation.
        try:
            if self.species_exchange is not None:
                adapted = self.species_exchange.tick(dt=dt, audio=audio)
                for cand in adapted[: max(0, int(self.population_size))]:
                    parent = self._snapshot_genome(self.renderer.genome)
                    before = self._score_current(audio)
                    self.renderer.genome = cand
                    self.renderer.evolve_shader()
                    after = self._score_current(audio)
                    delta = float(after) - float(before)
                    decision = self.guardian.approve(
                        {"score": after, "delta": delta},
                        parent_score=before,
                    )
                    if decision.approved:
                        self.memory.save_champion(cand, score=after, parent=parent)
                        if self.identity is not None:
                            try:
                                self.identity.record_champion(cand, score=after, parent=parent)
                            except Exception:
                                pass
                    else:
                        self.renderer.genome = parent
                        self.renderer.evolve_shader()
        except Exception:
            pass

        return dreams_applied

    def _score_current(self, audio: Any, *, concept: Any | None = None) -> float:
        # Use a monotonically increasing time for the proxy.
        frame = float(self.renderer.elapsed_time)

        novelty = None
        if concept is not None and getattr(self.fitness, "concept_archive", None) is not None:
            try:
                concept_id = concept.stable_id() if hasattr(concept, "stable_id") else None
                novelty_res = self.fitness.concept_archive.novelty(concept, concept_id=concept_id)
                novelty = novelty_res.novelty
            except Exception:
                novelty = None

        # Stage 10: renderer GPU cost penalty (proxy).
        gpu_cost = 0.0
        try:
            modules = getattr(self.renderer.genome, "modules", None)
            passes = None
            if hasattr(self.renderer, "renderer_genome"):
                passes = self.renderer.renderer_genome.pipeline.get("passes", None)
            feedback_resolution = 1.0
            if hasattr(self.renderer, "renderer_genome"):
                feedback_resolution = float(self.renderer.renderer_genome.pipeline.get("feedback_resolution", 1.0))

            gpu_cost = float(
                self.renderer.gpu_profiler.profiler_cost(
                    modules=modules,
                    pipeline_passes=passes,
                    dynamic_feedback_res=feedback_resolution,
                    cpu_draw_seconds=None,
                )
            )
        except Exception:
            gpu_cost = 0.0

        base = self.fitness.evaluate(frame=frame, audio=audio, novelty=novelty)

        # Fitness = visual/audio/novelty proxy - gpu_cost penalty.
        return float(base - 0.25 * gpu_cost)



    def _snapshot_genome(self, genome: Any) -> Any:
        # Prefer clone() when the genome supports it (e.g., ModularGenome).
        if hasattr(genome, "clone") and callable(getattr(genome, "clone")):
            try:
                return genome.clone()
            except Exception:
                pass

        # Fallback for legacy VisualGenome.
        g = type(genome)()
        if hasattr(g, "modules"):
            g.modules = list(getattr(genome, "modules", []))
        if hasattr(g, "params"):
            g.params = dict(getattr(genome, "params", {}))
        return g

    def _apply_dreams(self, dreams: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not dreams or not dreams[0].get("genome"):
            return []
        best = dreams[0]
        genome = best["genome"]
        score = best.get("score", 0.0)
        try:
            if score > 0.5:
                parent = self._snapshot_genome(self.renderer.genome)
                self.renderer.genome = genome
                self.renderer.evolve_shader()
                if self.identity is not None:
                    try:
                        self.identity.record_champion(genome, score=score, parent=parent)
                    except Exception:
                        pass
                return [best]
        except Exception:
            pass
        return []

    def _apply_brain_mutation(self, genome: Any, mutation_id: str) -> None:
        """Apply mutation_id to the active genome.

        Supports both:
        - legacy evo_renderer.genome.VisualGenome (modules + params)
        - ModularGenome (field/warp/palette/post + *_params)
        """
        identity = getattr(self, "identity", None)
        if identity is not None:
            try:
                strength = identity.visual_mutation_strength()
            except Exception:
                strength = 0.12
        else:
            strength = 0.12

        is_modular = hasattr(genome, "field") and hasattr(genome, "warp") and hasattr(genome, "post")

        if is_modular:
            # --- Map mutation IDs into ModularGenome module selections ---
            if mutation_id == "increase_feedback":
                # Ensure feedback post exists and bump its parameters if available.
                if "feedback" not in getattr(genome, "post", []):
                    genome.post = list(getattr(genome, "post", [])) + ["feedback"]
                if hasattr(genome, "post_params") and isinstance(genome.post_params, dict):
                    fp = genome.post_params.get("feedback", {})
                    if isinstance(fp, dict):
                        fp_mode = float(fp.get("mode", 0))
                        fp_speed = float(fp.get("speed", 1.0))
                        fp_amount = float(fp.get("amount", 0.5))
                        genome.post_params["feedback"] = {
                            "mode": int(fp_mode),
                            "speed": max(0.1, min(3.0, fp_speed + random.uniform(-0.3, 0.3))),
                            "amount": max(0.0, min(1.0, fp_amount + random.uniform(-0.1, 0.1))),
                        }
                genome.mutate(strength=strength)

            elif mutation_id == "change_colour":
                # Palette selection tweak + parameter mutation handled by genome.mutate()
                genome.mutate(strength=max(0.05, strength * 0.8))

            elif mutation_id == "add_fractal":
                # Map to selecting FBM field.
                genome.field = "fbm"
                genome.mutate(strength=strength)

            elif mutation_id == "add_particles":
                # Map to stronger warp variety.
                genome.warp = random.choice(["vortex", "curl"])
                genome.mutate(strength=strength)

            elif mutation_id == "add_noise":
                genome.warp = random.choice(["curl", "twist"])
                genome.mutate(strength=strength)

            elif mutation_id == "warp_module":
                genome.warp = random.choice(["curl", "vortex", "twist", "none"])
                genome.mutate(strength=strength)

            else:
                genome.mutate(strength=strength)

            # Avoid too many post modules.
            if hasattr(genome, "post") and isinstance(genome.post, list) and len(genome.post) > 3:
                genome.post = genome.post[-3:]

            return

        # --- Legacy VisualGenome mutations ---
        if mutation_id == "increase_feedback":
            genome.params["feedback"] = max(
                0.0,
                min(0.99, float(genome.params.get("feedback", 0.85)) + 0.03),
            )
            genome.mutate(strength=strength)

        elif mutation_id == "change_colour":
            for k in ("colour_a", "colour_b", "colour_c"):
                genome.params[k] = max(
                    0.0,
                    min(1.0, float(genome.params.get(k, 0.5)) + random.uniform(-0.25, 0.25)),
                )
            genome.mutate(strength=strength * 0.8)

        elif mutation_id == "add_fractal":
            if random.random() < 0.75:
                genome.modules.append("fractal")
            genome.mutate(strength=strength)

        elif mutation_id == "add_particles":
            if random.random() < 0.75:
                genome.modules.append("particles")
            genome.mutate(strength=strength)

        elif mutation_id == "add_noise":
            if random.random() < 0.7:
                genome.modules.append("noise")
            genome.mutate(strength=strength)

        elif mutation_id == "warp_module":
            if random.random() < 0.7:
                genome.modules.append("warp")
            genome.mutate(strength=strength)

        else:
            genome.mutate(strength=strength)

        # Avoid runaway module growth
        if hasattr(genome, "modules") and len(genome.modules) > 6:
            genome.modules = genome.modules[-6:]

    def _hive_vote(self, *, genome: Any, audio: Any) -> str:
        """Ask specialized agents for mutation intents and vote for one."""

        intents: list[dict[str, Any]] = []

        for agent in (self.structure_agent, self.motion_agent, self.colour_agent):
            intent = agent.suggest(genome, audio)
            if intent:
                intents.append(intent)

        # If everyone returns None, fall back to legacy brain.
        if not intents:
            return self.brain.choose(audio)

        # Weighted vote by mutation_id.
        scores: dict[str, float] = {}
        for intent in intents:
            mut_id = str(intent.get("mutation_id"))
            w = float(intent.get("weight", 1.0))
            scores[mut_id] = scores.get(mut_id, 0.0) + w

        # Pick highest voted mutation id.
        return max(scores.keys(), key=lambda k: scores[k])

    def _evolve_once(self, audio: Any, dream: bool) -> None:
        parent = self._snapshot_genome(self.renderer.genome)
        # Stage 8 (Option 1): use a random concept as novelty carrier.
        # Later stages will replace this with tree/ConceptDNA evolution.
        from creativity.concepts import random_concept_dna

        concept = random_concept_dna()
        before = self._score_current(audio, concept=concept)

        # Stage 12: identity genome bias for explorer vs conservative mode
        identity = getattr(self, "identity", None)
        if identity is not None and identity.should_explore_unknown():
            exploration_mutations = ["add_fractal", "add_particles", "add_noise", "warp_module"]
            if random.random() < 0.6:
                mutation_id = random.choice(exploration_mutations)
            else:
                mutation_id = self._hive_vote(genome=parent, audio=audio)
        elif identity is not None and identity.prefer_known_species():
            safe_mutations = ["increase_feedback", "change_colour", "tune_params"]
            if random.random() < 0.5:
                mutation_id = random.choice(safe_mutations)
            else:
                mutation_id = self._hive_vote(genome=parent, audio=audio)
        else:
            # Stage 7: multi-agent voting for a proposed mutation.
            mutation_id = self._hive_vote(genome=parent, audio=audio)

        # Stage 11: bias mutation choice with researcher theories.
        if getattr(self, "researcher", None) is not None:
            try:
                recommendation = self.researcher.apply_theory(
                    audio=audio,
                    genome_params=dict(getattr(parent, "params", {})),
                )
                if recommendation and random.random() < 0.25:
                    mapping = {
                        "increase_particle_birth_rate": "add_particles",
                        "increase_feedback_recursion": "increase_feedback",
                        "shift_towards_warm": "change_colour",
                        "increase_complexity": "add_fractal",
                        "increase_flow": "warp_module",
                        "increase_disruption": "add_noise",
                    }
                    mapped = mapping.get(list(recommendation.values())[0])
                    if mapped:
                        mutation_id = mapped
            except Exception:
                pass

        child = self._snapshot_genome(parent)
        self._apply_brain_mutation(child, mutation_id=mutation_id)

        # Evaluate candidate.
        self.renderer.genome = child
        self.renderer.evolve_shader()
        after = self._score_current(audio)

        delta = float(after) - float(before)

        # Guardian approval gate.
        decision = self.guardian.approve({"score": after, "delta": delta}, parent_score=before)
        if not decision.approved:
            # Revert to parent genome to avoid persisting bad visuals.
            self.renderer.genome = parent
            self.renderer.evolve_shader()
            return

        result = MutationResult(
            mutation=mutation_id,
            before_score=before,
            after_score=after,
        )
        self.memory.record_mutation(result, audio_state=audio)

        # Stage 12: record experience in identity system
        if identity is not None:
            try:
                identity.record_mutation(mutation_id, before, after)
                # ModularGenome doesn't have `modules`; record post modules instead.
                if hasattr(child, "modules"):
                    identity.record_modules(getattr(child, "modules", []), delta)
                elif hasattr(child, "post"):
                    identity.record_modules(getattr(child, "post", []), delta)
            except Exception:
                pass

        improvement = float(result.delta)

        # Rewards: legacy brain + agent memory.
        if improvement != 0.0:
            try:
                self.brain.reward(mutation_id, improvement)
            except Exception:
                pass

            # very lightweight agent memory update
            for agent in (self.structure_agent, self.motion_agent, self.colour_agent):
                likes = agent.memory.setdefault("likes", {})
                likes[mutation_id] = float(likes.get(mutation_id, 0.0)) + improvement


        # Save champions only if it's good (or during dream mode aggressively)
        # Record concept novelty carrier for archaeology.
        try:
            if getattr(self.fitness, "concept_archive", None) is not None:
                concept_id = concept.stable_id() if hasattr(concept, "stable_id") else None
                if concept_id is not None:
                    self.fitness.concept_archive.record(concept, concept_id=concept_id, score=after)
        except Exception:
            pass

        if dream:
            self.memory.save_champion(child, score=after, parent=parent)
            if identity is not None:
                try:
                    identity.record_champion(child, score=after, parent=parent)
                except Exception:
                    pass
        else:
            if after > before and after > 1.3:
                self.memory.save_champion(child, score=after, parent=parent)
                if identity is not None:
                    try:
                        identity.record_champion(child, score=after, parent=parent)
                    except Exception:
                        pass

        # Stage 9: discover/upload champion as species genetics.
        try:
            if self.species_exchange is not None:
                self.species_exchange.discover_or_upload(
                    genome=child,
                    fitness=after,
                    audio=audio,
                    mutation_id=mutation_id,
                )
        except Exception:
            pass

        # persist brain occasionally
        if random.random() < 0.08:
            try:
                self.brain.save()
            except Exception:
                pass



