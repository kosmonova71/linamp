from __future__ import annotations

import json
import os
import socket
import time
from typing import Any

from .genome_registry import SpeciesRegistry
from .migration import MigrationRules
from .reputation import ReputationScorer


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


class EvolutionNodeId:
    @staticmethod
    def default() -> str:
        host = socket.gethostname() or "node"
        pid = os.getpid()
        return f"{host}_{pid}"


class SpeciesExchange:
    """Stage 9 orchestrator.

    Implementation note:
      - For first pass, exchange is file-based (no sockets).
      - This still enables collective evolution by running multiple instances
        sharing a working directory (e.g., a mounted folder).
    """

    def __init__(
        self,
        *,
        base_dir: str | None = None,
        node_id: str | None = None,
        migration_interval_secs: float = 30.0,
    ):
        self.base_dir = base_dir or os.path.join(os.getcwd(), "evolution_world")
        self.registry = SpeciesRegistry(base_dir=self.base_dir)
        self.reputation = ReputationScorer()
        self.migration = MigrationRules()

        self.node_id = str(node_id or EvolutionNodeId.default())

        self.inbox_dir = os.path.join(self.base_dir, "inbox")
        self.outbox_dir = os.path.join(self.base_dir, "outbox")
        self.lineages_dir = os.path.join(self.base_dir, "lineages")
        self.discoveries_dir = os.path.join(self.base_dir, "discoveries")

        for d in (self.inbox_dir, self.outbox_dir, self.lineages_dir, self.discoveries_dir):
            os.makedirs(d, exist_ok=True)

        self._timer = 0.0
        self.migration_interval_secs = float(migration_interval_secs)

    def _local_env_fingerprint(self, audio: Any) -> dict[str, Any]:
        return {
            "audio_bass": _safe_float(getattr(audio, "bass", 0.0)),
            "audio_mid": _safe_float(getattr(audio, "mid", 0.0)),
            "audio_treble": _safe_float(getattr(audio, "treble", 0.0)),
        }

    @staticmethod
    def _genome_to_organism(*, genome: Any, fitness: float, origin: str, species: str, reputation: dict[str, Any]) -> dict[str, Any]:
        # Map current VisualGenome-ish shape -> requested organism schema.
        modules = list(getattr(genome, "modules", []))
        params = dict(getattr(genome, "params", {}))

        # physics genes: we adapt from existing params where possible.
        physics = {
            "gravity": float(params.get("gravity", 0.2)),
            "energy_decay": float(params.get("energy_decay", params.get("feedback", 0.9) * 0.1)),
        }

        return {
            "species": species,
            "genome": {
                "modules": modules,
                "params": params,
                "physics": physics,
            },
            "fitness": float(fitness),
            "origin": str(origin),
            "reputation": dict(reputation or {}),
        }

    def _score_candidate(self, *, fitness: float, novelty: float, stability: float, audio: Any) -> dict[str, Any]:
        # Approx audio_response & runtime_survival with heuristics.
        audio_response = float(
            (_safe_float(getattr(audio, "bass", 0.0)) + _safe_float(getattr(audio, "mid", 0.0)) + _safe_float(getattr(audio, "treble", 0.0))) / 3.0
        )
        runtime_survival = float(stability)

        rep = self.reputation.score(
            fitness=float(fitness),
            novelty=float(novelty),
            stability=float(stability),
            audio_response=audio_response,
            runtime_survival=runtime_survival,
        )
        return rep.to_dict()

    def discover_or_upload(self, *, genome: Any, fitness: float, audio: Any, mutation_id: str | None = None) -> str | None:
        """Convert genome->organism, score reputation, register, and place into outbox."""
        mutation_id = mutation_id or "unknown_mut"

        # Novelty/stability are not fully implemented yet; approximate for Stage 9.
        novelty = float(min(1.0, max(0.0, fitness / 120.0)))  # placeholder bounded
        stability = 0.75  # placeholder; in future: compute from mutation history variance

        # A simple species name from mutation_id + a coarse genome signature.
        species_name = f"{mutation_id}" if mutation_id else "unknown_species"

        rep = self._score_candidate(fitness=float(fitness), novelty=novelty, stability=stability, audio=audio)

        organism = self._genome_to_organism(
            genome=genome,
            fitness=float(fitness),
            origin=self.node_id,
            species=species_name,
            reputation=rep,
        )

        species_id = self.registry.add(organism)
        if not species_id:
            return None

        # Add to outbox as a mailbox message.
        msg = {
            "ts": time.time(),
            "node_id": self.node_id,
            "species_id": species_id,
            "organism": organism,
        }
        out_path = os.path.join(self.outbox_dir, f"{self.node_id}_out.json")
        # Append as list for simplicity.
        try:
            if os.path.exists(out_path):
                with open(out_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                if not isinstance(existing, list):
                    existing = []
            else:
                existing = []
        except Exception:
            existing = []

        existing.append(msg)
        tmp = out_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(existing[-200:], f, indent=2, sort_keys=True)
        os.replace(tmp, out_path)

        return species_id

    def _receive_from_inbox(self) -> int:
        count = 0
        for fn in os.listdir(self.inbox_dir):
            if not fn.endswith(".json"):
                continue
            path = os.path.join(self.inbox_dir, fn)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                continue

            msgs = payload if isinstance(payload, list) else [payload]
            for msg in msgs:
                try:
                    organism = msg.get("organism") if isinstance(msg, dict) else None
                    if not organism:
                        continue
                    self.registry.add(organism)
                    count += 1
                except Exception:
                    continue

            # mark as consumed
            try:
                os.replace(path, path + ".consumed")
            except Exception:
                pass

        return count

    def _adapt_genome_locally(self, received_organism: dict[str, Any], *, audio: Any, local_world: Any | None = None) -> Any:
        # We create an object shaped like VisualGenome used by renderer/evolution.
        # VisualGenome class exists in evo_renderer/genome.py and includes modules + params.
        from evo_renderer.genome import VisualGenome

        gv = VisualGenome()

        genome = dict(received_organism.get("genome") or {})
        gv.modules = list(genome.get("modules") or gv.modules)
        params = dict(genome.get("params") or {})

        # Local adaptation rules (audio style + world summaries placeholders)
        bass = _safe_float(getattr(audio, "bass", 0.0))
        treble = _safe_float(getattr(audio, "treble", 0.0))

        # Map audio->params nudges in safe ranges (VisualGenome clamps).
        if "feedback" in params:
            params["feedback"] = max(0.0, min(0.99, float(params["feedback"]) * (0.85 + 0.3 * bass)))
        else:
            params["feedback"] = max(0.0, min(0.99, float(gv.params.get("feedback", 0.85)) * (0.85 + 0.3 * bass)))

        if "warp_strength" in params:
            params["warp_strength"] = max(0.0, min(1.0, float(params["warp_strength"]) * (0.8 + 0.4 * treble)))

        # Occasional regional mutation to create new variant
        gv.modules = list(gv.modules)
        if gv.modules:
            import random

            if random.random() < 0.35:
                # mutate by using internal mutate; this respects clamp/stability.
                gv.params = dict(params)
                gv.mutate(strength=0.10)
            else:
                gv.params = dict(params)

        # Ensure required keys
        for k, v in VisualGenome().params.items():
            gv.params.setdefault(k, v)

        return gv

    def tick(self, *, dt: float, audio: Any, try_receive_candidates: int = 1) -> list[Any]:
        self._timer += float(dt)
        if self._timer < self.migration_interval_secs:
            return []
        self._timer = 0.0

        # Receive incoming messages.
        self._receive_from_inbox()

        # Select candidates from registry to propose locally adapted genomes.
        candidates = self.registry.random_sample(limit=20)

        # Migration selection: top/rare/unexpected using reputation stored in record.
        # We don’t have explicit novelty/unexpected metrics yet; reuse reputation fields.
        selected = self.migration.select_payload(
            candidates=candidates,
            get_reputation_score=lambda rec: rec.reputation.get("score", rec.fitness),
            get_novelty=lambda rec: rec.reputation.get("novelty", 0.5),
            get_unexpected=lambda rec: rec.reputation.get("novelty", 0.5),
        )

        adapted: list[Any] = []
        for _, rec in selected[: max(1, int(try_receive_candidates))]:
            # Need the exact organism JSON to adapt from. Rebuild minimal.
            received_organism = {
                "genome": rec.genome,
                "fitness": rec.fitness,
                "origin": rec.origin,
                "species": rec.species,
            }
            gv = self._adapt_genome_locally(received_organism, audio=audio)
            adapted.append(gv)

        return adapted

