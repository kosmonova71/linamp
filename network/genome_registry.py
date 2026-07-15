from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any


def _stable_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class SpeciesRecord:
    species: str
    genome: dict[str, Any]
    fitness: float
    origin: str
    reputation: dict[str, Any]
    physics: dict[str, Any] | None = None
    modules: list[str] | None = None


class SpeciesRegistry:
    """Evolution fossil record.

    Stores discovered organisms (species + genome + stats) as JSON files under:
      evolution_world/species/<species_id>.json

    Uniqueness:
      stable hash of genome/modules/physics (plus module ordering).
    """

    def __init__(self, base_dir: str | None = None):
        self.base_dir = base_dir or os.path.join(os.getcwd(), "evolution_world")
        self.species_dir = os.path.join(self.base_dir, "species")
        os.makedirs(self.species_dir, exist_ok=True)

        self._index: dict[str, SpeciesRecord] = {}

    def _compute_species_id(self, organism: dict[str, Any]) -> str:
        genome = organism.get("genome", {})
        physics = genome.get("physics") or organism.get("physics") or {}
        modules = genome.get("modules") or []

        payload = {
            "modules": list(modules),
            "params": genome.get("params") or {},
            "physics": dict(physics),
        }
        return _stable_hash(payload)

    def _organism_to_record(self, organism: dict[str, Any], *, species_id: str) -> SpeciesRecord:
        genome = dict(organism.get("genome") or {})
        reputation = dict(organism.get("reputation") or {})

        fitness = float(organism.get("fitness", 0.0))
        origin = str(organism.get("origin", "unknown"))

        modules = genome.get("modules")
        physics = genome.get("physics")

        species_name = str(organism.get("species", f"species_{species_id}"))
        # Normalize species field for external display; file id is species_id.
        return SpeciesRecord(
            species=f"{species_name}",
            genome=genome,
            fitness=fitness,
            origin=origin,
            reputation=reputation,
            physics=physics,
            modules=modules,
        )

    def add(self, organism: dict[str, Any]) -> str | None:
        """Add organism if unique; returns species_id."""
        if not isinstance(organism, dict):
            return None

        species_id = self._compute_species_id(organism)
        if species_id in self._index:
            return species_id

        path = os.path.join(self.species_dir, f"{species_id}.json")
        if os.path.exists(path):
            # Load existing into index
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                record = self._organism_to_record(data, species_id=species_id)
                self._index[species_id] = record
            except Exception:
                # If corrupt, treat as non-indexed and overwrite
                pass

        if species_id in self._index:
            return species_id

        record = self._organism_to_record(organism, species_id=species_id)

        payload = {
            "species_id": species_id,
            "species": record.species,
            "genome": record.genome,
            "fitness": record.fitness,
            "origin": record.origin,
            "reputation": record.reputation,
        }
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(tmp, path)

        self._index[species_id] = record
        return species_id

    def get(self, species_id: str) -> SpeciesRecord | None:
        return self._index.get(species_id)

    def all_species_ids(self) -> list[str]:
        if self._index:
            return list(self._index.keys())
        # lazy load
        for fn in os.listdir(self.species_dir):
            if not fn.endswith(".json"):
                continue
            species_id = fn[:-5]
            try:
                with open(os.path.join(self.species_dir, fn), "r", encoding="utf-8") as f:
                    data = json.load(f)
                record = self._organism_to_record(data, species_id=species_id)
                self._index[species_id] = record
            except Exception:
                continue
        return list(self._index.keys())

    def top_by_reputation(self, limit: int = 10) -> list[tuple[str, SpeciesRecord]]:
        limit = int(limit)
        out: list[tuple[str, SpeciesRecord]] = []
        for sid in self.all_species_ids():
            rec = self._index.get(sid)
            if not rec:
                continue
            out.append((sid, rec))
        out.sort(key=lambda t: (float(t[1].reputation.get("score", t[1].fitness))), reverse=True)
        return out[:limit]

    def random_sample(self, limit: int = 5) -> list[tuple[str, SpeciesRecord]]:
        import random

        ids = self.all_species_ids()
        random.shuffle(ids)
        out: list[tuple[str, SpeciesRecord]] = []
        for sid in ids[: int(limit)]:
            rec = self._index.get(sid)
            if rec:
                out.append((sid, rec))
        return out

