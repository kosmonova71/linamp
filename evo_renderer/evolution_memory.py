from __future__ import annotations

import json
import math
import os
import random
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# ============================================================================
# Data classes
# ============================================================================


@dataclass
class MutationResult:
    mutation: str
    before_score: float
    after_score: float

    @property
    def delta(self) -> float:
        return float(self.after_score) - float(self.before_score)


@dataclass
class MusicSignature:
    """Fingerprint of a music section for pattern matching (point 31)."""

    bass_profile: float = 0.0
    mid_profile: float = 0.0
    treble_profile: float = 0.0
    energy: float = 0.0
    beat: float = 0.0
    tempo: float = 0.0
    section: str = "unknown"

    # Enhanced Music DNA fields (point 31)
    bpm: float = 0.0
    danceability: float = 0.0
    spectral_centroid: float = 0.0
    beat_strength: float = 0.0
    harmonic_complexity: float = 0.0
    dynamic_range: float = 0.0
    key: str = "C"
    mood: str = "calm"

    @staticmethod
    def from_audio(audio: Any) -> "MusicSignature":
        bass = float(getattr(audio, "bass", getattr(audio, "audio", {}).get("bass", 0.0)))
        mid = float(getattr(audio, "mid", getattr(audio, "audio", {}).get("mid", 0.0)))
        treble = float(getattr(audio, "treble", getattr(audio, "audio", {}).get("treb", 0.0)))
        energy = float(getattr(audio, "energy", bass * 0.5 + mid * 0.3 + treble * 0.2))
        beat = float(getattr(audio, "beat", 0.0))
        tempo = float(getattr(audio, "tempo", 0.0))

        if energy > 1.2:
            section = "drop"
        elif energy > 0.7:
            section = "buildup"
        elif energy > 0.3:
            section = "verse"
        else:
            section = "ambient"

        return MusicSignature(
            bass_profile=bass,
            mid_profile=mid,
            treble_profile=treble,
            energy=energy,
            beat=beat,
            tempo=tempo,
            section=section,
        )

    def feature_vector(self) -> List[float]:
        return [
            self.bass_profile,
            self.mid_profile,
            self.treble_profile,
            self.energy,
            self.beat,
            self.tempo,
            self.bpm / 200.0,
            self.danceability,
            self.spectral_centroid,
            self.beat_strength,
            self.harmonic_complexity,
            self.dynamic_range,
        ]

    def similarity(self, other: "MusicSignature") -> float:
        a = self.feature_vector()
        b = other.feature_vector()
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a)) or 1.0
        mag_b = math.sqrt(sum(x * x for x in b)) or 1.0
        return dot / (mag_a * mag_b)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MusicSignature":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class WorldDNA:
    """The environment itself evolves over time."""

    gravity: float = 0.4
    field_density: float = 0.7
    colour_temperature: float = 0.5
    mutation_pressure: float = 0.3
    organism_capacity: int = 12

    def mutate(self, strength: float = 0.05) -> None:
        for attr in self.__dataclass_fields__:
            current = getattr(self, attr)
            if isinstance(current, int):
                delta = random.choice([-1, 0, 1])
                new_val = max(1, min(64, current + delta))
            else:
                delta = random.uniform(-strength, strength)
                new_val = max(0.0, min(1.0, current + delta))
            setattr(self, attr, new_val)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorldDNA":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SpeciesRecord:
    """A record of a species/genome instance in the ecosystem."""

    genome_json: str = ""
    lineage: str = ""
    generation: int = 0
    fitness: float = 0.0
    historical_fitness: float = 0.0
    adaptability: float = 0.0
    alive: bool = True
    born_ts: float = field(default_factory=time.time)
    died_ts: Optional[float] = None
    music_sig_json: Optional[str] = None
    cause_of_death: str = "replaced"

    def age_seconds(self) -> float:
        end = self.died_ts if self.died_ts else time.time()
        return end - self.born_ts


class GenomeArchive:
    """Genetic library: champions, extinct species, and mutation history."""

    def __init__(self):
        self.champions: List[Dict[str, Any]] = []
        self.extinct_species: List[SpeciesRecord] = []
        self.mutations: List[Dict[str, Any]] = []

    def add_champion(self, genome: Any, score: float, music_sig: Optional[MusicSignature] = None) -> None:
        entry = {
            "genome": self._serialize(genome),
            "score": float(score),
            "ts": time.time(),
            "music_section": music_sig.section if music_sig else "unknown",
            "generation": getattr(genome, "generation", 0),
            "lineage": getattr(genome, "lineage", ""),
        }
        self.champions.append(entry)
        self.champions.sort(key=lambda x: x["score"], reverse=True)
        self.champions = self.champions[:200]

    def record_extinction(self, record: SpeciesRecord) -> None:
        self.extinct_species.append(record)
        if len(self.extinct_species) > 500:
            self.extinct_species = self.extinct_species[-500:]

    def record_mutation(self, mutation: str, before: float, after: float, accepted: bool) -> None:
        self.mutations.append({
            "mutation": mutation,
            "before": float(before),
            "after": float(after),
            "delta": float(after) - float(before),
            "accepted": accepted,
            "ts": time.time(),
        })
        if len(self.mutations) > 2000:
            self.mutations = self.mutations[-2000:]

    def best_champion(self, limit: int = 5) -> List[Dict[str, Any]]:
        return self.champions[:limit]

    def top_extinct(self, limit: int = 10) -> List[SpeciesRecord]:
        scored = sorted(self.extinct_species, key=lambda r: r.historical_fitness, reverse=True)
        return scored[:limit]

    @staticmethod
    def _serialize(genome: Any) -> Dict[str, Any]:
        if hasattr(genome, "to_dict"):
            return genome.to_dict()
        if isinstance(genome, dict):
            return dict(genome)
        return {"_raw": str(genome)}


class EvolutionMemory:
    """Persistent learning memory with ecosystem-wide context.

    Stores:
      - mutation results (before/after/score delta)
      - champions (top genomes)
      - music patterns (audio -> genome associations)
      - species history (lineage, extinctions)
      - world DNA (environment parameters)
      - timeline (session events)
    """

    def __init__(self, db_path: str | None = None):
        default_db = os.path.join(os.path.dirname(__file__), "mutations.db")
        self.db_path = db_path or default_db
        self._init_db()
        self.archive = GenomeArchive()
        self.world_dna = WorldDNA()
        self._session_start = time.time()
        self._events: List[Dict[str, Any]] = []

    def _connect(self) -> sqlite3.Connection:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS mutation_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    mutation TEXT NOT NULL,
                    before_score REAL NOT NULL,
                    after_score REAL NOT NULL,
                    delta REAL NOT NULL,
                    audio_json TEXT
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS champions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    score REAL NOT NULL,
                    genome_json TEXT NOT NULL,
                    parent_genome_json TEXT
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS music_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    bass REAL NOT NULL,
                    mid REAL NOT NULL,
                    treble REAL NOT NULL,
                    energy REAL NOT NULL,
                    beat REAL NOT NULL,
                    tempo REAL NOT NULL,
                    section TEXT NOT NULL,
                    genome_json TEXT NOT NULL,
                    score REAL NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS species_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    lineage TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    fitness REAL NOT NULL,
                    historical_fitness REAL NOT NULL,
                    adaptability REAL NOT NULL,
                    alive INTEGER NOT NULL,
                    born_ts REAL NOT NULL,
                    died_ts REAL,
                    music_sig_json TEXT,
                    cause_of_death TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS world_dna (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    ts REAL NOT NULL,
                    params_json TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS timeline (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    detail TEXT
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_mutation_results_mut
                ON mutation_results(mutation);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_champions_score
                ON champions(score);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_music_section
                ON music_patterns(section);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_species_lineage
                ON species_history(lineage);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_timeline_ts
                ON timeline(ts);
                """
            )

    @staticmethod
    def _genome_to_json(genome: Any) -> str:
        modules = getattr(genome, "modules", [])
        params = getattr(genome, "params", {})
        payload = {"modules": list(modules), "params": dict(params)}
        return json.dumps(payload)

    def _event(self, event_type: str, detail: str = "") -> None:
        self._events.append({"ts": time.time(), "type": event_type, "detail": detail})

    # ------------------------------------------------------------------
    # Mutation recording
    # ------------------------------------------------------------------

    def record_mutation(self, result: MutationResult, audio_state: Any | None = None) -> None:
        audio_json = None
        if audio_state is not None:
            try:
                audio_json = json.dumps(
                    {
                        "bass": float(getattr(audio_state, "bass", 0.0)),
                        "mid": float(getattr(audio_state, "mid", 0.0)),
                        "treble": float(getattr(audio_state, "treble", 0.0)),
                    }
                )
            except Exception:
                audio_json = None

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO mutation_results (ts, mutation, before_score, after_score, delta, audio_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    time.time(),
                    str(result.mutation),
                    float(result.before_score),
                    float(result.after_score),
                    float(result.delta),
                    audio_json,
                ),
            )
        self.archive.record_mutation(result.mutation, result.before_score, result.after_score, result.delta > 0)
        self._event("mutation", result.mutation)

    def save_champion(self, genome: Any, score: float, parent: Any | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO champions (ts, score, genome_json, parent_genome_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    time.time(),
                    float(score),
                    self._genome_to_json(genome),
                    self._genome_to_json(parent) if parent is not None else None,
                ),
            )
        self.archive.add_champion(genome, score)
        self._event("champion", f"score={score:.3f}")

    def top_champions(self, limit: int = 5) -> list[dict[str, Any]]:
        limit = int(limit)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT ts, score, genome_json FROM champions
                ORDER BY score DESC, ts DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        out: list[dict[str, Any]] = []
        for ts, score, genome_json in rows:
            try:
                genome = json.loads(genome_json)
            except Exception:
                genome = {"raw": genome_json}
            out.append({"ts": ts, "score": float(score), "genome": genome})
        return out

    # ------------------------------------------------------------------
    # Music memory
    # ------------------------------------------------------------------

    def store_music_pattern(self, music_sig: MusicSignature, genome: Any, score: float) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO music_patterns
                    (ts, bass, mid, treble, energy, beat, tempo, section, genome_json, score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    time.time(),
                    float(music_sig.bass_profile),
                    float(music_sig.mid_profile),
                    float(music_sig.treble_profile),
                    float(music_sig.energy),
                    float(music_sig.beat),
                    float(music_sig.tempo),
                    music_sig.section,
                    json.dumps(self._serialize_genome(genome)),
                    float(score),
                ),
            )
        self._event("music_pattern", f"section={music_sig.section}")

    def find_similar_music(self, music_sig: MusicSignature, threshold: float = 0.75, limit: int = 5) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT ts, bass, mid, treble, energy, beat, tempo, section, genome_json, score
                FROM music_patterns
                ORDER BY ts DESC
                LIMIT 500
                """
            ).fetchall()

        matches = []
        for row in rows:
            ts, bass, mid, treble, energy, beat, tempo, section, genome_json, score = row
            stored = MusicSignature(
                bass_profile=bass, mid_profile=mid, treble_profile=treble,
                energy=energy, beat=beat, tempo=tempo, section=section,
            )
            sim = music_sig.similarity(stored)
            if sim >= threshold:
                try:
                    genome = json.loads(genome_json)
                except Exception:
                    genome = {}
                matches.append({
                    "similarity": sim,
                    "score": float(score),
                    "section": section,
                    "genome": genome,
                    "ts": ts,
                })

        matches.sort(key=lambda x: x["similarity"], reverse=True)
        return matches[:limit]

    # ------------------------------------------------------------------
    # Species history
    # ------------------------------------------------------------------

    def record_species_born(self, record: SpeciesRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO species_history
                    (lineage, generation, fitness, historical_fitness, adaptability,
                     alive, born_ts, died_ts, music_sig_json, cause_of_death)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.lineage,
                    record.generation,
                    float(record.fitness),
                    float(record.historical_fitness),
                    float(record.adaptability),
                    1 if record.alive else 0,
                    float(record.born_ts),
                    record.died_ts,
                    record.music_sig_json,
                    record.cause_of_death,
                ),
            )
        self._event("species_born", record.lineage)

    def record_species_died(self, lineage: str, cause: str = "replaced") -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE species_history
                SET alive = 0, died_ts = ?, cause_of_death = ?
                WHERE lineage = ? AND alive = 1
                """,
                (time.time(), cause, lineage),
            )
        rec = SpeciesRecord(lineage=lineage, cause_of_death=cause, died_ts=time.time())
        self.archive.record_extinction(rec)
        self._event("species_died", lineage)

    def species_stats(self) -> Dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN alive=1 THEN 1 ELSE 0 END) as alive,
                    AVG(fitness) as avg_fitness,
                    MAX(fitness) as max_fitness
                FROM species_history
                """
            ).fetchone()
        if row:
            total, alive, avg_f, max_f = row
            return {
                "total": total or 0,
                "alive": alive or 0,
                "avg_fitness": float(avg_f or 0.0),
                "max_fitness": float(max_f or 0.0),
            }
        return {"total": 0, "alive": 0, "avg_fitness": 0.0, "max_fitness": 0.0}

    # ------------------------------------------------------------------
    # World DNA
    # ------------------------------------------------------------------

    def save_world_dna(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO world_dna (id, ts, params_json)
                VALUES (1, ?, ?)
                """,
                (time.time(), json.dumps(self.world_dna.to_dict())),
            )

    def load_world_dna(self) -> None:
        with self._connect() as conn:
            row = conn.execute("SELECT params_json FROM world_dna WHERE id = 1").fetchone()
        if row:
            try:
                self.world_dna = WorldDNA.from_dict(json.loads(row[0]))
            except Exception:
                pass

    def evolve_world(self) -> None:
        self.world_dna.mutate(strength=0.05)
        self.save_world_dna()
        self._event("world_evolve", json.dumps(self.world_dna.to_dict()))

    # ------------------------------------------------------------------
    # Timeline
    # ------------------------------------------------------------------

    def timeline(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT ts, event_type, detail FROM timeline
                ORDER BY ts DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [{"ts": r[0], "type": r[1], "detail": r[2]} for r in rows]

    def _flush_events(self) -> None:
        if not self._events:
            return
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO timeline (ts, event_type, detail)
                VALUES (?, ?, ?)
                """,
                [(e["ts"], e["type"], e["detail"]) for e in self._events],
            )
        self._events.clear()

    # ------------------------------------------------------------------
    # Long-term fitness
    # ------------------------------------------------------------------

    def long_term_fitness(self, lineage_id: str, current_fitness: float, adaptability: float) -> float:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT AVG(fitness), COUNT(*) as cnt
                FROM species_history
                WHERE lineage = ? AND alive = 1
                """,
                (lineage_id,),
            ).fetchone()
        if row and row[1] and row[1] > 0:
            historical = max(0.0, min(1.0, float(row[0])))
        else:
            historical = current_fitness

        adaptability = max(0.0, min(1.0, float(adaptability)))
        current = max(0.0, min(1.0, float(current_fitness)))

        return (
            current * 0.5 +
            historical * 0.3 +
            adaptability * 0.2
        )

    # ------------------------------------------------------------------
    # Dream mode
    # ------------------------------------------------------------------

    def dream(self, genome_factory, count: int = 3) -> List[Any]:
        """Slowly mutate and explore new forms when audio is quiet."""
        dreams = []
        for _ in range(count):
            g = genome_factory()
            g.mutate(strength=0.02)
            dreams.append(g)
        self._event("dream", f"generated={count}")
        return dreams

    # ------------------------------------------------------------------
    # Species revival
    # ------------------------------------------------------------------

    def revive_species(self, music_sig: MusicSignature, threshold: float = 0.75) -> Optional[Dict[str, Any]]:
        matches = self.find_similar_music(music_sig, threshold=threshold, limit=1)
        if matches:
            self._event("revive", f"section={music_sig.section}")
            return matches[0]
        return None

    # ------------------------------------------------------------------
    # Session persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> bool:
        try:
            data = {
                "world_dna": self.world_dna.to_dict(),
                "champions": self.archive.champions[:20],
                "extinct_count": len(self.archive.extinct_species),
                "mutation_count": len(self.archive.mutations),
                "session_start": self._session_start,
                "saved_ts": time.time(),
                "species_stats": self.species_stats(),
            }
            Path(path).write_text(json.dumps(data, indent=2, default=str))
            self.save_world_dna()
            self._flush_events()
            return True
        except Exception as e:
            logger = __import__("logging").getLogger(__name__)
            logger.warning(f"Failed to save ecosystem: {e}")
            return False

    def load(self, path: Union[str, Path]) -> bool:
        try:
            data = json.loads(Path(path).read_text())
            if "world_dna" in data:
                self.world_dna = WorldDNA.from_dict(data["world_dna"])
            self.load_world_dna()
            return True
        except Exception as e:
            logger = __import__("logging").getLogger(__name__)
            logger.warning(f"Failed to load ecosystem: {e}")
            return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _serialize_genome(self, genome: Any) -> Dict[str, Any]:
        if hasattr(genome, "to_dict"):
            return genome.to_dict()
        if isinstance(genome, dict):
            return dict(genome)
        return {"_raw": str(genome)}
