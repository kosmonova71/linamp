from __future__ import annotations

import os
import json
import random



from .field import Field
from .particles import Entity, EntityDNA
from .forces import Forces
from .world_state import WorldState


class Ecosystem:
    def __init__(
        self,
        width: int,
        height: int,
        seed_entities: int = 16,
        max_entities: int = 96,
    ):
        self.width = int(width)
        self.height = int(height)

        self.field = Field(self.width, self.height)
        self.forces = Forces()

        self.entities: list[Entity] = []
        self.max_entities = int(max_entities)

        for _ in range(int(seed_entities)):
            self.entities.append(
                Entity(
                    x=random.uniform(0, self.width),
                    y=random.uniform(0, self.height),
                    vx=0.0,
                    vy=0.0,
                    dna=EntityDNA.random(),
                )
            )

        self.world_state = WorldState(
            width=self.width,
            height=self.height,
            energy=self.field.energy,
            temperature=[[0.0 for _ in range(self.width)] for _ in range(self.height)],
            density=[[0.0 for _ in range(self.width)] for _ in range(self.height)],
            flow=[[0.0 for _ in range(self.width)] for _ in range(self.height)],

            age=0.0,
        )

        self.age = 0.0

    def inject_from_audio(self, bass: float, mid: float, treble: float) -> None:
        # Map audio to field pressure and inject bursts.
        # MVP: inject at random hotspots; future: inject at entity positions.
        bass = float(bass)
        mid = float(mid)
        treble = float(treble)

        pressure = 0.35 * bass + 0.20 * mid + 0.15 * treble
        vib = 0.10 * treble
        beat = 0.25 * bass * max(0.0, treble)

        n_bursts = 2 + int(max(0.0, pressure) * 2.0)
        for _ in range(n_bursts):
            x = random.randrange(self.width)
            y = random.randrange(self.height)
            self.field.inject(x, y, amount=pressure + vib)

        # occasional beat burst
        if random.random() < 0.02 + float(bass) * 0.01:
            x = random.randrange(self.width)
            y = random.randrange(self.height)
            self.field.inject(x, y, amount=beat + pressure * 0.5)

    def step(self, dt: float, audio_bass: float, audio_mid: float, audio_treble: float) -> None:
        dt = float(dt)
        self.age += dt
        self.world_state.age = self.age

        # store audio for force stage
        self.world_state.audio_bass = float(audio_bass)
        self.world_state.audio_mid = float(audio_mid)
        self.world_state.audio_treble = float(audio_treble)

        # audio -> field energy
        self.inject_from_audio(audio_bass, audio_mid, audio_treble)

        # field diffusion
        self.field.diffuse(amount=0.25 + 0.15 * min(1.0, float(audio_bass)))

        # entity dynamics
        # mean energy (manual mean; no numpy)
        if self.field.energy:
            total = 0.0
            count = 0
            for row in self.field.energy:
                for v in row:
                    total += float(v)
                    count += 1
            global_energy_mean = float(total / float(count)) if count else 0.0
        else:
            global_energy_mean = 0.0


        # apply forces and lifecycle
        new_entities: list[Entity] = []
        for e in list(self.entities):
            self.forces.apply(e, self.world_state, dt)
            e.step_lifecycle(dt, global_energy=global_energy_mean)

            # split
            if len(self.entities) + len(new_entities) < self.max_entities and e.maybe_split(dt):
                child_dna = EntityDNA(
                    gravity=e.dna.gravity,
                    attraction=e.dna.attraction,
                    decay=e.dna.decay,
                    growth=e.dna.growth,
                    mass=e.dna.mass,
                    speed=e.dna.speed,
                    bass_affinity=e.dna.bass_affinity,
                    mid_affinity=e.dna.mid_affinity,
                    treble_affinity=e.dna.treble_affinity,
                    colour=e.dna.colour,
                )
                # mutate physics laws slightly
                child_dna.mutate(strength=0.15)

                child = Entity(
                    x=e.x,
                    y=e.y,
                    vx=-e.vx * 0.3,
                    vy=-e.vy * 0.3,
                    dna=child_dna,
                    life=60.0,
                    age=0.0,
                    energy=max(0.5, e.energy),
                )
                new_entities.append(child)

            # extinction
            if e.is_dead():
                continue

        self.entities = [e for e in self.entities if not e.is_dead()]

        # competition (MVP): cap density by random cull if too many
        if len(self.entities) > self.max_entities:
            self.entities = random.sample(self.entities, self.max_entities)

        self.entities.extend(new_entities)

        # derived fields for shader interface (MVP)
        # world_state.update_derived() is cheap; it uses energy only.
        self.world_state.update_derived(entities_count=len(self.entities))


    def world_summaries(self) -> dict[str, float]:
        e = self.field.energy
        total = 0.0
        maxv = 0.0
        count = 0
        for row in e:
            for v in row:
                vv = float(v)
                total += vv
                if vv > maxv:
                    maxv = vv
                count += 1
        mean = total / float(count) if count else 0.0

        return {
            "world_energy_mean": float(mean),
            "world_energy_max": float(maxv),
            "world_entity_count": float(len(self.entities)),
            "world_age": float(self.world_state.age),
        }


    def save_world(self, root: str, world_name: str = "earth_001") -> None:
        base = os.path.join(root, world_name)
        os.makedirs(base, exist_ok=True)

        physics_path = os.path.join(base, "physics.json")
        species_path = os.path.join(base, "species.json")

        # physics.json: store energy stats + seed dimensions
        payload_physics = {
            "width": self.width,
            "height": self.height,
            "age": self.world_state.age,
            "energy_mean": float(self.world_summaries().get("world_energy_mean", 0.0)),
            "energy_max": float(self.world_summaries().get("world_energy_max", 0.0)),

            # MVP: do not store full grid yet.
        }
        with open(physics_path, "w", encoding="utf-8") as f:
            json.dump(payload_physics, f)

        # species.json: store entity DNA summary
        payload_species = {
            "entities": [
                {
                    "x": e.x,
                    "y": e.y,
                    "life": e.life,
                    "dna": e.dna.__dict__,
                }
                for e in self.entities
            ]
        }
        with open(species_path, "w", encoding="utf-8") as f:
            json.dump(payload_species, f)

