from __future__ import annotations

from dataclasses import dataclass
import random



def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


@dataclass
class EntityDNA:
    """Physics DNA (evolvable laws)."""

    gravity: float = 0.3
    attraction: float = 0.8
    decay: float = 0.1
    growth: float = 0.6
    mass: float = 0.5
    speed: float = 0.8

    # simple audio targeting genes
    bass_affinity: float = 1.0
    treble_affinity: float = 0.5
    mid_affinity: float = 0.7

    colour: float = 0.5

    @staticmethod
    def random() -> "EntityDNA":
        return EntityDNA(
            gravity=random.uniform(-0.6, 0.8),
            attraction=random.uniform(0.0, 1.2),
            decay=random.uniform(0.02, 0.25),
            growth=random.uniform(0.1, 1.0),
            mass=random.uniform(0.2, 1.2),
            speed=random.uniform(0.2, 1.6),
            bass_affinity=random.uniform(0.0, 1.5),
            treble_affinity=random.uniform(0.0, 1.5),
            mid_affinity=random.uniform(0.0, 1.5),
            colour=random.random(),
        )

    def mutate(self, strength: float = 0.1) -> None:
        s = float(strength)
        if random.random() < 0.5:
            self.gravity += random.uniform(-s, s)
        if random.random() < 0.5:
            self.attraction += random.uniform(-s, s)
        if random.random() < 0.5:
            self.decay += random.uniform(-s, s)
        if random.random() < 0.5:
            self.growth += random.uniform(-s, s)
        if random.random() < 0.4:
            self.speed += random.uniform(-s, s)
        if random.random() < 0.4:
            self.mass += random.uniform(-s, s)

        if random.random() < 0.5:
            self.bass_affinity += random.uniform(-s, s)
        if random.random() < 0.5:
            self.mid_affinity += random.uniform(-s, s)
        if random.random() < 0.5:
            self.treble_affinity += random.uniform(-s, s)

        if random.random() < 0.35:
            self.colour = _clamp(self.colour + random.uniform(-s, s), 0.0, 1.0)

        # clamp to safe ranges
        self.gravity = _clamp(self.gravity, -1.0, 1.5)
        self.attraction = _clamp(self.attraction, -0.5, 2.0)
        self.decay = _clamp(self.decay, 0.0, 0.5)
        self.growth = _clamp(self.growth, 0.0, 2.0)
        self.speed = _clamp(self.speed, 0.05, 3.0)
        self.mass = _clamp(self.mass, 0.05, 3.0)
        self.bass_affinity = _clamp(self.bass_affinity, 0.0, 3.0)
        self.mid_affinity = _clamp(self.mid_affinity, 0.0, 3.0)
        self.treble_affinity = _clamp(self.treble_affinity, 0.0, 3.0)


@dataclass
class Entity:
    """Organism instance living in the world grid."""

    x: float
    y: float
    vx: float
    vy: float

    dna: EntityDNA

    # lifecycle
    life: float = 100.0
    age: float = 0.0
    energy: float = 1.0

    def pos_int(self, width: int, height: int) -> tuple[int, int]:
        xi = int(self.x) % int(width)
        yi = int(self.y) % int(height)
        return xi, yi

    def is_dead(self) -> bool:
        return self.life <= 0.0

    def step_lifecycle(self, dt: float, global_energy: float) -> None:
        dt = float(dt)
        self.age += dt

        # energy drains by decay, grows from environment
        self.energy += (global_energy * self.dna.growth - self.energy * self.dna.decay) * dt

        # life tied to energy
        life_drain = dt * (1.0 / (1.0 + max(0.0, self.energy)))
        self.life -= life_drain

    def maybe_split(self, dt: float) -> bool:
        """MVP split rule: high internal energy causes division."""
        dt = float(dt)
        # probability scaled by growth and dt
        p = max(0.0, (self.energy - 2.0) * 0.02 * self.dna.growth)
        if random.random() < p:
            # split consumes energy
            self.energy *= 0.5
            return True
        return False

