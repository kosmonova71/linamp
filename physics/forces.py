from __future__ import annotations

import random
from dataclasses import dataclass
from .particles import Entity




@dataclass
class Forces:
    """Compute forces from world + audio and apply to entities."""

    def apply(self, entity: Entity, world: object, dt: float) -> None:

        # Gravity-like pull (toward +y or -y depending on sign)
        dt = float(dt)
        g = float(entity.dna.gravity)
        entity.vy += g * dt * 0.6

        # Attraction: lightweight tendency toward higher field energy
        # (sample gradient approx using neighbor cells)
        fx, fy = world.sample_field_gradient(entity.x, entity.y)
        entity.vx += fx * entity.dna.attraction * dt * entity.dna.speed
        entity.vy += fy * entity.dna.attraction * dt * entity.dna.speed

        # Audio-driven excitation: push based on audio targeting genes
        audio_bias = (
            world.audio_bass * entity.dna.bass_affinity
            + world.audio_mid * entity.dna.mid_affinity
            + world.audio_treble * entity.dna.treble_affinity
        )
        # Add a small random-walk term to keep exploration emergent
        entity.vx += (random.uniform(-1.0, 1.0) * 0.03 + fx * 0.02) * audio_bias * dt
        entity.vy += (random.uniform(-1.0, 1.0) * 0.03 + fy * 0.02) * audio_bias * dt

        # simple damping
        entity.vx *= (1.0 - 0.15 * dt)
        entity.vy *= (1.0 - 0.15 * dt)

        # integrate
        entity.x += entity.vx * dt
        entity.y += entity.vy * dt

        # wrap in torus grid space
        entity.x %= world.width
        entity.y %= world.height

