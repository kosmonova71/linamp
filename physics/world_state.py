from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WorldState:
    """Renderer-facing world state container.

    MVP notes:
      - energy is a real CPU grid (Field.energy)
      - other fields are derived placeholders so shaders have something stable
    """

    width: int
    height: int

    energy: list
    temperature: list
    density: list
    flow: list

    age: float = 0.0

    audio_bass: float = 0.0
    audio_mid: float = 0.0
    audio_treble: float = 0.0

    def sample_field_energy(self, x: float, y: float) -> float:
        xi = int(x) % self.width
        yi = int(y) % self.height
        return float(self.energy[yi][xi])

    def sample_field_gradient(self, x: float, y: float) -> tuple[float, float]:
        """Central difference gradient on a torus grid."""
        xi = int(x) % self.width
        yi = int(y) % self.height

        left = float(self.energy[yi][(xi - 1) % self.width])
        right = float(self.energy[yi][(xi + 1) % self.width])
        up = float(self.energy[(yi - 1) % self.height][xi])
        down = float(self.energy[(yi + 1) % self.height][xi])

        gx = (right - left) * 0.5
        gy = (down - up) * 0.5
        return float(gx), float(gy)

    def update_derived(self, entities_count: int = 0) -> None:
        # MVP derived placeholders: cheap, but react to population density.
        dens = float(entities_count) / max(1.0, float(self.width * self.height))
        for y in range(self.height):
            for x in range(self.width):
                self.density[y][x] = dens
                self.temperature[y][x] = float(self.energy[y][x]) * 0.5 + dens * 0.2
                self.flow[y][x] = self.temperature[y][x]

