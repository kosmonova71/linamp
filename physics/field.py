from __future__ import annotations

from dataclasses import dataclass

from typing import List


def _gaussian_blur_approx(x: List[List[float]]) -> List[List[float]]:
    """Cheap diffusion/blur substitute.

    Uses separable 3-tap kernel repeated once.
    Keeps runtime low (Stage 6 MVP).
    """
    h = len(x)
    if h == 0:
        return x
    w = len(x[0]) if h else 0
    if w == 0:
        return x

    k0, k1, k2 = 0.25, 0.5, 0.25

    # Horizontal blur
    hx = [[0.0 for _ in range(w)] for _ in range(h)]
    for y in range(h):
        for xi in range(w):
            xl = (xi - 1) % w
            xr = (xi + 1) % w
            hx[y][xi] = k0 * x[y][xl] + k1 * x[y][xi] + k2 * x[y][xr]

    # Vertical blur
    out = [[0.0 for _ in range(w)] for _ in range(h)]
    for yi in range(h):
        for xi in range(w):
            yu = (yi - 1) % h
            yd = (yi + 1) % h
            out[yi][xi] = k0 * hx[yu][xi] + k1 * hx[yi][xi] + k2 * hx[yd][xi]

    return out



@dataclass
class Field:
    width: int
    height: int
    energy: List[List[float]]

    def __init__(self, width: int, height: int):
        self.width = int(width)
        self.height = int(height)
        self.energy = [[0.0 for _ in range(self.width)] for _ in range(self.height)]


    def diffuse(self, amount: float = 0.35) -> None:
        """Diffuse energy (blur + blend)."""
        amount = float(amount)
        amount = max(0.0, min(1.0, amount))
        blurred = _gaussian_blur_approx(self.energy)
        # blend old + blurred
        for y in range(self.height):
            row = self.energy[y]
            b_row = blurred[y]
            for x in range(self.width):
                row[x] = (1.0 - amount) * row[x] + amount * b_row[x]


    def inject(self, x: int, y: int, amount: float) -> None:
        """Inject energy at integer cell coordinates."""
        xi = int(x) % self.width
        yi = int(y) % self.height
        self.energy[yi][xi] = float(self.energy[yi][xi] + float(amount))


    def summary(self) -> dict[str, float]:
        total = 0.0
        maxv = 0.0
        count = 0
        for row in self.energy:
            for v in row:
                total += float(v)
                if float(v) > maxv:
                    maxv = float(v)
                count += 1
        mean = total / float(count) if count else 0.0
        return {"energy_mean": float(mean), "energy_max": float(maxv)}


