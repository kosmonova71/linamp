from __future__ import annotations

import random

from .modules import MODULES


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _clamp(x: float, lo: float, hi: float) -> float:
    x = float(x)
    return max(lo, min(hi, x))


class VisualGenome:
    """Stage 3 genome.

    Structure (anatomy):
      - ordered list of module names (module DNA)

    Parameters (still used for continuous tuning and stability):
      - warp_strength (used by warp module)
      - feedback (ping-pong persistence multiplier)

    Appearance genes (colour genes):
      - colour_a, colour_b, colour_c as palette inputs (currently simple)

    Behaviour genes (mutation rate):
      - mutation_rate influences how often anatomy changes
    """

    def __init__(self):
        module_names = list(MODULES.keys())
        # Start with a stable minimal species.
        self.modules: list[str] = [
            random.choice(module_names),
            "warp" if "warp" in module_names else random.choice(module_names),
        ]

        # Safe ranges
        self.params: dict[str, float] = {
            "warp_strength": random.uniform(0.25, 0.9),
            "feedback": random.uniform(0.6, 0.95),
            # colour genes (simple palette seeds)
            "colour_a": random.random(),
            "colour_b": random.random(),
            "colour_c": random.random(),
            # behaviour gene
            "mutation_rate": random.uniform(0.05, 0.25),
        }

    def mutate(self, strength: float = 0.12) -> None:
        """Mutation changes both anatomy (module list) and behaviour/params."""

        # Behaviour gene: probability of anatomy-changing mutation.
        if random.random() < float(self.params.get("mutation_rate", 0.12)):
            action = random.choice(["add", "remove", "replace", "tune"])
            self._mutate_anatomy(action=action)
        else:
            # Always do some continuous tuning.
            self._mutate_params(strength=strength)

        # Keep at least one module.
        if not self.modules:
            self.modules.append(random.choice(list(MODULES.keys())))

    def _mutate_anatomy(self, action: str) -> None:
        module_names = list(MODULES.keys())

        if action == "add":
            module = random.choice(module_names)
            self.modules.append(module)

        elif action == "remove":
            if len(self.modules) > 1:
                self.modules.pop(random.randrange(len(self.modules)))

        elif action == "replace":
            index = random.randrange(len(self.modules))
            self.modules[index] = random.choice(module_names)

        elif action == "tune":
            self._mutate_params(strength=0.12)

    def _mutate_params(self, strength: float = 0.12) -> None:
        # Occasional larger jump
        if random.random() < 0.15:
            strength *= 2.0

        # Warp strength
        self.params["warp_strength"] = _clamp01(
            self.params["warp_strength"] + random.uniform(-strength, strength)
        )
        # Feedback must remain < 1 for stability
        self.params["feedback"] = _clamp(
            self.params["feedback"] + random.uniform(-strength, strength), 0.0, 0.99
        )

        # Colour genes (wrap in [0,1])
        for k in ("colour_a", "colour_b", "colour_c"):
            self.params[k] = _clamp01(self.params[k] + random.uniform(-strength, strength))

        # Mutation rate in reasonable band
        self.params["mutation_rate"] = _clamp(
            self.params["mutation_rate"] + random.uniform(-0.05, 0.05), 0.01, 0.6
        )


