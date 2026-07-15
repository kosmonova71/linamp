from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Any


@dataclass
class ShaderDNA:
    """Stage-4 shader DNA.

    This is an intermediate representation that is safe to mutate.
    We never mutate raw GLSL strings.

    operations is an ordered list of gene dicts.
    """

    operations: list[dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def random_initial(max_ops: int = 6) -> "ShaderDNA":
        # A stable minimal species: start with 3 genes.
        ops = [
            {"op": "noise", "scale": random.uniform(2.0, 10.0)},
            {"op": "warp", "amount": random.uniform(0.05, 0.35)},
            {"op": "colour", "mode": random.randint(0, 2)},
        ]
        # Optionally extend.
        for _ in range(random.randint(0, max_ops - len(ops))):
            op = random.choice(["noise", "warp", "feedback", "distort", "fractal", "colour"])
            ops.append(ShaderDNA._random_gene(op=op))

        # Ensure at least one feedback to let persistence do something.
        if not any(g.get("op") == "feedback" for g in ops):
            ops.insert(0, {"op": "feedback", "decay": random.uniform(0.75, 0.97)})

        return ShaderDNA(operations=ops[:max_ops])

    @staticmethod
    def _random_gene(op: str) -> dict[str, Any]:
        if op == "noise":
            return {"op": "noise", "scale": random.uniform(2.0, 12.0)}
        if op == "warp":
            return {"op": "warp", "amount": random.uniform(0.03, 0.4)}
        if op == "feedback":
            return {"op": "feedback", "decay": random.uniform(0.7, 0.99)}
        if op == "distort":
            return {"op": "distort", "strength": random.uniform(0.01, 0.25)}
        if op == "fractal":
            return {"op": "fractal", "iter": random.randint(2, 6)}
        if op == "colour":
            return {"op": "colour", "mode": random.randint(0, 2)}
        # Fallback: treat unknown as no-op noise.
        return {"op": "noise", "scale": random.uniform(2.0, 10.0)}

