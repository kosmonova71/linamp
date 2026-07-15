from __future__ import annotations

import random
from typing import Any

from .shader_dna import ShaderDNA


class MutationEngine:
    """Mutation engine for ShaderDNA.

    Operators:
      - change_value: tweak parameters of an existing gene
      - insert: append a new gene
      - remove: delete a gene (but never empty)
      - duplicate: duplicate a gene

    The key safety constraint is: keep DNA within bounds and clamp numeric values.
    """

    def __init__(
        self,
        max_ops: int = 10,
        min_ops: int = 2,
        value_strength: float = 0.12,
    ):
        self.max_ops = int(max_ops)
        self.min_ops = int(min_ops)
        self.value_strength = float(value_strength)

    def mutate(self, dna: ShaderDNA) -> ShaderDNA:
        # Mutate a copy (avoid in-place corruption for population bookkeeping)
        new_ops = [dict(g) for g in dna.operations]
        if not new_ops:
            return ShaderDNA.random_initial(max_ops=self.max_ops)

        action = random.choice(["change_value", "insert", "remove", "duplicate"])

        if action == "change_value":
            gene = random.choice(new_ops)
            self._change_gene_value(gene)

        elif action == "insert":
            if len(new_ops) < self.max_ops:
                op = random.choice(["noise", "warp", "feedback", "distort", "fractal", "colour"])
                new_ops.append(ShaderDNA._random_gene(op=op))
            else:
                # If full, do a value change instead.
                gene = random.choice(new_ops)
                self._change_gene_value(gene)

        elif action == "remove":
            if len(new_ops) > self.min_ops:
                new_ops.pop(random.randrange(len(new_ops)))
            else:
                gene = random.choice(new_ops)
                self._change_gene_value(gene)

        elif action == "duplicate":
            if len(new_ops) < self.max_ops:
                idx = random.randrange(len(new_ops))
                new_ops.insert(idx, dict(new_ops[idx]))
            else:
                gene = random.choice(new_ops)
                self._change_gene_value(gene)

        # Clamp after mutation.
        new_ops = new_ops[: self.max_ops]
        if len(new_ops) < self.min_ops:
            # Re-pad with stable genes.
            while len(new_ops) < self.min_ops:
                new_ops.append(ShaderDNA._random_gene(op=random.choice(["noise", "warp", "colour"])) )

        # Ensure at least one feedback for persistence.
        if not any(g.get("op") == "feedback" for g in new_ops):
            new_ops.insert(0, ShaderDNA._random_gene(op="feedback"))

        return ShaderDNA(operations=new_ops)

    def _change_gene_value(self, gene: dict[str, Any]) -> None:
        op = gene.get("op")
        s = self.value_strength

        if op == "noise":
            gene["scale"] = self._clamp(gene.get("scale", 6.0) + random.uniform(-s, s) * 10.0, 1.0, 18.0)
        elif op == "warp":
            gene["amount"] = self._clamp(gene.get("amount", 0.2) + random.uniform(-s, s) * 0.5, 0.0, 0.6)
        elif op == "feedback":
            gene["decay"] = self._clamp(gene.get("decay", 0.9) + random.uniform(-s, s) * 0.1, 0.5, 0.99)
        elif op == "distort":
            gene["strength"] = self._clamp(gene.get("strength", 0.1) + random.uniform(-s, s) * 0.2, 0.0, 0.5)
        elif op == "fractal":
            gene["iter"] = int(self._clamp(int(gene.get("iter", 4)) + random.choice([-1, 1]), 1, 8))
        elif op == "colour":
            # Usually keep small discrete domain.
            if random.random() < 0.4:
                gene["mode"] = int((int(gene.get("mode", 0)) + random.choice([-1, 1])) % 3)
        else:
            # Unknown op: replace with noise.
            gene.clear()
            gene.update(ShaderDNA._random_gene(op="noise"))

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(x)))

