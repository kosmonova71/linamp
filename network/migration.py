from __future__ import annotations

import random
from typing import Any


class MigrationRules:
    """Migration policy: avoid random upload everything.

    Stage 9 first pass uses reputation+fitness ordering and stochastic
    selection to preserve diversity.
    """

    def __init__(
        self,
        *,
        top_percent: float = 1.0,
        rare_percent: float = 5.0,
        unexpected_percent: float = 2.0,
        max_payload: int = 20,
        rng_seed: int | None = None,
    ):
        self.top_percent = float(top_percent)
        self.rare_percent = float(rare_percent)
        self.unexpected_percent = float(unexpected_percent)
        self.max_payload = int(max_payload)
        self.random = random.Random(rng_seed)

    def select_payload(self, *, candidates: list[tuple[str, Any]], get_reputation_score, get_novelty, get_unexpected) -> list[tuple[str, Any]]:
        """candidates are (species_id, record) pairs."""
        if not candidates:
            return []

        scored = []
        for sid, rec in candidates:
            rep = float(get_reputation_score(rec))
            scored.append((sid, rec, rep, float(get_novelty(rec)), float(get_unexpected(rec))))

        # Sort by reputation score
        scored.sort(key=lambda t: t[2], reverse=True)

        n = len(scored)
        top_n = max(1, int(round(n * (self.top_percent / 100.0))))
        rare_n = max(1, int(round(n * (self.rare_percent / 100.0))))
        unexpected_n = max(0, int(round(n * (self.unexpected_percent / 100.0))))

        picked_ids: set[str] = set()
        payload: list[tuple[str, Any]] = []

        # top 1%
        for sid, rec, _, _, _ in scored[:top_n]:
            if sid in picked_ids:
                continue
            payload.append((sid, rec))
            picked_ids.add(sid)
            if len(payload) >= self.max_payload:
                return payload

        # rare discoveries: choose high novelty among remaining
        remaining = [t for t in scored if t[0] not in picked_ids]
        remaining.sort(key=lambda t: t[3], reverse=True)
        for sid, rec, _, _, _ in remaining[:rare_n]:
            if sid in picked_ids:
                continue
            payload.append((sid, rec))
            picked_ids.add(sid)
            if len(payload) >= self.max_payload:
                return payload

        # unexpected behaviour: heuristic picks high unexpected metric
        remaining = [t for t in scored if t[0] not in picked_ids]
        remaining.sort(key=lambda t: t[4], reverse=True)
        for sid, rec, _, _, _ in remaining[:unexpected_n]:
            if sid in picked_ids:
                continue
            payload.append((sid, rec))
            picked_ids.add(sid)
            if len(payload) >= self.max_payload:
                return payload

        # diversity filler: random from remaining
        remaining = [t for t in scored if t[0] not in picked_ids]
        self.random.shuffle(remaining)
        for sid, rec, _, _, _ in remaining:
            if len(payload) >= self.max_payload:
                break
            payload.append((sid, rec))

        return payload

