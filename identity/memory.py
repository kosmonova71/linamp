"""Memory system.

Stores past experiences as timestamped events and allows pattern-based recall.
This is not human consciousness — it is a software model of persistence.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryEvent:
    ts: float
    event_type: str
    data: dict[str, Any] = field(default_factory=dict)

    def matches(self, pattern: str) -> bool:
        s = json.dumps(self.data, default=str)
        return pattern.lower() in self.event_type.lower() or pattern.lower() in s.lower()


class Memory:
    """Persistent event memory with capped history and pattern recall.

    Stores the last N events. Older events are pruned.
    Persists to disk via JSON lines file.
    """

    def __init__(self, state_path: str | None = None, max_events: int = 10000):
        self.events: list[MemoryEvent] = []
        self.max_events = int(max_events)
        self.state_path = state_path or os.path.join(
            os.path.dirname(__file__), "memory_events.jsonl"
        )
        self._load()

    def remember(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        evt = MemoryEvent(ts=time.time(), event_type=event_type, data=data or {})
        self.events.append(evt)
        self._append_to_disk(evt)
        self._prune()

    def recall(self, pattern: str, limit: int = 50) -> list[MemoryEvent]:
        limit = int(limit)
        matches = [e for e in self.events if e.matches(pattern)]
        return matches[-limit:]

    def recent(self, n: int = 20) -> list[MemoryEvent]:
        n = int(n)
        return self.events[-n:]

    def count(self) -> int:
        return len(self.events)

    def summary(self) -> dict[str, Any]:
        types: dict[str, int] = {}
        for e in self.events:
            types[e.event_type] = types.get(e.event_type, 0) + 1
        return {
            "total": len(self.events),
            "by_type": types,
            "oldest": self.events[0].ts if self.events else None,
            "newest": self.events[-1].ts if self.events else None,
        }

    def _prune(self) -> None:
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events :]

    def _append_to_disk(self, evt: MemoryEvent) -> None:
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            line = json.dumps(
                {"ts": evt.ts, "event_type": evt.event_type, "data": evt.data},
                default=str,
            )
            with open(self.state_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def _load(self) -> None:
        try:
            if not os.path.exists(self.state_path):
                return
            with open(self.state_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        self.events.append(
                            MemoryEvent(
                                ts=float(obj.get("ts", 0.0)),
                                event_type=str(obj.get("event_type", "")),
                                data=obj.get("data", {}),
                            )
                        )
                    except Exception:
                        continue
            self._prune()
        except Exception:
            pass

    def clear(self) -> None:
        self.events.clear()
        try:
            if os.path.exists(self.state_path):
                os.remove(self.state_path)
        except Exception:
            pass

    def __repr__(self) -> str:
        return f"Memory(events={len(self.events)})"
