"""Persistent visual identity model.

Provides:
  - Mood: dynamic control signal
  - Goals: competing long-term objectives
  - Preferences: learned likes/dislikes
  - Memory: event storage with pattern recall
  - VisualIdentity: core identity with identity genome, lineage, dreams
"""

from .state import IdentityGenome, SpeciesLineage, VisualIdentity
from .mood import Mood
from .goals import Goals
from .preferences import Preferences
from .memory import Memory, MemoryEvent

__all__ = [
    "VisualIdentity",
    "IdentityGenome",
    "SpeciesLineage",
    "Mood",
    "Goals",
    "Preferences",
    "Memory",
    "MemoryEvent",
]
