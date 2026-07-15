from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AudioState:
    """Smoothed audio features for reactive rendering.

    Stores both raw band levels and derived smoothed energy/beat.
    """

    # Smoothed bands
    bass: float = 0.0
    mid: float = 0.0
    treble: float = 0.0

    # Derived smoothed signals
    energy: float = 0.0
    beat: float = 0.0

    def set_levels(self, levels: dict) -> None:
        """Update from raw FFT-derived levels.

        Expected keys:
          - bass, mid, treb/treble
          - beat (optional)

        Uses exponential smoothing to avoid jitter.
        """
        smoothing = float(levels.get("smoothing", 0.85))

        raw_bass = float(levels.get("bass", 0.0))
        raw_mid = float(levels.get("mid", 0.0))
        # allow both treble/treb
        raw_treble = float(levels.get("treble", levels.get("treb", 0.0)))
        raw_beat = float(levels.get("beat", 0.0))

        # Clamp raw inputs first (avoid NaNs in shader uniforms)
        raw_bass = max(0.0, min(2.0, raw_bass))
        raw_mid = max(0.0, min(2.0, raw_mid))
        raw_treble = max(0.0, min(2.0, raw_treble))
        raw_beat = max(0.0, min(2.0, raw_beat))

        # Exponential smoothing (continuous control signal)
        self.bass = self.bass * smoothing + raw_bass * (1.0 - smoothing)
        self.mid = self.mid * smoothing + raw_mid * (1.0 - smoothing)
        self.treble = self.treble * smoothing + raw_treble * (1.0 - smoothing)

        # Derived energy (weighted sum)
        self.energy = self.bass * 0.5 + self.mid * 0.3 + self.treble * 0.2

        # Smoothed beat: keep a responsive but stable accumulator
        beat_smoothing = 0.80
        self.beat = self.beat * beat_smoothing + raw_beat * (1.0 - beat_smoothing)

        # Defensive clamp
        self.bass = max(0.0, min(2.0, float(self.bass)))
        self.mid = max(0.0, min(2.0, float(self.mid)))
        self.treble = max(0.0, min(2.0, float(self.treble)))
        self.energy = max(0.0, min(2.0, float(self.energy)))
        self.beat = max(0.0, min(2.0, float(self.beat)))


