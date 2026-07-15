from __future__ import annotations

from creativity.archive import ConceptArchaeology


class Fitness:
    """Fitness scoring for an organism.

    Stage 8 (Option 1): introduce novelty via concept archaeology.
    Stage 13: add intelligence fitness (neural brain reward/penalty ratio).

    Note: still uses audio/time proxies for "visual quality" because there is
    no image readback.
    """

    def __init__(self, concept_archive: ConceptArchaeology | None = None):
        self.concept_archive = concept_archive

    def evaluate(
        self,
        frame: float,
        audio,
        *,
        novelty: float | None = None,
    ) -> float:
        bass = float(getattr(audio, "bass", 0.0))
        mid = float(getattr(audio, "mid", 0.0))
        treble = float(getattr(audio, "treble", 0.0))

        motion = (0.5 + 0.5 * __import__("math").sin(frame * 0.25)) * (
            0.4 + 0.6 * (bass + mid + treble) / 3.0
        )
        audio_sync = bass * (
            0.7 + 0.3 * __import__("math").sin(frame * 0.6 + treble * 2.0)
        )
        complexity = 0.3 * mid + 0.5 * treble

        novelty_term = float(novelty) if novelty is not None else 0.0

        return float(motion + audio_sync + complexity + novelty_term)

    def intelligence_fitness(
        self,
        *,
        brain_fitness: float = 0.0,
        visual_quality: float = 0.0,
        audio_adaptation: float = 0.0,
        survival: float = 0.0,
        cooperation: float = 0.0,
        novelty: float = 0.0,
    ) -> float:
        """Fitness now measures intelligence.

        fitness = visual_quality * 0.25 + audio_adaptation * 0.25
                  + survival * 0.20 + cooperation * 0.15 + novelty * 0.15
        """
        raw = (
            0.25 * float(visual_quality)
            + 0.25 * float(audio_adaptation)
            + 0.20 * float(survival)
            + 0.15 * float(cooperation)
            + 0.15 * float(novelty)
        )
        blended = raw * 0.75 + 0.25 * min(1.0, max(0.0, float(brain_fitness)))
        return float(max(0.0, min(1.0, blended)))


