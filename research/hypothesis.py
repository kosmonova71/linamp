from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Hypothesis:
    """A proposed causal relationship between a genome change and a visual outcome.

    Attributes:
        change: Machine-readable identifier for the mutation to apply.
            Examples: "increase_feedback_recursion", "add_fractal_module",
            "increase_feedback", "boost_particle_birth_rate".
        prediction: Natural-language description of the expected outcome.
        target_parameter: Optional genome param to watch for correlation.
        min_experiments: How many runs before the hypothesis can be promoted
            to a theory or discarded.
        priority: Scheduling priority. Higher means the researcher tries it sooner.
    """

    change: str
    prediction: str
    target_parameter: str = ""
    min_experiments: int = 100
    priority: float = 0.0
    attempts: int = 0
    successes: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "change": self.change,
            "prediction": self.prediction,
            "target_parameter": self.target_parameter,
            "min_experiments": self.min_experiments,
            "priority": self.priority,
            "attempts": self.attempts,
            "successes": self.successes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Hypothesis:
        return cls(
            change=str(data.get("change", "")),
            prediction=str(data.get("prediction", "")),
            target_parameter=str(data.get("target_parameter", "")),
            min_experiments=int(data.get("min_experiments", 100)),
            priority=float(data.get("priority", 0.0)),
            attempts=int(data.get("attempts", 0)),
            successes=int(data.get("successes", 0)),
        )
