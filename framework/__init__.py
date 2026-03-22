# Implemented directly in __init__ because it's very short and for convenience

from typing import Any, Dict, Tuple

Context = Dict[str, Any]
StepResult = Tuple[Any, float, bool, bool, Dict[str, Any]]


class ContextSampler:
    """
    Base sampler interface for generating training or evaluation contexts.
    """

    def sample(self, level: int) -> Context:
        """
        Samples one context payload for a requested difficulty level.

        Args:
            level: Difficulty or depth requested by the caller.
        """
        raise NotImplementedError

    def update(self, metrics: Dict[str, Any]) -> None:
        """
        Optional hook for adaptive samplers.

        Args:
            metrics: Feedback metrics from training or evaluation.
        """
        pass


class Environment:
    """
    Minimal environment interface with gym-style reset/step semantics.
    """

    def reset(self, context: Context) -> Any:
        """
        Resets environment state from a context payload.

        Args:
            context: Environment initialization payload.
        """
        raise NotImplementedError

    def step(self, action: Any) -> StepResult:
        """
        Applies one action and returns a step tuple.

        Args:
            action: Environment action payload.
        """
        raise NotImplementedError
