# Implemented directly in __init__ because it's very short and for convenience

from typing import Tuple, Dict, Any

class ContextSampler:
    def sample(self, level: int) -> Dict[str, Any]: raise NotImplementedError
    def update(self, metrics) -> None: pass


class Environment:
    def reset(self, context: Dict[str, Any]) -> Any: raise NotImplementedError
    def step(self, action) -> Tuple[Any, float, bool, bool, Dict]: raise NotImplementedError

