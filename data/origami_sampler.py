from framework import ContextSampler
from paper import Vec2, Segment, Paper
import numpy as np
import numpy.typing as npt
from typing import Tuple, Dict, Any, Optional

class OrigamiSampler(ContextSampler):
    """
    A robust, stateless sampler for generating origami tasks at specific difficulty levels.
    
    Guarantees:
    - If sample(k) returns a task, it has EXACTLY k folds.
    - It will never return a task with < k folds (no fallback/partial credit).
    """

    def __init__(self,
                 resolution: Tuple[int, int], 
                 max_fold_attempts: int = 200,
                 max_paper_retries: int = 10):
        """
        Args:
            resolution: (width, height) tuple for rasterization.
            max_fold_attempts: How many times to try finding a VALID fold line before giving up on a step.
            max_paper_retries: How many times to restart the entire paper if we get stuck.
        """

        self.res: Tuple[int, int] = resolution
        self.max_fold_attempts = max_fold_attempts
        self.max_paper_retries = max_paper_retries

    def sample(self, level: int) -> Optional[Dict[str, Any]]:
        """
        Generates a task with EXACTLY `level` folds.
        
        Args:
            level: The integer number of folds required.
            
        Returns:
            Dict containing the task data, or None if generation failed.
        """
        # Try to generate the full paper N times
        for _ in range(self.max_paper_retries):
            result = self._generate_single_paper(level)
            if result is not None:
                return result
        
        return None

    def _generate_single_paper(self, target_folds: int) -> Optional[Dict[str, Any]]:
        base_paper = Paper()
        target_paper = base_paper.copy()
        actions = []

        for _ in range(target_folds):
            points = target_paper.compute_boundary_points(0.01)
            if len(points) < 2:
                return None

            fold_found = False
            for _ in range(self.max_fold_attempts):
                i, j = np.random.randint(0, len(points), size=2)
                if i == j:
                    continue

                p1, p2 = points[i], points[j]
                if np.linalg.norm(p1 - p2) < 1e-3:
                    continue

                fold_line = Segment(Vec2(*p1), Vec2(*p2))
                before = len(target_paper.layers)

                try:
                    ok = target_paper.fold(fold_line)
                except Exception:
                    continue

                if not ok or len(target_paper.layers) <= before:
                    continue

                actions.append(np.concatenate([p1, p2]).astype(np.float32))
                fold_found = True
                break

            if not fold_found:
                return None

        # === REPLAY VERIFICATION (critical) ===
        replay = base_paper.copy()
        try:
            for a in actions:
                replay.fold(Segment(Vec2(a[0], a[1]), Vec2(a[2], a[3])))
        except Exception:
            return None

        if len(replay.layers) != len(target_paper.layers):
            return None

        target_mask = replay.rasterize(*self.res).astype(bool)

        return {
            "base_paper": base_paper,
            "target_mask": target_mask,
            "total_action": np.stack(actions),
            "difficulty": target_folds,
            "target_difficulty": target_folds,
        }


