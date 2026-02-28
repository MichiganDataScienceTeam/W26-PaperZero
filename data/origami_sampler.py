from framework import ContextSampler
from paper import Vec2, Segment, Paper, Segment, Vec2
import numpy as np
import random
import numpy.typing as npt
from typing import Tuple, Dict, Any

class OrigamiSampler(ContextSampler):
    """
    Stateless sampler for generating 2D origami tasks at a target level.
    """

    def __init__(self,
                 max_fold_attempts: int = 20,
                 max_paper_retries: int = 10):
        """
        Args:
            max_fold_attempts: How many times to try finding a VALID fold line before giving up on a step.
            max_paper_retries: How many times to restart the entire paper if we get stuck.
        """

        self.max_fold_attempts = max_fold_attempts
        self.max_paper_retries = max_paper_retries

    def sample(self, level: int) -> Dict[str, Any]:
        """
        Generates a folded paper task with up to `level` successful folds.

        Args:
            level: Maximum number of fold steps to attempt.

        Returns:
            Dict with keys:
            "total_action": np.ndarray of shape (4 * actual_folds,) containing
                flattened fold endpoints [x1, y1, x2, y2, ...].
            "actual_folds": Number of successful folds applied.
            "final_paper": Folded Paper state after sampling.
        """

        paper = Paper()
        total_action = []

        for _ in range(level):
            for _ in range(self.max_paper_retries):
                curr_layers = len(paper.layers)
                fold_array = self._generate_single_fold(paper)
                if len(paper.layers) != curr_layers:
                    total_action.append(fold_array)
                    break

        total_action_array = (
            np.concatenate(total_action)
            if total_action
            else np.array([], dtype=np.float64)
        )

        return {
            "total_action": total_action_array,
            "actual_folds": len(total_action),
            "final_paper": paper
        }

    
    def _generate_single_fold(self, paper: Paper) -> npt.NDArray:
        EPSILON = 1e-10
        BOUNDARY_DIST = 0.01
        pts, idx = paper.compute_boundary_points(BOUNDARY_DIST)
        n = len(idx)-1
        
        # Sample S1 and P1
        s1 = random.randint(0, n-1)
        p1 = pts[random.randint(idx[s1], idx[s1+1]-1)]
        
        # Exclude S1 and its endpoints
        banned, excl = [pts[idx[s1]], pts[idx[s1+1]-1]], {s1}
        
        for b in banned:
            if (p1[0]-b[0])**2 + (p1[1]-b[1])**2 < EPSILON:
                starts, ends = pts[idx[:-1]], pts[idx[1:]-1]
                
                bad_indices = np.where((np.sum((starts-p1)**2, axis=1) < EPSILON) |
                                    (np.sum((ends-p1)**2, axis=1) < EPSILON))[0]
                
                excl.update(bad_indices)
                for i in bad_indices: 
                    banned.extend([starts[i], ends[i]])

                break

        # Sample S2 and P2
        for _ in range(self.max_fold_attempts):
            while (s2 := random.randint(0, n-1)) in excl: pass
            start, end = idx[s2], idx[s2+1]
            p2 = pts[random.randint(start, end-1)]
            for b in banned:
                if (p2[0]-b[0])**2 + (p2[1]-b[1])**2 < EPSILON:
                    break
            else:
                try:
                    if paper.fold(Segment(Vec2(p1[0], p1[1]), Vec2(p2[0], p2[1]))):
                        return np.concatenate([p1, p2])
                except:
                    pass
        
        return np.array([])
