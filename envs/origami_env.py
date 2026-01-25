from framework import Environment
from paper import Vec2, Segment, Paper

import numpy as np

import numpy.typing as npt
from typing import Tuple, Dict, Union


class OrigamiEnv(Environment):
    """
    RL environment for 2D Origami

    Args:
        resolution: the tuple describing the shape of the 2D rastered target
        max_steps: the max steps an episode can run for before termination
    """

    def __init__(self, resolution: Tuple[int, int], max_steps: int):
        self.res = resolution
        self.max_steps = max_steps
        
        # Default placeholder states
        self.reset({
            "base_paper": Paper(),
            "target_mask": np.ones(self.res, dtype=bool)
        })

    def reset(self, context: Dict[str, Union[Paper, npt.NDArray]]) -> npt.NDArray[np.float32]:
        """
        Resets and initializes the environment with the provided context
        and returns the the resulting observation

        Args:
            context: The context {"base_paper": Paper, "target_mask": NDArray[bool]}
        """
        self.paper: Paper = context["base_paper"].copy()         # type: ignore
        self.target_mask: npt.NDArray = context["target_mask"]   # type: ignore
        self.target_image = self.target_mask.astype(np.float32)[None, ...]
        self.current_step = 0
        
        current_mask = self.paper.rasterize(*self.res)
        
        return self._get_obs(current_mask)

    def step(self, action: npt.NDArray[np.float32]) -> Tuple[npt.NDArray[np.float32], float, bool, bool, Dict]:
        """
        Performs a single fold and returns the resulting state tuple
        with (obs, reward, done, truncated, info)

        Args:
            action: (4,) float array [x1, y1, x2, y2]
        """
        # Update step
        self.current_step += 1

        # Compute fold
        valid_fold = False
        try:
            valid_fold = self.paper.fold(
                Segment(
                    Vec2(action[0], action[1]),
                    Vec2(action[2], action[3])
                )
            )
        except Exception:
            valid_fold = False # Be super duper sure it's not True
        current_mask = self.paper.rasterize(*self.res).astype(bool)
        
        # Check termination
        iou = self._calculate_iou(current_mask, self.target_mask)
        done = iou > 0.85
        truncated = self.current_step >= self.max_steps

        # Compute reward
        reward = 0 if valid_fold else -0.1
        if done or truncated:
            reward += iou + done
            
        return self._get_obs(current_mask), reward, done, truncated, {}

    def _get_obs(self, current_mask: npt.NDArray[np.bool_]) -> npt.NDArray[np.float32]:
        """
        Constructs the (2, H, W) observation tensor

        Args:
            current_mask: the 2D boolean mask returned by rasterizing the paper
        """
        return np.stack([current_mask, self.target_mask]).astype(np.float32)

    def _calculate_iou(self, mask1: npt.NDArray[np.bool_], mask2: npt.NDArray[np.bool_]) -> float:
        """
        Computes the IoU (intersection over union) of two 2D binary masks
        of equal shape. Note this is also commutative because union and
        intersection are both commutative.

        Args:
            mask1: one of the masks
            mask2: the other mask
        """
        intersection = (mask1 & mask2).sum()
        union = (mask1 | mask2).sum()
        return float(intersection / union) if union > 0 else 0

