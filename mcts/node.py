from __future__ import annotations

from paper import Paper, Segment
import numpy as np
import matplotlib.pyplot as plt
import torch
import math
from mcts.train import get_model
from mcts.temp_segment_finder import find_segments

from typing import Any, List, Tuple
import numpy.typing as npt


TRAINING_RECORD_SCHEMA_VERSION = 1
RASTER_RESOLUTION = 128


def _segment_key(seg: Segment, decimals: int = 6) -> Tuple[float, float, float, float]:
    """Deterministic directed action identity for one fold segment."""
    return (
        round(float(seg.p1.x), decimals),
        round(float(seg.p1.y), decimals),
        round(float(seg.p2.x), decimals),
        round(float(seg.p2.y), decimals),
    )


def _point_to_grid_index(
    x: float,
    y: float,
    bounds: Tuple[float, float, float, float],
    height: int,
    width: int,
) -> Tuple[int, int]:
    """Map global-coordinate point to raster index using paper bounds."""
    min_x, max_x, min_y, max_y = bounds
    span_x = max(max_x - min_x, 1e-12)
    span_y = max(max_y - min_y, 1e-12)

    xn = (x - min_x) / span_x
    yn = (y - min_y) / span_y

    xi = min(max(int(round(xn * (width - 1))), 0), width - 1)
    yi = min(max(int(round(yn * (height - 1))), 0), height - 1)
    return yi, xi


def _build_state_tensor(paper: Paper, target_mask: npt.NDArray[np.float32]) -> torch.Tensor:
    current = np.array(paper.rasterize(RASTER_RESOLUTION, RASTER_RESOLUTION, 0.0), dtype=np.float32).reshape(
        RASTER_RESOLUTION, RASTER_RESOLUTION
    )
    assert target_mask.shape == (
        RASTER_RESOLUTION,
        RASTER_RESOLUTION,
    ), f"target_mask must be {(RASTER_RESOLUTION, RASTER_RESOLUTION)}, got {target_mask.shape}"
    state = np.stack([current, target_mask], axis=0)
    assert state.shape == (
        2,
        RASTER_RESOLUTION,
        RASTER_RESOLUTION,
    ), f"state must be {(2, RASTER_RESOLUTION, RASTER_RESOLUTION)}, got {state.shape}"
    return torch.tensor(state, dtype=torch.float32).unsqueeze(0)


class Node:
    def __init__(
        self,
        paper: Paper,
        parent: Node | None = None,
        segment: Segment | None = None,
        target_mask: npt.NDArray | None = None,
    ):
        super().__init__()

        # tree
        self.parent: Node | None = parent
        self.children: List[Node] = []

        # data
        self.paper: Paper = paper
        self.segment: Segment | None = segment
        if target_mask is None:
            self.target_mask = np.ones((RASTER_RESOLUTION, RASTER_RESOLUTION), dtype=np.float32)
        else:
            self.target_mask = np.array(target_mask, dtype=np.float32).reshape(RASTER_RESOLUTION, RASTER_RESOLUTION)
        
        self.N = 0      # number of visits
        self.W = 0.0    # sum of all values
        self.Q = self.W / self.N if self.N != 0 else 0.0    # W/N (0 if N = 0)
        self.P = 0.0 if not self.parent else self.parent.P     # prior
        self.action_key: Tuple[float, float, float, float] | None = (
            None if segment is None else _segment_key(segment)
        )

        # constants
        self.sumB = None

    # For testing purposes
    def render(self):
        img = self.paper.rasterize(128, 128, 0.0)
        img = np.array(img)
        plt.imshow(img, cmap="gray", origin="lower")
        plt.text(0, -17, f'Visits: {self.N}\nPrior: {self.P}\nValue: {self.W}', fontsize=12)
        plt.show()

    def expand(self):
        """Make Children"""
        model = get_model()
        # Generate list of segments
        segments = find_segments(self.paper)
        if len(segments) == 0:
            return

        # Deduplicate actions by deterministic key.
        unique_segments: list[Segment] = []
        seen: set[Tuple[float, float, float, float]] = set()
        for seg in segments:
            key = _segment_key(seg)
            if key in seen:
                continue
            seen.add(key)
            unique_segments.append(seg)

        assert self.target_mask.shape == (
            RASTER_RESOLUTION,
            RASTER_RESOLUTION,
        ), f"target_mask shape mismatch: {self.target_mask.shape}"

        model.eval()
        with torch.no_grad():
            img_tensor = _build_state_tensor(self.paper, self.target_mask)
            policy, value = model(img_tensor)

        policy = policy.squeeze(0)  # (2, H, W)

        start_logits = policy[0]
        end_logits   = policy[1]

        start_probs = torch.softmax(start_logits.flatten(), dim=0).view_as(start_logits)
        end_probs   = torch.softmax(end_logits.flatten(), dim=0).view_as(end_logits)

        priors = []
        h, w = start_probs.shape
        min_x, max_x, min_y, max_y = self.paper.compute_bounds()
        bounds = (float(min_x), float(max_x), float(min_y), float(max_y))

        for seg in unique_segments:
            pt1, pt2 = seg.p1, seg.p2

            y1, x1 = _point_to_grid_index(float(pt1.x), float(pt1.y), bounds, h, w)
            y2, x2 = _point_to_grid_index(float(pt2.x), float(pt2.y), bounds, h, w)

            pr1 = start_probs[y1, x1]
            pr2 = end_probs[y2, x2]

            priors.append((pr1 * pr2).item())

        # Normalize
        priors = np.array(priors)
        if len(priors) == 0:
            priors = np.array([])
        elif priors.sum() > 0:
            priors /= priors.sum()
        else:
            priors = np.ones_like(priors) / len(priors)

        # Create children - expand
        for i, P in enumerate(priors):
            copyP = self.paper.copy()
            if not copyP.fold(unique_segments[i]):
                continue
            child = Node(
                paper=copyP,
                parent=self,
                segment=unique_segments[i],
                target_mask=self.target_mask,
            )
            child.P = P
            self.children.append(child)
    
        # Evaluate & Backprop - update W, N, Q for all visited nodes
        node: Node = self
        while node is not None:
            node.W += value.item()
            node.N += 1
            node.Q = node.W / node.N
            node = node.parent


    def select(self, c=1):
        # Make sure there's children
        if len(self.children) == 0:
            return None # We're done (leaf node)

        self.sumB = sum(child.N for child in self.children)

        # Uses PUCT instead
        def puct_score(self, child, c):
            explore = c * child.P * math.sqrt(self.sumB + 1) / (1 + child.N)
            return child.Q + explore

        return max(self.children, key=lambda child: puct_score(self, child, c))

    def export_training_record(self) -> dict[str, Any]:
        """
        Stable sample schema for downstream training scripts.
        This captures the model input state plus action identities.
        """
        state = _build_state_tensor(self.paper, self.target_mask).squeeze(0).cpu().numpy()
        action_keys = [child.action_key for child in self.children]
        priors = [float(child.P) for child in self.children]

        return {
            "schema_version": TRAINING_RECORD_SCHEMA_VERSION,
            "state": state,
            "target_mask": np.array(self.target_mask, dtype=np.float32),
            "action_keys": action_keys,
            "priors": np.array(priors, dtype=np.float32),
            "value": float(self.Q),
            "visits": int(self.N),
        }
    
if __name__ == "__main__":
    paper = Paper()
    root = Node(paper=paper, target_mask=np.ones((128, 128), dtype=np.float32))

    root.expand()
    new_node = root.select()
    if new_node is not None:
        new_node.render()   # Check if None

        new_node.expand()
        new_node2 = new_node.select()
        if new_node2 is not None:
            new_node2.render()  # Check if None
