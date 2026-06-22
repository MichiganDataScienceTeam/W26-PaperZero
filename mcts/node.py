from __future__ import annotations

from paper import Paper, Segment
import numpy as np
import torch
import math
from mcts.segment_finder import find_segments

from typing import TYPE_CHECKING, Any, List, Tuple
import numpy.typing as npt

if TYPE_CHECKING:
    from mcts.encoder import ThinkArchitecture


TRAINING_RECORD_SCHEMA_VERSION = 2
RASTER_RESOLUTION = 128
POLICY_SPLAT_RADIUS = 2
POLICY_SPLAT_SIGMA = 0.8


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
    scale = min((width - 1) / span_x, (height - 1) / span_y)

    x_r = (x - min_x) * scale
    y_r = (height - 1) - (y - min_y) * scale

    xi = min(max(int(round(x_r)), 0), width - 1)
    yi = min(max(int(round(y_r)), 0), height - 1)
    return yi, xi


def _build_state_tensor(paper: Paper, target_mask: npt.NDArray[np.float32]) -> torch.Tensor:
    current = np.array(paper.rasterize(RASTER_RESOLUTION, RASTER_RESOLUTION, 0.0), dtype=np.float32).reshape(
        RASTER_RESOLUTION, RASTER_RESOLUTION
    )
    state = np.stack([current, target_mask], axis=0)
    return torch.tensor(state, dtype=torch.float32).unsqueeze(0)


def _add_gaussian_splat(
    target: npt.NDArray[np.float32],
    y_center: int,
    x_center: int,
    weight: float,
    radius: int = POLICY_SPLAT_RADIUS,
    sigma: float = POLICY_SPLAT_SIGMA,
) -> None:
    h, w = target.shape
    sigma2 = 2.0 * sigma * sigma
    r2 = float(radius * radius)
    for y in range(max(0, y_center - radius), min(h - 1, y_center + radius) + 1):
        for x in range(max(0, x_center - radius), min(w - 1, x_center + radius) + 1):
            dy = float(y - y_center)
            dx = float(x - x_center)
            d2 = dx * dx + dy * dy
            if d2 > r2:
                continue
            target[y, x] += float(weight * np.exp(-d2 / sigma2))


class Node:
    def __init__(
        self,
        paper: Paper,
        model: ThinkArchitecture,
        target_mask: npt.NDArray,
        parent: Node | None = None,
        segment: Segment | None = None,
    ):
        super().__init__()

        # tree
        self.parent: Node | None = parent
        self.children: List[Node] = []

        # data
        self.paper: Paper = paper
        self.segment: Segment | None = segment
        self.target_mask = np.asarray(target_mask, dtype=np.float32).reshape(RASTER_RESOLUTION, RASTER_RESOLUTION)

        # AlphaZero statistics: visit count, value sum, mean value, prior.
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.P = 0.0 if self.parent is None else self.parent.P
        self.action_key: Tuple[float, float, float, float] | None = (
            None if segment is None else _segment_key(segment)
        )
        self.model = model
        self.last_policy_target: npt.NDArray[np.float32] | None = None
        self.last_value_eval: float | None = None
        self.was_expanded: bool = False

    def _expand_from_network_outputs(self, policy_logits: torch.Tensor, value_scalar: float) -> None:
        """Expand using precomputed network outputs for this node state."""
        self.was_expanded = False
        self.last_policy_target = None
        self.last_value_eval = None

        # Generate list of segments
        folds = find_segments(self.paper)
        if not folds:
            return

        # policy_logits is (2, H, W): channel 0 scores fold start points,
        # channel 1 scores fold end points. Normalise each to a spatial prior.
        start_probs = torch.softmax(policy_logits[0].flatten(), dim=0).view_as(policy_logits[0]).detach().cpu().numpy()
        end_probs = torch.softmax(policy_logits[1].flatten(), dim=0).view_as(policy_logits[1]).detach().cpu().numpy()
        h, w = start_probs.shape
        bounds = tuple(float(v) for v in self.paper.compute_bounds())

        candidates: list[tuple[Segment, Paper, float, tuple[int, int, int, int]]] = []
        for segment, folded_paper in folds:
            y1, x1 = _point_to_grid_index(float(segment.p1.x), float(segment.p1.y), bounds, h, w)
            y2, x2 = _point_to_grid_index(float(segment.p2.x), float(segment.p2.y), bounds, h, w)
            prior = float(start_probs[y1, x1] * end_probs[y2, x2])
            candidates.append((segment, folded_paper, prior, (y1, x1, y2, x2)))

        raw_priors = np.array([candidate[2] for candidate in candidates], dtype=np.float32)
        if raw_priors.sum() > 0:
            priors = raw_priors / raw_priors.sum()
        else:
            priors = np.ones_like(raw_priors, dtype=np.float32) / len(raw_priors)

        start_target = np.zeros((h, w), dtype=np.float32)
        end_target = np.zeros((h, w), dtype=np.float32)

        for i, (seg, folded_paper, _, (y1, x1, y2, x2)) in enumerate(candidates):
            child = Node(
                paper=folded_paper,
                model=self.model,
                parent=self,
                segment=seg,
                target_mask=self.target_mask,
            )
            child.P = float(priors[i])
            self.children.append(child)
            _add_gaussian_splat(start_target, y1, x1, float(priors[i]))
            _add_gaussian_splat(end_target, y2, x2, float(priors[i]))

        start_sum = float(start_target.sum())
        end_sum = float(end_target.sum())
        if start_sum > 0:
            start_target /= start_sum
        if end_sum > 0:
            end_target /= end_sum

        dense_target = np.stack([start_target, end_target], axis=0).astype(np.float32)
        self.last_policy_target = dense_target
        self.last_value_eval = float(value_scalar)
        self.was_expanded = True
    
        # Evaluate & Backprop - update W, N, Q for all visited nodes
        node: Node | None = self
        while node is not None:
            node.W += value_scalar
            node.N += 1
            node.Q = node.W / node.N
            node = node.parent

    def expand(self):
        """Make children by running model on current state, then expanding."""
        self.model.eval()
        with torch.no_grad():
            model_device = next(self.model.parameters()).device
            img_tensor = _build_state_tensor(self.paper, self.target_mask).to(model_device)
            policy, value = self.model(img_tensor)
        self._expand_from_network_outputs(policy.squeeze(0), float(value.item()))


    def select(self, c: float = 1.0) -> "Node | None":
        """Return the child maximising the PUCT score, or None at a leaf."""
        if not self.children:
            return None

        explore_scale = c * math.sqrt(sum(child.N for child in self.children) + 1)

        def puct_score(child: Node) -> float:
            return child.Q + explore_scale * child.P / (1 + child.N)

        return max(self.children, key=puct_score)

    def export_training_record(self) -> dict[str, Any]:
        """
        Stable sample schema for downstream training scripts.
        This captures the model input state and dense policy target.
        """
        if self.last_policy_target is None:
            raise RuntimeError("Node has no dense policy target. Call expand() and ensure expansion succeeds.")
        state = _build_state_tensor(self.paper, self.target_mask).squeeze(0).cpu().numpy()

        return {
            "schema_version": TRAINING_RECORD_SCHEMA_VERSION,
            "state": state,
            "policy_target": np.array(self.last_policy_target, dtype=np.float32),
            "value": float(self.Q),
            "visits": int(self.N),
        }
