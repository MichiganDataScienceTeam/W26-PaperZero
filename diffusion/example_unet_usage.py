from __future__ import annotations

import numpy as np
import torch

from data.origami_sampler import OrigamiSampler
from diffusion.data.pipeline import PackedTrajectoryDataset, PipelineConfig, TrajectoryExample
from diffusion.models.unet import UNetDenoiser
from paper import Paper, Segment, Vec2


def _make_trajectory(level: int = 3) -> TrajectoryExample:
    sampler = OrigamiSampler()
    sample = sampler.sample(level=level)

    actual_folds = int(sample["actual_folds"])
    actions = sample["total_action"].reshape(actual_folds, 4).astype(np.float32)

    paper = Paper()
    papers = [paper.copy()]
    for action in actions:
        fold = Segment(
            Vec2(float(action[0]), float(action[1])),
            Vec2(float(action[2]), float(action[3])),
        )
        paper.fold(fold)
        papers.append(paper.copy())

    return TrajectoryExample(papers=papers, actions=actions)


def main() -> None:
    # Trajectory stuff
    trajectory = _make_trajectory(level=3)
    cfg = PipelineConfig(state_mode="vae", timesteps=100)
    dataset = PackedTrajectoryDataset([trajectory], cfg=cfg, device="cpu")

    x_t, t, target, slot_valid_mask, slot_pin_mask = dataset[0]

    # UNet stuff
    model = UNetDenoiser(token_dim=dataset.token_dim, state_dim=dataset.state_dim, base_ch=64, time_dim=128)
    model.eval()

    with torch.no_grad():
        pred = model(
            x_t.unsqueeze(0),
            torch.tensor([t], dtype=torch.long),
            slot_valid_mask.unsqueeze(0),
            slot_pin_mask.unsqueeze(0),
        )

    print("x_t:", tuple(x_t.shape))
    print("target:", tuple(target.shape))
    print("pred:", tuple(pred.shape))
    print("inference_ok:", pred.shape == (1, x_t.shape[0], x_t.shape[1]))


if __name__ == "__main__":
    main()
