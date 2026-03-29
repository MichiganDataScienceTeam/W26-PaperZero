from __future__ import annotations

import numpy as np

from data.origami_sampler import OrigamiSampler
from diffusion.data.pipeline import PackedTrajectoryDataset, PipelineConfig, TrajectoryExample
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
    trajectory1 = _make_trajectory(level=3)
    trajectory2 = _make_trajectory(level=4)
    cfg = PipelineConfig(
        state_mode="vae",
        timesteps=100,
    )
    dataset = PackedTrajectoryDataset([trajectory1, trajectory2], cfg=cfg, device="cpu")
    x_t, t, target, slot_valid_mask, slot_pin_mask = dataset[0]

    print("Config check:")
    print("dataset_size:", len(dataset))
    print("state_mode:", cfg.state_mode)
    print("slot_valid_mask:", tuple(slot_valid_mask.shape))
    print("slot_pin_mask:", tuple(slot_pin_mask.shape))

    print("\nFirst trajectory:")
    print("folds:", int(trajectory1.actions.shape[0]))
    print("x_t:", tuple(x_t.shape))
    print("t:", int(t))
    print("target:", tuple(target.shape))


if __name__ == "__main__":
    main()
