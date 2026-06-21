from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data.origami_sampler import OrigamiSampler
from diffusion.data.pipeline import PackedTrajectoryDataset, PipelineConfig, TrajectoryExample
from diffusion.models.unet import UNetDenoiser
from diffusion.utils import get_device
from paper import Paper, Segment, Vec2


@dataclass(frozen=True)
class TrainConfig:
    num_trajectories: int = 64
    level: int = 3
    horizon: int = 15
    image_size: int = 64
    timesteps: int = 100
    min_snr_gamma: float | None = 5.0
    batch_size: int = 8
    steps: int = 120
    lr: float = 1e-4
    weight_decay: float = 1e-4
    base_ch: int = 32
    time_dim: int = 128
    seed: int = 2026
    x0_loss_weight: float = 0.1


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_trajectory(sampler: OrigamiSampler, level: int) -> TrajectoryExample:
    sample = None
    while sample is None or int(sample["actual_folds"]) < 2:
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


def build_dataset(cfg: TrainConfig, device: torch.device) -> PackedTrajectoryDataset:
    sampler = OrigamiSampler(max_fold_attempts=80, max_paper_retries=30)
    trajectories = [_make_trajectory(sampler, level=cfg.level) for _ in range(cfg.num_trajectories)]
    pipe_cfg = PipelineConfig(
        horizon=cfg.horizon,
        timesteps=cfg.timesteps,
        state_mode="raw",
        image_size=cfg.image_size,
        noise_target="v",
        min_snr_gamma=cfg.min_snr_gamma,
    )
    return PackedTrajectoryDataset(trajectories, cfg=pipe_cfg, device=device)


def _masked_weighted_mse(pred: Tensor, target: Tensor, denoise_mask: Tensor, loss_weight: Tensor) -> Tensor:
    squared = (pred - target).pow(2) * denoise_mask
    per_sample_num = squared.flatten(start_dim=1).sum(dim=1)
    per_sample_den = denoise_mask.flatten(start_dim=1).sum(dim=1).clamp_min(1.0)
    per_sample = per_sample_num / per_sample_den
    return (per_sample * loss_weight).mean()


def _expand_schedule(values: Tensor, t: Tensor, x: Tensor) -> Tensor:
    """
    Gather scalar schedule values at batch timesteps and broadcast to token shape.
    """
    gathered = values.to(device=t.device)[t]
    out = gathered.to(device=x.device, dtype=x.dtype)
    while out.ndim < x.ndim:
        out = out.unsqueeze(-1)
    return out


def run_training(cfg: TrainConfig, device: str | torch.device | None = None) -> dict[str, object]:
    _seed_everything(cfg.seed)
    dev = get_device(device)

    dataset = build_dataset(cfg, device=dev)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    iterator = iter(dataloader)

    model = UNetDenoiser(
        token_dim=dataset.token_dim,
        state_dim=dataset.state_dim,
        base_ch=cfg.base_ch,
        time_dim=cfg.time_dim,
    ).to(dev)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Fixed batch for before/after sanity metric.
    fixed = next(iter(dataloader))
    fx_t, ft, ftarget, fvalid, fpin, fdenoise, _ = fixed
    fx_t = fx_t.to(dev)
    ft = ft.to(dev, dtype=torch.long)
    ftarget = ftarget.to(dev)
    fvalid = fvalid.to(dev)
    fpin = fpin.to(dev)
    fdenoise = fdenoise.to(dev)

    model.eval()
    with torch.no_grad():
        fpred0 = model(fx_t, ft, fvalid, fpin)
        fixed_mse_before = float(_masked_weighted_mse(fpred0, ftarget, fdenoise, torch.ones(fx_t.shape[0], device=dev)))

    losses: list[float] = []
    model.train()
    for step in range(cfg.steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)

        x_t, t, target, slot_valid, slot_pin, denoise_mask, loss_weight = batch
        x_t = x_t.to(dev)
        t = t.to(dev, dtype=torch.long)
        target = target.to(dev)
        slot_valid = slot_valid.to(dev)
        slot_pin = slot_pin.to(dev)
        denoise_mask = denoise_mask.to(dev)
        loss_weight = loss_weight.to(dev)

        pred = model(x_t, t, slot_valid, slot_pin)
        v_loss = _masked_weighted_mse(pred, target, denoise_mask, loss_weight)

        # Auxiliary x0 loss for cleaner denoising trajectory.
        alpha_t = _expand_schedule(dataset.alphas, t, x_t)
        sigma_t = _expand_schedule(dataset.sigmas, t, x_t)
        x0_pred = alpha_t * x_t - sigma_t * pred
        x0_target = alpha_t * x_t - sigma_t * target
        x0_loss = _masked_weighted_mse(x0_pred, x0_target, denoise_mask, loss_weight)

        loss = v_loss + float(cfg.x0_loss_weight) * x0_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))

    model.eval()
    with torch.no_grad():
        fpred1 = model(fx_t, ft, fvalid, fpin)
        fixed_mse_after = float(_masked_weighted_mse(fpred1, ftarget, fdenoise, torch.ones(fx_t.shape[0], device=dev)))

    return {
        "device": str(dev),
        "losses": losses,
        "fixed_mse_before": fixed_mse_before,
        "fixed_mse_after": fixed_mse_after,
        "dataset_size": len(dataset),
        "token_dim": dataset.token_dim,
        "state_dim": dataset.state_dim,
        "model": model,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal fast diffusion trainer (raw-state).")
    parser.add_argument("--device", type=str, default=None, help="cuda, mps, cpu, or leave unset for auto")
    parser.add_argument("--num-trajectories", type=int, default=64)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--base-ch", type=int, default=32)
    parser.add_argument("--time-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--x0-loss-weight", type=float, default=0.1)
    parser.add_argument("--no-min-snr", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = TrainConfig(
        num_trajectories=args.num_trajectories,
        steps=args.steps,
        batch_size=args.batch_size,
        image_size=args.image_size,
        horizon=args.horizon,
        timesteps=args.timesteps,
        base_ch=args.base_ch,
        time_dim=args.time_dim,
        lr=args.lr,
        seed=args.seed,
        x0_loss_weight=args.x0_loss_weight,
        min_snr_gamma=None if args.no_min_snr else 5.0,
    )
    out = run_training(cfg, device=args.device)
    losses = out["losses"]
    print("device:", out["device"])
    print("dataset_size:", out["dataset_size"])
    print("token_dim:", out["token_dim"])
    print("state_dim:", out["state_dim"])
    print("loss_first:", f"{losses[0]:.6f}")
    print("loss_last:", f"{losses[-1]:.6f}")
    print("fixed_mse_before:", f"{out['fixed_mse_before']:.6f}")
    print("fixed_mse_after:", f"{out['fixed_mse_after']:.6f}")


if __name__ == "__main__":
    main()
