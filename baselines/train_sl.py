"""
Supervised learning baseline for fold-action regression.

Run as a module:
python -m baselines.train_sl
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data.origami_sampler import OrigamiSampler
from paper import Paper
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class TrainConfig:
    level: int = 1
    train_size: int = 10_000
    test_size: int = 5_000
    refresh_rate: int = 5
    res: int = 64
    batch_size: int = 256
    total_epochs: int = 100
    lr: float = 3e-4
    max_fold_attempts: int = 200


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SLPolicy(nn.Module):
    def __init__(self, res: int, output_dim: int, channels: int = 2):
        super().__init__()
        self.register_buffer("grid", self._make_coord_grid(res))

        self.features = nn.Sequential(
            nn.Conv2d(channels + 2, 32, 5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, channels + 2, res, res)
            flat_size = self.features(dummy_input).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flat_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def _make_coord_grid(self, res: int) -> torch.Tensor:
        x = torch.linspace(0, 1, res)
        y = torch.linspace(0, 1, res)
        y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([x_grid, y_grid], dim=0).unsqueeze(0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch = obs.shape[0]
        grid = self.grid.expand(batch, -1, -1, -1)
        obs_augmented = torch.cat([obs, grid], dim=1)
        return self.fc(self.features(obs_augmented))


def _to_fixed_action(total_action: Any, action_dim: int) -> np.ndarray:
    action = np.asarray(total_action, dtype=np.float32).reshape(-1)
    if action.size >= action_dim:
        return action[:action_dim]

    padded = np.zeros(action_dim, dtype=np.float32)
    padded[: action.size] = action
    return padded


def sample_to_observation_and_action(
    sample: dict[str, Any], config: TrainConfig
) -> Tuple[np.ndarray, np.ndarray]:
    target_paper = sample["final_paper"]

    base_mask = Paper().rasterize(config.res, config.res)
    target_mask = target_paper.rasterize(config.res, config.res)
    observation = np.stack([base_mask, target_mask]).astype(np.float32)

    action_dim = config.level * 4
    action = _to_fixed_action(sample["total_action"], action_dim)
    return observation, action


def generate_dataset(
    n_samples: int,
    label: str,
    config: TrainConfig,
    sampler: OrigamiSampler | None = None,
) -> TensorDataset:
    print(
        f"GENERATING {label} ({n_samples} samples, Level {config.level})...",
        end=" ",
        flush=True,
    )
    sampler = sampler or OrigamiSampler(max_fold_attempts=config.max_fold_attempts)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []

    started_at = time.time()
    while len(observations) < n_samples:
        sample = sampler.sample(level=config.level)
        obs, act = sample_to_observation_and_action(sample, config)
        observations.append(obs)
        actions.append(act)

    elapsed = time.time() - started_at
    print(f"Done in {elapsed:.1f}s")

    obs_tensor = torch.from_numpy(np.asarray(observations))
    action_tensor = torch.from_numpy(np.asarray(actions))
    return TensorDataset(obs_tensor, action_tensor)


def run_train_epoch(
    model: nn.Module,
    train_dl: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, int]:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for obs, target in train_dl:
        obs, target = obs.to(device), target.to(device)
        pred = model(obs)
        loss = criterion(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = len(obs)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples, total_samples


def run_eval_epoch(
    model: nn.Module,
    test_dl: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for obs, target in test_dl:
            obs, target = obs.to(device), target.to(device)
            pred = model(obs)
            loss = criterion(pred, target)
            batch_size = len(obs)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples


def main() -> None:
    config = TrainConfig()
    device = get_device()
    print(f"Running on: {device}")

    model = SLPolicy(res=config.res, output_dim=config.level * 4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.MSELoss()

    test_ds = generate_dataset(config.test_size, "Test Set (Fixed)", config)
    test_dl = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)
    print("STARTING TRAINING LOOP")

    train_dl: DataLoader | None = None
    for epoch in range(1, config.total_epochs + 1):
        if (epoch - 1) % config.refresh_rate == 0:
            train_ds = generate_dataset(config.train_size, f"Train Set (Epoch {epoch})", config)
            train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)

        assert train_dl is not None
        started_at = time.time()
        avg_train, train_samples = run_train_epoch(model, train_dl, criterion, optimizer, device)
        avg_test = run_eval_epoch(model, test_dl, criterion, device)
        fps = int(train_samples / max(time.time() - started_at, 1e-8))

        print(
            f"Epoch {epoch:03d} | Train Loss: {avg_train:.6f} | "
            f"Test Loss: {avg_test:.6f} | FPS: {fps}"
        )


if __name__ == "__main__":
    main()

