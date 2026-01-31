"""
To run this script, do NOT try to run this file as a standalone
script (ex. python baselines/train_sl.py)

Instead, run it as a module:
python -m baselines.train_sl

This is not because the code itself is special, just that it's
in a directory and imports sibling modules as a standalone script
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from torch.utils.data import TensorDataset, DataLoader
from data.origami_sampler import OrigamiSampler 


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Running on: {DEVICE}")


# Data Config
LEVEL = 1
TRAIN_SIZE = 10_000
TEST_SIZE  = 5_000
REFRESH_RATE = 5

# Training Config
RES = 64
BATCH_SIZE = 256
TOTAL_EPOCHS = 100
LR = 3e-4


def make_coord_grid(batch_size, res):
    x = torch.linspace(0, 1, res, device=DEVICE)
    y = torch.linspace(0, 1, res, device=DEVICE)
    y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([x_grid, y_grid], dim=0).unsqueeze(0).repeat(batch_size, 1, 1, 1)

class SLPolicy(nn.Module):
    def __init__(self, channels=2):
        super().__init__()
        
        # Pre-compute coordinate grid once
        self.register_buffer("grid", self._make_coord_grid(RES))
        
        self.features = nn.Sequential(
            # Input channels + 2 for (x, y) grid
            nn.Conv2d(channels + 2, 32, 5, stride=2, padding=2),
            nn.BatchNorm2d(32), nn.ReLU(),
            
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Flatten()
        )
        
        # Run a dummy pass to find flat size cause I'm not calculating that
        with torch.no_grad():
            dummy_input = torch.zeros(1, channels + 2, RES, RES)
            flat_size = self.features(dummy_input).shape[1]
        
        output_dim = LEVEL * 4
        self.fc = nn.Sequential(
            nn.Linear(flat_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def _make_coord_grid(self, res):
        x = torch.linspace(0, 1, res)
        y = torch.linspace(0, 1, res)
        y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([x_grid, y_grid], dim=0).unsqueeze(0)

    def forward(self, x):
        B = x.shape[0]
        grid = self.grid.expand(B, -1, -1, -1)
        
        x_aug = torch.cat([x, grid], dim=1)
        return self.fc(self.features(x_aug))


def generate_batch(n_samples, label):
    print(f"GENERATING {label} ({n_samples} samples, Level {LEVEL})...", end=" ", flush=True)
    sampler = OrigamiSampler((RES, RES), max_fold_attempts=200)
    obs_list, act_list = [], []
    
    count = 0
    t_start = time.time()
    
    while count < n_samples:
        data = sampler.sample(level=LEVEL)
        if data is None: continue
            
        obs = np.stack([
            data["base_paper"].rasterize(RES, RES),
            data["target_mask"]
        ]).astype(np.float32)
        
        act = data["total_action"].flatten().astype(np.float32)
        
        obs_list.append(obs)
        act_list.append(act)
        count += 1
    
    dt = time.time() - t_start
    print(f"Done in {dt:.1f}s")
    
    X = torch.tensor(np.array(obs_list))
    Y = torch.tensor(np.array(act_list))
    return TensorDataset(X, Y)


def main():
    model = SLPolicy().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    # Static Test Set
    test_ds = generate_batch(TEST_SIZE, "Test Set (Fixed)")
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"STARTING TRAINING LOOP")
    
    train_dl = None
    
    for epoch in range(1, TOTAL_EPOCHS + 1):
        # Refresh training data sometimes
        if (epoch - 1) % REFRESH_RATE == 0:
            train_ds = generate_batch(TRAIN_SIZE, f"Train Set (Epoch {epoch})")
            train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        
        # Train
        t0 = time.time()
        model.train()
        train_loss = 0
        train_samples = 0
        
        for obs, target in train_dl:
            obs, target = obs.to(DEVICE), target.to(DEVICE)
            
            pred = model(obs)
            loss = criterion(pred, target)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            train_loss += loss.item() * len(obs)
            train_samples += len(obs)
            
        # Test
        model.eval()
        test_loss = 0
        test_samples = 0
        
        with torch.no_grad():
            for obs, target in test_dl:
                obs, target = obs.to(DEVICE), target.to(DEVICE)
                pred = model(obs)
                loss = criterion(pred, target)
                test_loss += loss.item() * len(obs)
                test_samples += len(obs)
        
        avg_train = train_loss / train_samples
        avg_test  = test_loss / test_samples
        fps = int(train_samples / (time.time() - t0))
        
        print(f"Epoch {epoch:03d} | Train Loss: {avg_train:.6f} | Test Loss: {avg_test:.6f} | FPS: {fps}")


if __name__ == "__main__":
    main()

