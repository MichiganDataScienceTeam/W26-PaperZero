import math
import torch
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from paper import Paper, Segment, Vec2
from data.origami_sampler import OrigamiSampler
from diffusion.vae import VAE, TUNED_IMG_SIZE, TUNED_LATENT_DIM


STATE_DIM = TUNED_LATENT_DIM  # 1024
ACTION_DIM = 4                # (x1, y1, x2, y2)
IMG_SIZE = TUNED_IMG_SIZE     # 128


def _replay_folds(actions: np.ndarray, level: int):
    """
    Replay fold actions on a fresh Paper, capturing every intermediate state.

    Args:
        actions: flat array of shape (level * 4,) with concatenated fold coords
        level: number of folds

    Returns:
        List of (level + 1) Paper objects: [paper_before_fold_0, ..., paper_after_all_folds]
    """
    action_pairs = actions.reshape(level, 4)
    paper = Paper()
    states = [paper.copy()]

    for a in action_pairs:
        seg = Segment(Vec2(float(a[0]), float(a[1])),
                       Vec2(float(a[2]), float(a[3])))
        paper.fold(seg)
        states.append(paper.copy())

    return states


def _rasterize_states(states, img_size: int = IMG_SIZE) -> np.ndarray:
    """
    Rasterize a list of Paper objects into a single numpy array.

    Returns:
        np.ndarray of shape (num_states, 1, img_size, img_size), float32 in [0, 1]
    """
    images = []
    for p in states:
        img = p.rasterize(img_size, img_size).astype(np.float32)
        images.append(img[np.newaxis, :, :])  # add channel dim
    return np.stack(images)


@torch.no_grad()
def _encode_states(images: np.ndarray, vae: VAE, device: torch.device) -> Tensor:
    """
    Encode rasterized images through the VAE encoder.

    Args:
        images: (num_states, 1, H, W) float32 array
        vae: VAE model (used in eval mode, no grad)
        device: torch device

    Returns:
        Tensor of shape (num_states, STATE_DIM)
    """
    x = torch.from_numpy(images).to(device)
    mu, _ = vae.encode_stats(x)
    return mu


def _interleave(latents: Tensor, actions: np.ndarray, level: int) -> Tensor:
    """
    Interleave state latents and action vectors into a flat trajectory.

    Layout: [s0 | a0 | s1 | a1 | ... | a_{k-1} | s_k]
    Dim:    (level + 1) * STATE_DIM + level * ACTION_DIM

    Returns:
        1-D tensor of the trajectory
    """
    action_pairs = torch.from_numpy(
        actions.reshape(level, ACTION_DIM).astype(np.float32)
    ).to(latents.device)

    parts = []
    for i in range(level):
        parts.append(latents[i])         # s_i  (STATE_DIM,)
        parts.append(action_pairs[i])    # a_i  (ACTION_DIM,)
    parts.append(latents[level])         # s_k  (STATE_DIM,)

    return torch.cat(parts)


def generate_trajectories(
    n_samples: int,
    level: int = 3,
    vae: VAE | None = None,
    device: torch.device | None = None,
) -> Tensor:
    """
    Generate a dataset of interleaved state-action trajectory tensors.

    Args:
        n_samples: number of trajectories to generate
        level: number of folds per trajectory
        vae: a VAE instance (randomly initialized is fine); if None one is created
        device: torch device

    Returns:
        Tensor of shape (n_samples, traj_dim) where
        traj_dim = (level+1)*STATE_DIM + level*ACTION_DIM
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available()
                              else "mps" if torch.backends.mps.is_available()
                              else "cpu")

    if vae is None:
        vae = VAE()
    vae = vae.to(device).eval()

    sampler = OrigamiSampler(max_fold_attempts=50, max_paper_retries=20)
    trajectories = []

    while len(trajectories) < n_samples:
        data = sampler.sample(level=level)

        if data["actual_folds"] != level:
            continue

        states = _replay_folds(data["total_action"], level)
        images = _rasterize_states(states)
        latents = _encode_states(images, vae, device)
        traj = _interleave(latents, data["total_action"], level)
        trajectories.append(traj)

        if len(trajectories) % 100 == 0:
            print(f"  Generated {len(trajectories)}/{n_samples} trajectories")

    return torch.stack(trajectories).cpu()


# ---------------------------------------------------------------------------
# Mask
# ---------------------------------------------------------------------------

def build_mask(level: int = 3,
               state_dim: int = STATE_DIM,
               action_dim: int = ACTION_DIM) -> Tensor:
    """
    Build binary mask M for noising.

    M = 0 for first state (s0) and last state (s_k) positions.
    M = 1 everywhere else (intermediate states + all actions).

    Returns:
        1-D bool tensor of length (level+1)*state_dim + level*action_dim
    """
    traj_dim = (level + 1) * state_dim + level * action_dim
    mask = torch.ones(traj_dim)

    # s0 sits at the very beginning
    mask[:state_dim] = 0.0

    # s_k sits at the very end
    mask[-state_dim:] = 0.0

    return mask


# ---------------------------------------------------------------------------
# Cosine noise schedule
# ---------------------------------------------------------------------------

def cosine_schedule(T: int) -> tuple[Tensor, Tensor]:
    """
    Cosine noise schedule satisfying alpha^2 + sigma^2 = 1.

    alpha_t = cos(t / T * pi / 2)   decreases 1 -> 0
    sigma_t = sin(t / T * pi / 2)   increases 0 -> 1

    Args:
        T: total number of diffusion timesteps

    Returns:
        (alphas, sigmas) each of shape (T + 1,)
    """
    steps = torch.linspace(0, 1, T + 1)
    alphas = torch.cos(steps * math.pi / 2)
    sigmas = torch.sin(steps * math.pi / 2)
    return alphas, sigmas


# ---------------------------------------------------------------------------
# Noising
# ---------------------------------------------------------------------------

def add_noise(x_0: Tensor, mask: Tensor, alpha_t: float | Tensor,
              sigma_t: float | Tensor) -> Tensor:
    """
    Apply forward diffusion noise to a trajectory.

    x_t = (1 - M) * x_0  +  M * (alpha_t * x_0 + sigma_t * eps)

    Args:
        x_0:      clean trajectory, shape (..., traj_dim)
        mask:     binary mask, broadcastable to x_0
        alpha_t:  scalar or per-sample alpha
        sigma_t:  scalar or per-sample sigma

    Returns:
        Noised trajectory x_t with same shape as x_0, and the sampled noise eps
    """
    eps = torch.randn_like(x_0)
    noised = alpha_t * x_0 + sigma_t * eps
    x_t = (1.0 - mask) * x_0 + mask * noised
    return x_t, eps


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DiffusionDataset(Dataset):
    """
    PyTorch Dataset that wraps pre-generated trajectories and returns
    (x_0, x_t, eps, t, mask) tuples with a randomly sampled timestep.
    """

    def __init__(self, trajectories: Tensor, T: int = 1000, level: int = 3):
        """
        Args:
            trajectories: (N, traj_dim) clean trajectory tensor
            T: number of diffusion timesteps
            level: number of folds (used to build the mask)
        """
        self.trajectories = trajectories
        self.T = T
        self.mask = build_mask(level)
        self.alphas, self.sigmas = cosine_schedule(T)

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int):
        x_0 = self.trajectories[idx]

        t = torch.randint(1, self.T + 1, (1,)).item()
        alpha_t = self.alphas[t]
        sigma_t = self.sigmas[t]

        x_t, eps = add_noise(x_0, self.mask, alpha_t, sigma_t)

        return x_0, x_t, eps, t, self.mask


if __name__ == "__main__":
    LEVEL = 3
    N = 50
    T = 100

    traj_dim = (LEVEL + 1) * STATE_DIM + LEVEL * ACTION_DIM
    print(f"Level={LEVEL}, traj_dim={traj_dim}")
    print(f"  states: {LEVEL + 1} x {STATE_DIM} = {(LEVEL + 1) * STATE_DIM}")
    print(f"  actions: {LEVEL} x {ACTION_DIM} = {LEVEL * ACTION_DIM}")

    print(f"\nGenerating {N} trajectories...")
    trajectories = generate_trajectories(N, level=LEVEL)
    print(f"Trajectories shape: {trajectories.shape}")

    mask = build_mask(LEVEL)
    print(f"Mask shape: {mask.shape}, masked fraction: {mask.mean():.3f}")

    alphas, sigmas = cosine_schedule(T)
    print(f"Schedule: alpha[0]={alphas[0]:.4f}, alpha[T]={alphas[-1]:.4f}")
    print(f"          sigma[0]={sigmas[0]:.4f}, sigma[T]={sigmas[-1]:.4f}")
    print(f"          alpha^2 + sigma^2 = {(alphas**2 + sigmas**2)[0]:.6f} (should be 1)")

    ds = DiffusionDataset(trajectories, T=T, level=LEVEL)
    x_0, x_t, eps, t, m = ds[0]
    print(f"\nDataset sample: x_0={x_0.shape}, x_t={x_t.shape}, eps={eps.shape}, t={t}")
    print(f"  x_0[0:5] (s0, should be clean): {x_0[:5]}")
    print(f"  x_t[0:5] (s0, should match x_0): {x_t[:5]}")
    print(f"  x_0[-5:] (s3, should be clean): {x_0[-5:]}")
    print(f"  x_t[-5:] (s3, should match x_0): {x_t[-5:]}")
