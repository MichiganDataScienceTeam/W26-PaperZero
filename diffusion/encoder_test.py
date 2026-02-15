import torch
from diffusion.encoder2 import AutoEncoder, VAE
from data.origami_sampler import OrigamiSampler

def generate_dataset(img_size, folds, num):
    sampler = OrigamiSampler((64,64))
    papers = []
    for i in range(num):
        vec = sampler.sample(folds)
        papers.append(vec["target_mask"])
    


def test_autoencoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoEncoder(
        in_channels=2,
        img_size=128,
        latent_dim=128,
        base_ch=32
    ).to(device)

    model.eval()

    x = torch.randn(4, 2, 128, 128, device=device)

    with torch.no_grad():
        x_hat = model(x)

    print("=== AutoEncoder Test ===")
    print("Input shape: ", x.shape)
    print("Output shape:", x_hat.shape)
    print()


def test_vae():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VAE(
        in_channels=2,
        img_size=128,
        latent_dim=128,
        base_ch=32
    ).to(device)

    model.eval()

    x = torch.randn(4, 2, 128, 128, device=device)

    with torch.no_grad():
        x_hat, mu, log_var = model(x)

    print("=== VAE Test ===")
    print("Recon shape :", x_hat.shape)
    print("Mu shape    :", mu.shape)
    print("LogVar shape:", log_var.shape)
    print()

def train_vae():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(
        in_channels=2,
        img_size=64,
        latent_dim=64,
        base_ch=32
    ).to(device)


if __name__ == "__main__":
    test_autoencoder()
    test_vae()
