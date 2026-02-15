import torch
from encoder import AutoEncoder, VAE

def test_autoencoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoEncoder(
        in_channels=2,
        img_size=128,
        latent_dim=128,
        base_ch=32
    ).to(device)

    x = torch.randn(4, 2, 128, 128, device=device)

    with torch.no_grad():
        x_hat = model(x)

    print("AutoEncoder:")
    print(" input :", x.shape)
    print(" output:", x_hat.shape)


def test_vae():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VAE(
        in_channels=2,
        img_size=128,
        latent_dim=128,
        base_ch=32
    ).to(device)

    x = torch.randn(4, 2, 128, 128, device=device)

    with torch.no_grad():
        x_hat, mu, log_var = model(x)

    print("\nVAE:")
    print(" recon :", x_hat.shape)
    print(" mu    :", mu.shape)
    print(" logvar:", log_var.shape)


if __name__ == "__main__":
    test_autoencoder()
    test_vae()
