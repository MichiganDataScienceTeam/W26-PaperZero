import torch
from unet import UNet

# Instantiate the model with default parameters
model = UNet()

# Create dummy input: batch of 1, 1 channel, 128x128 image
x = torch.randn(1, 1, 128, 128)

# Create dummy timestep: batch of 1, random integer from 0 to 999
t = torch.randint(0, 1000, (1,))

# Run forward pass (no gradients needed for testing)
with torch.no_grad():
    out = model(x, t)

# Print shapes and confirm success
print(f"Input shape: {x.shape}")      # Expected: torch.Size([1, 1, 128, 128])
print(f"Timestep shape: {t.shape}")   # Expected: torch.Size([1])
print(f"Output shape: {out.shape}")   # Expected: torch.Size([1, 1, 128, 128])
print("Test passed: UNet forward pass successful")