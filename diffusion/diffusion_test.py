from diffusion.encoder2 import VAE

from data.origami_sampler import OrigamiSampler
import matplotlib.pyplot as plt
import numpy as np
import torch

# sampler = OrigamiSampler()

# target = sampler.sample(5)

# model = ResidualBlock(2,2)
# #target["target_mask"].reshape((1,64,64))
# output = model(torch.ones((1,2,64,64)))
# print(output)
# print(output.shape)

# plt.imshow(target["target_mask"],cmap="gray")
# plt.title("After folding")
# plt.show()

model = VAE(2,64,128,)