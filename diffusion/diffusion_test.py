from diffusion.encoder import ResidualBlock

from data.origami_sampler import OrigamiSampler
import matplotlib.pyplot as plt
import numpy as np
import torch

sampler = OrigamiSampler((64,64))

target = sampler.sample(5)

model = ResidualBlock(1,3)
#target["target_mask"].reshape((1,64,64))
output = model(torch.ones((1,1,64,64)))
print(output)

plt.imshow(target["target_mask"],cmap="gray")
plt.title("After folding")
plt.show()