from data.origami_sampler import OrigamiSampler
import matplotlib.pyplot as plt

sampler = OrigamiSampler((64,64))

vec = sampler.sample(1)

print(vec)

plt.imshow(vec["target_mask"],cmap="gray")
plt.title("After folding")
plt.show()