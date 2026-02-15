import matplotlib.pyplot as plt
from paper import Paper, Segment, Vec2

p = Paper()
plt.imshow(p.rasterize(128,128),cmap="gray")
plt.title("Before folding")
plt.show()

s = Segment(Vec2(0,0),Vec2(1,1,))
p.fold(s)

plt.imshow(p.rasterize(128,128),cmap="gray")
plt.title("After folding")
plt.show()