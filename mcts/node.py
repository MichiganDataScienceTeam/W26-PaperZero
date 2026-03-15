# select, expand, evalualte, backprop
import torch.nn as nn
from paper import Paper, Segment, Vec2
import numpy as np
import matplotlib.pyplot as plt
import torch
import math
from encoder import ThinkArchitecture

network = ThinkArchitecture(1, 128, 3)

class Node:
    def __init__(self, paper: Paper, parent=None):
        super().__init__()

        # tree
        self.parent = parent
        self.children = []

        # data
        self.paper = paper
        self.N = 0      # number of visits
        self.W = 0.0    # sum of all values
        self.Q = self.W / self.N if self.N != 0 else 0.0    # W/N (0 if N = 0)
        self.P = 0.0 if not self.parent else self.parent.P     # prior

        # constants
        self.sumB = None

    # For testing purposes
    def render(self):
        img = self.paper.rasterize(128, 128, 0.0)
        img = np.array(img)
        plt.imshow(img, cmap="gray", origin="lower")
        plt.text(0, -17, f'Visits: {self.N}\nPrior: {self.P}\nValue: {self.W}', fontsize=12)
        plt.show()

    def expand(self):
        """Make Children"""

        policy, value = network(self.paper)
        self.W += value

        # TODO: Generate list of segments

        # Calculate prior
        
        # Select node

        # Generate the child node

        # Back prop
        

    def select(self, c=1):
        # Make sure there's children
        if len(self.children) > 0:
            return None # We're done (leaf node)

        # Update sumB if necessary
        if self.sumB is None:
            for child in self.children:
                self.sumB += child.N

        # Uses PUCT instead
        def puct_score(self, child, c):
            explore = c * self.P * math.sqrt(self.sumB) / (1 + child.N)
            return self.Q + explore

        return max(self.children, key=puct_score)
