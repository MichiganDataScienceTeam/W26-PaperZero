# select, expand, evalualte, backprop
import torch.nn as nn
from paper import Paper, Segment, Vec2
import numpy as np
import matplotlib.pyplot as plt
import torch

class Node:
    def __init__(self, paper: Paper, parent=None):
        super().__init__()

        # tree
        self.parent = parent
        self.children = []

        # data
        self.paper = paper
        self.prior = 0.0
        self.visits = 0
        self.value = 0.0

    # For testing purposes
    def render(self):
        img = self.paper.rasterize(128, 128, 0.0)
        img = np.array(img)
        plt.imshow(img, cmap="gray", origin="lower")
        plt.text(0, -17, f'Visits: {self.visits}\nPrior: {self.prior}\nValue: {self.value}', fontsize=12)
        plt.show()

    def expand(self):
        """Make Children"""
    

    def select(self, c=1):
        # Uses PUCT instead
        def uct_score(self, child, c):
            explore = c * self.prior * np.sqrt(np.self.visits) / (1 + child.visits)
            return self.value + explore
        
        assert(len(self.children) > 0)

        return max(self.children, key=uct_score)
