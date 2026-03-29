# select, expand, evalualte, backprop
import torch.nn as nn
from paper import Paper, Segment
import numpy as np
import matplotlib.pyplot as plt
import torch
import math
from mcts.encoder import ThinkArchitecture
from mcts.temp_segment_finder import find_segments

from typing import List

model = ThinkArchitecture(1, 128, 3)

# TODO - figure out how we add a target image [i think it should be given to the ThinkArchitecture]

class Node:
    def __init__(self, paper: Paper, parent=None, segment=None):
        super().__init__()

        # tree
        self.parent: Node = parent
        self.children: List[Node] = []

        # data
        self.paper: Paper = paper
        self.segment: Segment = segment
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
        # Generate list of segments
        segments = find_segments(self.paper)
        
        img = self.paper.rasterize(128, 128, 0.0)
        img = np.array(img).reshape(128, 128)
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        policy, value = model(img_tensor)

        policy = policy.squeeze(0)  # (2, H, W)

        start_logits = policy[0]
        end_logits   = policy[1]

        start_probs = torch.softmax(start_logits.flatten(), dim=0).view_as(start_logits)
        end_probs   = torch.softmax(end_logits.flatten(), dim=0).view_as(end_logits)

        priors = []

        for seg in segments:
            pt1, pt2 = seg.p1, seg.p2

            pr1 = start_probs[int(pt1.x), int(pt1.y)]
            pr2 = end_probs[int(pt2.x), int(pt2.y)]

            priors.append((pr1 * pr2).item())

        # Normalize
        priors = np.array(priors)
        if priors.sum() > 0:
            priors /= priors.sum()
        else:
            priors = np.ones_like(priors) / len(priors)

        # Create children - expand
        for i, P in enumerate(priors):
            copyP = self.paper.copy()
            copyP.fold(segments[i])
            child = Node(paper=copyP, parent=self, segment=segments[i])
            child.P = P
            self.children.append(child)
    
        # Evaluate & Backprop - update W, N, Q for all visited nodes
        node: Node = self
        while node is not None:
            node.W += value.item()
            node.N += 1
            node.Q = node.W / node.N
            node = node.parent


    def select(self, c=1):
        # Make sure there's children
        if len(self.children) == 0:
            return None # We're done (leaf node)

        # Update sumB if necessary
        if self.sumB is None:
            self.sumB = 0
            for child in self.children:
                self.sumB += child.N

        # Uses PUCT instead
        def puct_score(self, child, c):
            explore = c * self.P * math.sqrt(self.sumB) / (1 + child.N)
            return self.Q + explore

        return max(self.children, key=lambda child: puct_score(self, child, c))
    
if __name__ == "__main__":
    paper = Paper()
    root = Node(paper=paper)

    root.expand()
    new_node = root.select()
    new_node.render()   # Check if None

    new_node.expand()
    new_node2 = new_node.select()
    new_node2.render()  # Check if None