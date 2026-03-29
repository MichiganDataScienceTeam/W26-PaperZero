"""
Efficient Training
"""
import torch.nn as nn
from paper import Paper, Segment
import numpy as np
import matplotlib.pyplot as plt
import torch
import math
from collections import deque
from mcts.encoder import ThinkArchitecture
from mcts.temp_segment_finder import find_segments
from mcts.node import Node


def tree_algo(root: Node, num_simulations: int = 50):
    """Select, expand, and backprop."""
    for _ in range(num_simulations):
        node = root

        # select
        while (len(node.children) > 0):
            next = node.select()
            if next is None:
                break
            node = next

        # expand (evaluate + backdrop are within this call)
        node.expand()
    return root


def train(root: Node, num_simulation: int):
    """Update weights."""



def main():
    batch = []
    buf = deque() 


if __name__ == "__main__":
    main()