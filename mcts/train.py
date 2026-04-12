"""
Efficient Training
"""
import random
import torch
from torch.utils.data import Dataset, DataLoader
from paper import Paper
import numpy as np
from typing import List
import numpy.typing as npt
from collections import deque
from data.origami_sampler import OrigamiSampler
from mcts.node import Node, RASTER_RESOLUTION
from mcts.encoder import ThinkArchitecture

model = ThinkArchitecture(2, 128, 3)

def get_model():
    return model


class FoldDataset(Dataset):
    """
    Wraps the replay buffer.

    Each record comes from Node.export_training_record() and contains:
        state       : np.ndarray  [2, H, W]   (current paper + target)
        priors      : np.ndarray  [num_actions]  visit-count policy targets
        action_keys : list of (p1x,p1y,p2x,p2y) tuples
        value       : float       Q-value target

    We need to turn the sparse per-action priors into a dense [2, H, W]
    policy map that matches what PolicyHead outputs, then supervise with MSE
    (or KL) against the MCTS visit distribution.
    """

    H = W = RASTER_RESOLUTION

    def __init__(self, records: list[dict]):
        self.samples = []
        for rec in records:
            state        = torch.tensor(rec["state"],        dtype=torch.float32)   # [2,H,W]
            value_target = torch.tensor([rec["value"]],      dtype=torch.float32)   # [1]

            # Build dense policy targets [2, H, W] from sparse action priors
            policy_target = torch.tensor(rec["policy_target"], dtype=torch.float32)

            self.samples.append((state, policy_target, value_target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _calculate_iou(mask1: npt.NDArray[np.bool_], mask2: npt.NDArray[np.bool_]) -> float:
        """
        Computes the IoU (intersection over union) of two 2D binary masks
        of equal shape. Note this is also commutative because union and
        intersection are both commutative.

        Args:
            mask1: one of the masks
            mask2: the other mask
        """
        intersection = (mask1 & mask2).sum()
        union = (mask1 | mask2).sum()
        return float(intersection / union) if union > 0 else 0


def tree_algo(root: Node, num_simulations: int = 15) -> Node:
    """Select, expand, and backprop."""
    for _ in range(num_simulations):
        node: Node = root
        next_node: Node = node
        depth = 0
        iou = 0

        # run fixed number of times
        while depth < 15 and iou < 0.9: 
            # expand (evaluate + backdrop are within this call)
            if not node.children:
                node.expand()
                break
    
            # select
            next = node.select()
            if next is None:
                break
            node = next

            # update depth
            depth += 1

            # iou calculation
            iou = _calculate_iou(node.target_mask.astype(bool), node.paper.rasterize(128, 128, 0.0).astype(bool))

        node = next_node.select()
    
    return root


def train(data_loader: torch.utils.data.DataLoader, model: ThinkArchitecture):
    """
    Update weights.
    Train epoch = 1

    data_loader: DataLoader providing batches of input data and corresponding labels.
    """ 
    # move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # use Q to propigate for value
    # TODO Need to get which grid square the prior corresponds to for policy head
    model.train()
    policy_criterion = torch.nn.MSELoss()
    value_criterion  = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    total_loss = 0.0
    for state, policy_target, value_target in data_loader:
        state = state.to(device)
        policy_target = policy_target.to(device)
        value_target = value_target.to(device)

        optimizer.zero_grad()

        policy_pred, value_pred = model(state)          # [B,2,H,W],  [B,1]

        policy_loss = policy_criterion(policy_pred, policy_target)
        value_loss = value_criterion(value_pred, value_target)
        loss = policy_loss + value_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"  avg loss: {total_loss / len(data_loader):.4f}")
    return model

def evaluate(testing_nodes: List[Node], num_simulations: int = 50) -> float:
    """
    Run MCTS on each test root, greedily walk to the best leaf by Q,
    and return average IoU across all test cases.
    """
    ious = []
    for root in testing_nodes:
        searched = tree_algo(root, num_simulations=num_simulations)

        # Greedily always take highest-Q child
        node = searched
        while node.children:
            node = node.select(c=0)

        final_mask = (
            np.array(node.paper.rasterize(RASTER_RESOLUTION, RASTER_RESOLUTION, 0.0),
                     dtype=np.float32)
            .reshape(RASTER_RESOLUTION, RASTER_RESOLUTION)
            .astype(bool)
        )
        ious.append(_calculate_iou(node.target_mask.astype(bool), final_mask))

    avg = float(np.mean(ious))
    print(f"  avg IoU over {len(testing_nodes)} test cases: {avg:.4f}")
    return avg

def main():
    # Generate Roots/Target Masks
    sampler: OrigamiSampler = OrigamiSampler()
    training_nodes: List[Node] = []
    testing_nodes: List[Node] = []

    total_datasize = 100
    for _ in range(int(0.7 * total_datasize)):
        paper = sampler.sample(random.randint(1, 3))["final_paper"]
        target_mask = paper.rasterize(128, 128).astype(bool)
        training_nodes.append(Node(Paper(), model, target_mask))
    for _ in range(int(0.3 * total_datasize)):
        paper = sampler.sample(random.randint(1, 3))["final_paper"]
        target_mask = paper.rasterize(128, 128).astype(bool)
        testing_nodes.append(Node(Paper(), model, target_mask))

    buffer = deque() 
    buffer_size = 15
    max_epoch = 15
    epoch = 0

    while epoch < max_epoch:
        epoch += 1

        buffer.clear()

        # Populate buffer
        train_set = random.sample(training_nodes, k=buffer_size) 
        for train_node in train_set:
            buffer.append(tree_algo(train_node).export_training_record())

        # Create a DataLoader
        batch = FoldDataset(list(buffer))
        dataloader = DataLoader(batch, batch_size=buffer_size, shuffle=True)

        # Train & Update Weights
        train(dataloader, model)
        
    
    # Evaluate Our Model
    evaluate(testing_nodes)


if __name__ == "__main__":
    main()