"""
Efficient Training
"""
import torch
from torch.utils.data import Dataset, DataLoader
from paper import Paper, Segment
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import numpy.typing as npt
from collections import deque
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
            policy_target = self._build_policy_map(rec["action_keys"], rec["priors"])

            self.samples.append((state, policy_target, value_target))

    def _build_policy_map(self, action_keys, priors) -> torch.Tensor:
        """
        action_keys : list of (p1x, p1y, p2x, p2y)  -- normalized [0,1] coords
        priors      : np.ndarray of same length
        Returns     : [2, H, W] float32 tensor
                        channel 0 = start-point heatmap
                        channel 1 = end-point   heatmap
        """
        start_map = np.zeros((self.H, self.W), dtype=np.float32)
        end_map   = np.zeros((self.H, self.W), dtype=np.float32)

        for (p1x, p1y, p2x, p2y), prob in zip(action_keys, priors):
            # action_keys store raw coords — clamp to grid
            x1 = min(max(int(round(p1x * (self.W - 1))), 0), self.W - 1)
            y1 = min(max(int(round(p1y * (self.H - 1))), 0), self.H - 1)
            x2 = min(max(int(round(p2x * (self.W - 1))), 0), self.W - 1)
            y2 = min(max(int(round(p2y * (self.H - 1))), 0), self.H - 1)

            start_map[y1, x1] += prob
            end_map  [y2, x2] += prob

        return torch.tensor(np.stack([start_map, end_map]), dtype=torch.float32)  # [2,H,W]

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


def tree_algo(root: Node, num_simulations: int = 50):
    """Select, expand, and backprop."""
    node = root
    for _ in range(num_simulations):
        next_node = node
        depth = 0
        iou = 0

        # run fixed number of times
        while depth < 15 and iou >= 0.9: 
            # expand (evaluate + backdrop are within this call)
            node.expand()
    
            # select
            next = node.select()
            if next is None:
                break
            node = next

            # update depth
            depth += 1

            # iou calculation
            iou = _calculate_iou(node.target_mask.astype(bool), node.parent.rasterize(128, 128).astype(bool))

        node = next_node.select()
    
    return root


def train(data_loader: torch.utils.data.DataLoader, model: ThinkArchitecture):
    """
    Update weights.

    data_loader: DataLoader providing batches of input data and corresponding labels.

    Description:
        This function sets the model to training mode and use the data loader to iterate through the entire dataset.
        For each batch, it performs the following steps:
        1. Resets the gradient calculations in the optimizer.
        2. Performs a forward pass to get the model predictions.
        3. Computes the loss between predictions and true labels using the specified `criterion`.
        4. Performs a backward pass to calculate gradients.
        5. Updates the model weights using the `optimizer`.
    """
    # use Q to propigate for value
    # TODO Need to get which grid square the prior corresponds to for policy head 

    model.train()
    policy_criterion = torch.nn.MSELoss()
    value_criterion  = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    total_loss = 0.0
    for state, policy_target, value_target in data_loader:
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

def main():
    # Generate Roots/Target Masks
    training_nodes: List[Node] = []
    testing_nodes: List[Node] = []

    buffer = deque() 
    buffer_size = 15
    max_epoch = 15
    epoch = 0

    while epoch < max_epoch:
        epoch += 1

        buffer.clear()

        # Populate buffer
        for i in range(buffer_size):
            buffer.append(tree_algo(training_nodes[i]).export_training_record())

        # Create a DataLoader
        batch = FoldDataset(list(buffer))
        dataloader = DataLoader(batch, batch_size=buffer_size, shuffle=True)

        # Train & Update Weights
        train(dataloader, model)
        
    
    # Evaluate Our Model
    # Run our tree algo and get the final leaf node for each testing_node
    # Calculate averge IOU? and print it out


if __name__ == "__main__":
    main()