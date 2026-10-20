import os
import sys

import pytest
import torch
from torch_geometric.data import Data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture
def toy_graph():
    """8 nodes, 3 classes, undirected edges (both directions) + one isolated node.

    0-1, 1-2, 2-0 (triangle), 2-3, 3-4, 4-5, 5-6; node 7 isolated.
    """
    pairs = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 6)]
    src = [a for a, b in pairs] + [b for a, b in pairs]
    dst = [b for a, b in pairs] + [a for a, b in pairs]
    x = torch.eye(8)
    y = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2])
    return Data(x=x, edge_index=torch.tensor([src, dst]), y=y, num_nodes=8)


@pytest.fixture
def toy_predictions():
    preds = [0, 0, 1, 1, 1, 2, 2, 0]          # nodes 2 and 7 misclassified
    conf = [0.9, 0.8, 0.55, 0.7, 0.6, 0.95, 0.85, 0.4]
    return preds, conf
