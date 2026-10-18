"""Smoke test: load a model package, check live inference, run a small link-stability sample.

Loads the dataset with Planetoid directly; importing `app` would build the whole Dash UI.
"""
import os

import torch
from torch_geometric.datasets import Planetoid

from run_specific_cpa import instantiate_model, load_precomputed_package
from app_stability import run_link_stability_analysis

MODEL = 'GCN'
DATASET = 'Cora'
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

print(f"Smoke test: model={MODEL}, dataset={DATASET}")
package = load_precomputed_package(MODEL, DATASET)
if not package:
    print('No package found; aborting smoke test')
    exit(1)

dataset = Planetoid(root=DATA_DIR, name=DATASET)
data = dataset[0]

model = instantiate_model(package, MODEL, dataset.num_node_features, dataset.num_classes)
if not model:
    print('Failed to instantiate model; aborting')
    exit(1)

with torch.no_grad():
    preds = model(data.x, data.edge_index).argmax(dim=1)
test_acc = (preds[data.test_mask] == data.y[data.test_mask]).float().mean().item()
print(f'Live test accuracy: {test_acc:.3f}')

print('Running link stability analysis on 5 edges...')
results = run_link_stability_analysis(model, data, sigma=0.05, sample_size=5)
print('Results count:', len(results))
if not results:
    print('No link-stability results; failing')
    exit(1)
print('Sample result:', results[0])
