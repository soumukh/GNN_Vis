import os
import torch
from torch_geometric.datasets import Planetoid
from gnnvis.models import load_package, load_model
from link_prediction_utils import create_link_prediction_candidates, get_top_k_predictions

MODEL = 'GCN'
DATASET = 'Cora'
TOP_K = 20
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

import json
import traceback

def main():
    print(f"Generating top-{TOP_K} link predictions for {MODEL}/{DATASET}")
    package = load_package(MODEL, DATASET)
    if not package:
        print('No package found; aborting')
        return 1

    # Load directly with Planetoid (the app's dataset cache has no 'data' object,
    # and importing app would build the whole Dash UI)
    dataset = Planetoid(root=DATA_DIR, name=DATASET)
    data = dataset[0]
    model = load_model(package, MODEL, dataset.num_node_features, dataset.num_classes, device='cpu')

    model.eval()
    try:
        with torch.no_grad():
            out, emb = model(data.x, data.edge_index, return_embeddings=True)

        candidates = create_link_prediction_candidates(data, target_node=None, exclude_existing=True)
        print('Candidates size:', candidates.size())

        preds = get_top_k_predictions(emb, candidates, model, k=TOP_K)
        print('Top predictions:')
        for p in preds:
            print(p)

        # Ensure JSON-serializable
        serializable = []
        for p in preds:
            serializable.append({'source': int(p['source']), 'target': int(p['target']), 'score': float(p['score'])})

        with open('link_predictions.json', 'w') as f:
            json.dump(serializable, f)
        print('Saved link_predictions.json')
    except Exception:
        print('Error during prediction:')
        traceback.print_exc()
        return 2

if __name__ == '__main__':
    raise SystemExit(main())