"""
Inference helper used by the app, plus the explanation functions that now live
in gnnvis/explainers (re-exported here for scripts that import them from app_expl).
"""

import torch

from gnnvis.explainers.cpa import run_cpa_iv  # noqa: F401
from gnnvis.explainers.structural import explain_link_prediction  # noqa: F401

# Device local to this module
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def run_inference(model, data):
    model.eval()
    out = model(data.x.to(device), data.edge_index.to(device))
    try:
        probs = torch.softmax(out, dim=-1)
    except Exception:
        probs = out
    probs_np = probs.cpu().numpy()
    preds = probs_np.argmax(axis=-1)
    confidences = probs_np.max(axis=-1)
    return {'preds': preds.tolist(), 'confidences': confidences.tolist()}
