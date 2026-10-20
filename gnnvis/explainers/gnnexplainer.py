"""GNNExplainer (PyG) for node classification and for single links.

The app's node explanation and both stability analyses build their PyG
Explainer here, so they explain the model in the same way (they differ only in
the number of optimisation epochs).
"""

import torch
from torch_geometric.explain import Explainer, GNNExplainer

from gnnvis.explainers.registry import ATTRIBUTION, register_explainer
from gnnvis.tasks import NODE_CLASSIFICATION

NODE_MODEL_CONFIG = dict(mode='multiclass_classification', task_level='node', return_type='log_probs')
# raw logits: PyG applies the sigmoid itself in binary_classification
LINK_MODEL_CONFIG = dict(mode='binary_classification', task_level='graph', return_type='raw')

APP_EPOCHS = 100        # interactive explanation of one node
STABILITY_EPOCHS = 20   # two explanations per node / edge, thousands of them


def model_device(model):
    """Device the model's parameters live on (falls back to CPU)."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device('cpu')


def node_mask_explainer(model, epochs=APP_EPOCHS):
    """PyG Explainer learning an edge mask for one node's predicted class.
    Call it as ``explainer(x=..., edge_index=..., target=..., index=node_idx)``."""
    return Explainer(model=model, algorithm=GNNExplainer(epochs=epochs), explanation_type='model',
                     edge_mask_type='object', model_config=NODE_MODEL_CONFIG)


class LinkLogit(torch.nn.Module):
    """Wraps the base GNN so GNNExplainer can explain a single link (src→dst).
    forward(x, edge_index) → raw logit [1,1] for a binary_classification task
    with return_type='raw'. (Returning a sigmoid here would make PyG treat every
    edge as class 1 and apply sigmoid twice in the loss.)"""
    def __init__(self, base_model, src: int, dst: int):
        super().__init__()
        self.base_model = base_model
        self.src = int(src)
        self.dst = int(dst)

    def forward(self, x, edge_index, **kwargs):
        _, emb = self.base_model(x, edge_index, return_embeddings=True)
        ep = torch.tensor([[self.src], [self.dst]], device=x.device, dtype=torch.long)
        if getattr(self.base_model, 'enable_link_prediction', False):
            score = self.base_model.predict_links(emb, ep)
        else:
            score = (emb[self.src] * emb[self.dst]).sum()
        return score.view(1, 1)


def link_mask_explainer(model, source, target, epochs=STABILITY_EPOCHS):
    """PyG Explainer learning an edge mask for the link source→target.
    Call it as ``explainer(x=..., edge_index=...)``."""
    wrapper = LinkLogit(model, source, target)
    wrapper.eval()
    return Explainer(model=wrapper, algorithm=GNNExplainer(epochs=epochs), explanation_type='model',
                     edge_mask_type='object', model_config=LINK_MODEL_CONFIG)


def edge_mask(explanation):
    """The explanation's edge mask as a numpy array, or None."""
    mask = getattr(explanation, 'edge_mask', None)
    if mask is not None:
        return mask.detach().cpu().numpy()
    mask = explanation.get('edge_mask', None)
    if mask is not None:
        return mask.detach().cpu().numpy()
    for key in explanation.keys() if hasattr(explanation, 'keys') else []:
        if 'edge' in key.lower() and 'mask' in key.lower():
            val = explanation[key]
            if val is not None:
                print(f"[STABILITY] Found edge mask under key '{key}'")
                return val.detach().cpu().numpy()
    try:
        avail = list(explanation.keys()) if hasattr(explanation, 'keys') else dir(explanation)
        print(f"[STABILITY] WARNING — edge_mask not found. Available keys: {avail}")
    except Exception:
        pass
    return None


def neighbour_importance(mask, edge_index, node_idx):
    """{neighbour: max |mask| over the edges joining it to node_idx}."""
    mask = mask.cpu()
    edge_index = edge_index.cpu()
    incident = ((edge_index[0] == node_idx) | (edge_index[1] == node_idx)).nonzero().flatten().tolist()
    importance = {}
    for i in incident:
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        neighbour = src if dst == node_idx else dst
        value = abs(mask[i].item())
        importance[neighbour] = max(importance[neighbour], value) if neighbour in importance else value
    return importance


def gradient_x_input(model, x, edge_index, node_idx, cls):
    """{feature: |d score / d x| * x} for node_idx's class ``cls`` (falls back
    to |gradient| when the product is all zero); None if it cannot be computed."""
    try:
        model.eval()
        x_in = x.clone().detach()
        x_in.requires_grad = True
        out = model(x_in, edge_index)
        if cls is None:
            cls = int(torch.argmax(out[node_idx]).item())
        score = out[node_idx, cls]
        model.zero_grad()
        score.backward(retain_graph=False)
        grads = x_in.grad[node_idx].cpu().abs().numpy()
        vals = (grads * x_in.detach().cpu().numpy()[node_idx]).astype(float)
        if not vals.any():
            vals = grads
        return {i: float(vals[i]) for i in range(len(vals))}
    except Exception:
        return None


@register_explainer('GNNExplainer', NODE_CLASSIFICATION, ATTRIBUTION,
                    description='Edge mask learned by GNNExplainer (neighbour importance) '
                                'and gradient x input (feature importance).')
def explain_node(model, data, node_idx, predicted_class=None, epochs=APP_EPOCHS):
    """Explain node_idx's prediction (``predicted_class``; the model's argmax
    if None). Returns the payload of the explanation store."""
    dev = model_device(model)
    x, edge_index = data.x.to(dev), data.edge_index.to(dev)
    if predicted_class is None:
        with torch.no_grad():
            predicted_class = int(model(x, edge_index)[node_idx].argmax())
    explanation = node_mask_explainer(model, epochs)(
        x=x, edge_index=edge_index, target=torch.tensor([predicted_class], device=dev), index=node_idx)
    mask = explanation.get('edge_mask')
    return {
        'edge_mask': mask.cpu().tolist() if mask is not None else None,
        'node_importance': neighbour_importance(mask, edge_index, node_idx) if mask is not None else {},
        'feature_importance': gradient_x_input(model, x, edge_index, node_idx, predicted_class),
        'explanation_type': NODE_CLASSIFICATION,
        'node_idx': node_idx,
    }

