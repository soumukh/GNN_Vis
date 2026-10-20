"""Shared behaviour of all node-classification GNNs in the project."""

import torch


class GNNBase(torch.nn.Module):
    """Two-layer node classifier with an optional link-prediction head.

    Subclasses build ``conv1``/``conv2`` and implement ``forward``, which must
    accept ``(x, edge_index, edge_weight=None, return_embeddings=False, **kwargs)``
    and return log-probabilities, plus the hidden embeddings when
    ``return_embeddings`` is True.

    ``init_args`` holds exactly the constructor arguments (besides the input /
    output sizes) needed to rebuild the model from a saved package.
    """

    model_type = None       # set by @register_model
    capabilities = {}       # set by @register_model

    def __init__(self, dropout_rate, enable_link_prediction, init_args):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.enable_link_prediction = enable_link_prediction
        self.init_args = dict(init_args)

    def _add_link_head(self, embedding_dim):
        """Subclasses call this LAST in __init__: creating it after conv1/conv2
        keeps the random-initialisation order of the original models, so
        training with a fixed seed reproduces earlier runs."""
        if self.enable_link_prediction:
            self.link_predictor = torch.nn.Sequential(
                torch.nn.Linear(embedding_dim * 2, embedding_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout_rate),
                torch.nn.Linear(embedding_dim, 1),
            )

    def predict_links(self, embeddings, edge_index):
        """Raw link scores (logits) for the node pairs in ``edge_index`` [2, E].
        Always returns a 1-D tensor of length E."""
        if not self.enable_link_prediction:
            raise ValueError("Link prediction not enabled for this model")
        edge_features = torch.cat([embeddings[edge_index[0]], embeddings[edge_index[1]]], dim=1)
        return self.link_predictor(edge_features).squeeze(-1)

    @torch.no_grad()
    def inference(self, data):
        """Full-graph pass in eval mode: (log_probs, hidden embeddings)."""
        self.eval()
        return self(data.x, data.edge_index, return_embeddings=True)
