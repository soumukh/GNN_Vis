import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from .base import GNNBase
from .registry import register_model


@register_model("GCN", stability=True)
class GCNNet(GNNBase):
    """Two-layer GCN. Embeddings are the hidden layer after ReLU."""

    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate=0.5,
                 enable_link_prediction=False, **legacy_kwargs):
        # **legacy_kwargs: tolerate extra keys in old packages' init args
        super().__init__(
            dropout_rate, enable_link_prediction,
            init_args={'hidden_channels': hidden_channels, 'dropout_rate': dropout_rate,
                       'enable_link_prediction': enable_link_prediction},
        )
        # No cached=True: models are trained on the link-prediction training
        # graph but evaluated on others; a cached GCNConv would silently reuse
        # the first graph's normalised adjacency.
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self._add_link_head(hidden_channels)

    def forward(self, x, edge_index, edge_weight=None, return_embeddings=False, **kwargs):
        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        embeddings = x
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        log_probs = F.log_softmax(x, dim=1)
        return (log_probs, embeddings) if return_embeddings else log_probs
