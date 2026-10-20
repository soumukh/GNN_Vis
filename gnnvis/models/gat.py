import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from .base import GNNBase
from .registry import register_model


@register_model("GAT", attention=True)
class GATNet(GNNBase):
    """Two-layer GAT. Embeddings are the concatenated heads after ELU."""

    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout_rate=0.6,
                 enable_link_prediction=False, **legacy_kwargs):
        super().__init__(
            dropout_rate, enable_link_prediction,
            init_args={'hidden_channels': hidden_channels, 'heads': heads, 'dropout_rate': dropout_rate,
                       'enable_link_prediction': enable_link_prediction},
        )
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout_rate)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False,
                             dropout=dropout_rate)
        self._add_link_head(hidden_channels * heads)

    def forward(self, x, edge_index, edge_weight=None, return_embeddings=False, **kwargs):
        # edge_weight is accepted for a uniform interface; GATConv does not use it
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        embeddings = x
        x = self.conv2(x, edge_index)
        log_probs = F.log_softmax(x, dim=1)
        return (log_probs, embeddings) if return_embeddings else log_probs

    @torch.no_grad()
    def inference(self, data, return_attention_weights=False):
        """Eval-mode full-graph pass; optionally also the attention weights of
        both layers as ((edge_index, alpha), (edge_index, alpha))."""
        self.eval()
        if not return_attention_weights:
            return super().inference(data)
        x = F.dropout(data.x, p=self.dropout_rate, training=self.training)
        x, att1 = self.conv1(x, data.edge_index, return_attention_weights=True)
        x = F.elu(x)
        embeddings = x
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x, att2 = self.conv2(x, data.edge_index, return_attention_weights=True)
        return F.log_softmax(x, dim=1), embeddings, (att1, att2)
