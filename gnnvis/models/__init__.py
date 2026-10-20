"""GNN models: the single definition used by training, precompute scripts and the app.

To add a model: create a module next to gcn.py / gat.py with a ``GNNBase``
subclass decorated with ``@register_model("NAME", ...capabilities)`` and import
it below. Nothing else needs to know the model's name.
"""

from .registry import (MODEL_REGISTRY, register_model, get_model_class,
                       available_models, model_capabilities)
from .base import GNNBase
from .gcn import GCNNet   # noqa: F401  (registers "GCN")
from .gat import GATNet   # noqa: F401  (registers "GAT")
from .io import MODEL_DIR, build_model, package_path, load_package, load_model

__all__ = [
    'MODEL_REGISTRY', 'register_model', 'get_model_class', 'available_models', 'model_capabilities',
    'GNNBase', 'GCNNet', 'GATNet',
    'MODEL_DIR', 'build_model', 'package_path', 'load_package', 'load_model',
]
