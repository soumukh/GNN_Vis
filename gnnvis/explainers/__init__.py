"""Explainers, registered per task and kind (see registry.py).

Importing this package registers the built-in ones:

    node classification  attribution   GNNExplainer
                         causal_paths  CPA-IV
    link prediction      attribution   Structural
                         causal_paths  CPA-IV
"""

from gnnvis.explainers.registry import (
    ATTRIBUTION, CAUSAL_PATHS, ExplainerSpec, available_explainers, get_explainer, register_explainer,
)
from gnnvis.explainers import gnnexplainer, structural, cpa  # noqa: F401  (register the built-ins)

__all__ = ['ATTRIBUTION', 'CAUSAL_PATHS', 'ExplainerSpec', 'available_explainers', 'get_explainer',
           'register_explainer']
