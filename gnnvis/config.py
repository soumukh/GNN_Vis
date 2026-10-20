"""Project-wide settings shared by the app, its layout and the scripts."""

from gnnvis.models import available_models

AVAILABLE_DATASETS = ['Cora', 'CiteSeer']
AVAILABLE_MODELS = available_models()   # every model registered in gnnvis/models

DEFAULT_DATASET = 'Cora'
DEFAULT_MODEL = 'GCN'
