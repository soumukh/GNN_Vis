"""Build models and load saved model packages.

A package is the pickle written by ``train_models.py``
(``models/{model_type}_{dataset}_tuned.pkl``): a dict with at least
``model_state_dict`` and ``model_init_args``.
"""

import os
import pickle

import torch

from .registry import get_model_class

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(PROJECT_DIR, 'models')


def build_model(model_type, num_features, num_classes, **init_args):
    """A fresh, untrained model of a registered type."""
    return get_model_class(model_type)(in_channels=num_features, out_channels=num_classes, **init_args)


def package_path(model_type, dataset_name, model_dir=None):
    return os.path.join(model_dir or MODEL_DIR, f"{model_type}_{dataset_name}_tuned.pkl")


def load_package(model_type, dataset_name, model_dir=None):
    """The saved package dict, or None if the file is missing or unreadable."""
    path = package_path(model_type, dataset_name, model_dir)
    if not os.path.exists(path):
        print(f"ERROR: Pre-computed file not found: {path}")
        return None
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load pre-computed file '{path}'. Error: {e}")
        return None


def load_model(package, model_type, num_features, num_classes, device=None, strict=True):
    """Rebuild the model saved in ``package``, in eval mode.

    Loading is strict: a package whose weights do not fit the registered
    architecture raises instead of silently dropping layers (the old
    ``strict=False`` loads hid mismatched model copies).
    """
    model = build_model(model_type, num_features, num_classes, **package.get('model_init_args', {}))
    state_dict = package.get('model_state_dict')
    if not state_dict:
        raise ValueError("Package has no model_state_dict")
    state_dict = {k: torch.tensor(v) if isinstance(v, list) else v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=strict)
    if device is not None:
        model.to(device)
    return model.eval()
