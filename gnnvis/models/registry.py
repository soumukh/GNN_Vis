"""Model registry.

Every model class registers itself under a name ("GCN", "GAT", ...) together
with its capabilities. Training, the precompute scripts and the app all look
models up here, so adding a model means adding one module that uses
``@register_model`` (and importing it in ``gnnvis/models/__init__.py``).

Capabilities are what the UI checks instead of model names:

    attention   the model exposes attention weights (GAT attention panel)
    stability   explanation-stability analysis is supported for this model
"""

MODEL_REGISTRY = {}

DEFAULT_CAPABILITIES = {'attention': False, 'stability': False}


def register_model(name, **capabilities):
    """Class decorator: register a model class under ``name``."""
    unknown = set(capabilities) - set(DEFAULT_CAPABILITIES)
    if unknown:
        raise ValueError(f"Unknown capabilities for {name}: {sorted(unknown)}")

    def decorator(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered")
        cls.model_type = name
        cls.capabilities = {**DEFAULT_CAPABILITIES, **capabilities}
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model_class(name):
    try:
        return MODEL_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown model type '{name}'. Registered: {available_models()}") from None


def available_models():
    return list(MODEL_REGISTRY)


def model_capabilities(name):
    """Capabilities of a registered model type (all False for unknown names)."""
    cls = MODEL_REGISTRY.get(name)
    return dict(cls.capabilities) if cls else dict(DEFAULT_CAPABILITIES)
