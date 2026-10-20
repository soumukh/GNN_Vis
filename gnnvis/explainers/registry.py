"""Explainer registry.

An explainer is a function ``fn(model, data, target, **options)``:

* ``target`` is a node index for node classification and a ``(source, target)``
  pair for link prediction;
* it returns a JSON-serialisable result that the UI stores and draws.

Each explainer is registered for one task and one kind:

* ``'attribution'``: which neighbours / features matter (importance plots,
  subgraph highlighting);
* ``'causal_paths'``: which paths through the graph cause the prediction
  (the CPA-IV card).

The callbacks ask for ``get_explainer(task, kind)``. A new explainer becomes
the one the app uses when it is registered with ``default=True``.
"""

from dataclasses import dataclass
from typing import Callable

ATTRIBUTION = 'attribution'
CAUSAL_PATHS = 'causal_paths'


@dataclass(frozen=True)
class ExplainerSpec:
    name: str
    task: str
    kind: str
    fn: Callable
    description: str = ''

    def __call__(self, model, data, target, **options):
        return self.fn(model, data, target, **options)


_REGISTRY = {}   # (task, name) -> ExplainerSpec
_DEFAULTS = {}   # (task, kind) -> name


def register_explainer(name, task, kind, description='', default=False):
    """Decorator: register ``fn`` as explainer ``name`` for ``task``. The first
    explainer registered for a (task, kind) is its default unless a later one
    passes ``default=True``."""
    if kind not in (ATTRIBUTION, CAUSAL_PATHS):
        raise ValueError(f"Unknown explainer kind {kind!r}")

    def decorator(fn):
        if (task, name) in _REGISTRY:
            raise ValueError(f"Explainer {name!r} is already registered for {task!r}")
        _REGISTRY[(task, name)] = ExplainerSpec(name, task, kind, fn, description)
        if default or (task, kind) not in _DEFAULTS:
            _DEFAULTS[(task, kind)] = name
        return fn
    return decorator


def get_explainer(task, kind, name=None):
    """The explainer ``name`` (or the default one) of ``kind`` for ``task``."""
    if name is None:
        if (task, kind) not in _DEFAULTS:
            raise KeyError(f"No {kind} explainer registered for {task!r}")
        name = _DEFAULTS[(task, kind)]
    spec = _REGISTRY.get((task, name))
    if spec is None or spec.kind != kind:
        raise KeyError(f"No {kind} explainer {name!r} for {task!r}")
    return spec


def available_explainers(task=None, kind=None):
    return [s for s in _REGISTRY.values()
            if (task is None or s.task == task) and (kind is None or s.kind == kind)]
