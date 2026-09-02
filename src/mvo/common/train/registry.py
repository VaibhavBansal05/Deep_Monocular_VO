"""Model registry, so scripts/train.py --model m4 works without imports everywhere."""

from typing import Callable

_REGISTRY: dict[str, Callable] = {}


def register(name: str):
    """Decorator: @register("m4") above your model class."""
    def _wrap(cls):
        if name in _REGISTRY:
            raise KeyError(f"model {name!r} already registered by {_REGISTRY[name]}")
        _REGISTRY[name] = cls
        return cls
    return _wrap


def build(name: str, **kwargs):
    """Instantiate a registered model by name."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown model {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def available() -> list[str]:
    return sorted(_REGISTRY)


def import_all_models() -> None:
    """Import every model package so their @register decorators fire.

    Missing/unfinished model packages are skipped, so one member's broken
    branch never blocks the other three.
    """
    import importlib

    for mod in (
        "mvo.models.m1_classical_mlp",
        "mvo.models.m2_cnn_rnn",
        "mvo.models.m3_selfsup_posedepth",
        "mvo.models.m4_cnn_attention",
    ):
        try:
            importlib.import_module(mod)
        except ImportError:
            pass
