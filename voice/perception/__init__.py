"""Perception stack: observers, localization utilities, SAM helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict

__all__ = ["VLMObserver", "ObservationContext", "TargetLocalizer", "DepthSnapshot"]

_LAZY_IMPORTS: Dict[str, str] = {
    "VLMObserver": "voice.perception.observer",
    "ObservationContext": "voice.perception.observer",
    "TargetLocalizer": "voice.perception.localize_target",
    "DepthSnapshot": "voice.perception.localize_target",
}


def __getattr__(name: str) -> Any:
    """
    Lazily import perception components to avoid circular imports and heavy
    dependencies during package initialisation.
    """
    module_path = _LAZY_IMPORTS.get(name)
    if not module_path:
        raise AttributeError(f"module 'voice.perception' has no attribute '{name}'")
    module = import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> Any:
    return sorted(list(__all__) + [key for key in globals() if not key.startswith("_")])
