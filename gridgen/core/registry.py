"""
Plugin registry.

Components register themselves under a name and are constructed by that name, so
an experiment can be described entirely by configuration:

    generator = build("generator", "staged-feedergan", device="cuda")

Registration is by decorator, and importing a plugin package is what makes its
entries available.  Nothing in the framework core registers anything — the core
only defines the interfaces and the tables.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, TypeVar

T = TypeVar("T")

#: Kinds of component the framework knows how to look up.
KINDS = ("data_source", "generator", "topology_generator", "attribute_generator",
         "derived_attributes", "constraint", "metric", "category_labeler")

_REGISTRY: dict[str, dict[str, Callable[..., Any]]] = {k: {} for k in KINDS}


def register(kind: str, name: str, *, overwrite: bool = False) -> Callable[[T], T]:
    """Decorator registering a factory (usually a class) under ``kind``/``name``.

    >>> @register("metric", "degree-mmd")
    ... class DegreeMMD(Metric): ...
    """
    if kind not in _REGISTRY:
        raise ValueError(f"unknown kind {kind!r}; expected one of {KINDS}")

    def decorator(factory: T) -> T:
        table = _REGISTRY[kind]
        if name in table and not overwrite:
            raise ValueError(
                f"{kind} {name!r} is already registered by {table[name]!r}; "
                "pass overwrite=True to replace it")
        table[name] = factory  # type: ignore[assignment]
        return factory

    return decorator


def build(kind: str, name: str, /, **kwargs: Any) -> Any:
    """Construct a registered component by name."""
    if kind not in _REGISTRY:
        raise ValueError(f"unknown kind {kind!r}; expected one of {KINDS}")
    table = _REGISTRY[kind]
    if name not in table:
        raise KeyError(
            f"no {kind} registered as {name!r}. Available: {sorted(table)}. "
            "Did you import the plugin package that provides it?")
    return table[name](**kwargs)


def available(kind: str) -> list[str]:
    """Names registered for ``kind``."""
    if kind not in _REGISTRY:
        raise ValueError(f"unknown kind {kind!r}; expected one of {KINDS}")
    return sorted(_REGISTRY[kind])


def catalogue() -> dict[str, list[str]]:
    """Everything currently registered, for printing or logging a run."""
    return {kind: sorted(table) for kind, table in _REGISTRY.items() if table}
