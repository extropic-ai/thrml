import abc
from dataclasses import dataclass, is_dataclass
from typing import ClassVar

import jax
from jax import numpy as jnp


class _CounterMeta(abc.ABCMeta):
    """Metaclass that automatically calls __post_init__ and provides unique ordering.

    Used internally by THRML for node identification and ordering.
    """

    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        if not is_dataclass(cls):
            post_init = getattr(instance, "__post_init__", None)
            if callable(post_init):
                post_init()
        return instance

    def __lt__(cls, other):
        # todo: make sure this is sufficient to distinguish and be unique for JAX
        if not isinstance(other, type):
            raise NotImplementedError
        return (cls.__module__, cls.__qualname__) < (other.__module__, other.__qualname__)


class _UniqueID(metaclass=_CounterMeta):
    """
    This is a way of ensuring that there is a unique identifier
    for subclasses, without them being required to call super().__init__().
    """

    __slots__ = ("_hash",)
    _counter: ClassVar[int] = 0
    _hash: int

    def __post_init__(self):
        self._hash = _UniqueID._counter
        _UniqueID._counter += 1

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _UniqueID):
            return False
        return self._hash == other._hash

    def __hash__(self) -> int:
        return self._hash

    def __lt__(self, other):
        if isinstance(other, _UniqueID):
            return self._hash < other._hash
        raise RuntimeError("less than only defined between _UniqueIDs")


@dataclass(eq=False)
class AbstractNode(_UniqueID):
    """
    A node in a PGM.

    Every node used in a PGM must inherit from this class. When compiling a program, each node is assigned a
    shape and datatype that are used to organize the state of the sampling program in a jax-friendly way.
    """

    def __new__(cls, *args, **kwargs):
        if cls is AbstractNode:
            raise TypeError(f"only children of '{cls.__name__}' may be instantiated")
        return object.__new__(cls)


class SpinNode(AbstractNode):
    """A node that represents a random variable that takes on a state in {-1, 1}."""

    pass


class CategoricalNode(AbstractNode):
    """A node that represents a random variable that may take on any one of K possible discrete states,
    represented by a positive integer in (0, K]."""

    pass


DEFAULT_NODE_SHAPE_DTYPES = {
    SpinNode: jax.ShapeDtypeStruct(tuple(), dtype=jnp.bool_),
    CategoricalNode: jax.ShapeDtypeStruct(tuple(), dtype=jnp.uint8),
}
