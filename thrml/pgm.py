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

    The identifier is a process-global counter, so it is unique only among the
    instances a single process created, and only for as long as that process
    lives. See the warning on [`thrml.AbstractNode`][].
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

    **Node identity is process-local.** A node's identity — what `==`, `hash` and `<` use, and therefore what
    every [`thrml.Block`][], [`thrml.BlockSpec`][] and node-keyed dict looks a node up by — is a counter handed
    out in creation order within one process. It is not derived from anything about the node itself, so nodes
    are only comparable against other nodes the same process created.

    In particular, a node that is serialized (with `pickle`, say) keeps its counter value, and reloading it in
    another process will make it compare equal to whichever unrelated node that process happened to create at
    the same point in its own creation order. Reload a graph in one piece — the nodes together with the blocks
    and edges that refer to them — and do not mix reloaded nodes with freshly built ones.
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
    represented by an integer in [0, K)."""

    pass


DEFAULT_NODE_SHAPE_DTYPES = {
    SpinNode: jax.ShapeDtypeStruct(tuple(), dtype=jnp.bool_),
    CategoricalNode: jax.ShapeDtypeStruct(tuple(), dtype=jnp.uint8),
}
