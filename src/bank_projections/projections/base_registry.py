from abc import ABC
from typing import Generic, TypeVar, get_args

T = TypeVar('T', bound=ABC)


class BaseRegistry(ABC, Generic[T]):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry: dict[str, type[T]] = {}

    @classmethod
    def register(cls, name: str, item_class: type[T]) -> None:
        base_class = cls._get_base_class()
        if not issubclass(item_class, base_class):
            raise ValueError(f"Class {item_class} must be a subclass of {base_class}")
        cls._registry[name] = item_class

    @classmethod
    def get(cls, name: str) -> type[T]:
        if name not in cls._registry:
            raise ValueError(f"{cls.__name__} '{name}' is not registered.")
        return cls._registry[name]

    @classmethod
    def _get_base_class(cls) -> type[T]:
        """Extract the base class from the Generic type parameter."""
        for base in cls.__orig_bases__:
            if hasattr(base, '__origin__') and base.__origin__ is BaseRegistry:
                args = get_args(base)
                if args:
                    return args[0]
        raise RuntimeError(f"Could not determine base class for {cls.__name__}")  # pragma: no cover
