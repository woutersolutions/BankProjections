from abc import ABC
from typing import Generic, TypeVar

T = TypeVar("T", bound=ABC)


class BaseRegistry(ABC, Generic[T]):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry: dict[str, T] = {}

    @classmethod
    def register(cls, name: str, item: T) -> None:
        if name in cls._registry:
            raise ValueError(f"{cls.__name__} '{name}' is already registered.")
        cls._registry[name] = item

    @classmethod
    def get(cls, name: str) -> T:
        if name not in cls._registry:
            raise ValueError(f"{cls.__name__} '{name}' is not registered.")
        return cls._registry[name]
