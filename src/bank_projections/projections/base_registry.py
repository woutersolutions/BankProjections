from abc import ABC
from typing import Generic, TypeVar

T = TypeVar("T", bound=ABC)


class BaseRegistry(ABC, Generic[T]):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.items: dict[str, T] = {}

    @classmethod
    def register(cls, name: str, item: T) -> None:
        if name in cls.items:
            raise ValueError(f"{cls.__name__} '{name}' is already registered.")
        cls.items[name] = item

    @classmethod
    def get(cls, name: str) -> T:
        if name not in cls.items:
            raise ValueError(f"{cls.__name__} '{name}' is not registered.")
        return cls.items[name]
