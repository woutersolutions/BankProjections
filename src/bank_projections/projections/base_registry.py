from abc import ABC
from typing import Any, ClassVar, TypeVar

from loguru import logger

from bank_projections.utils.parsing import strip_identifier

T = TypeVar("T", bound=ABC)


class BaseRegistry[T](ABC):  # noqa: B024
    items: ClassVar[dict[str, Any]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.items = {}

    @classmethod
    def register(cls, name: str, item: T) -> None:
        stripped_name = strip_identifier(name)
        if stripped_name is None:
            raise ValueError(f"Invalid identifier: {name}")
        if stripped_name in cls.items:
            logger.warning(f"{cls.__name__} '{stripped_name}' was already registered.")
        cls.items[stripped_name] = item

    @classmethod
    def get(cls, name: str) -> T:
        stripped_name = strip_identifier(name)
        if stripped_name is None:
            raise ValueError(f"Invalid identifier: {name}")
        if stripped_name not in cls.items:
            raise ValueError(f"{cls.__name__} '{stripped_name}' is not registered.")
        item: T = cls.items[stripped_name]
        return item

    @classmethod
    def is_registered(cls, name: str) -> bool:
        return strip_identifier(name) in cls.items

    @classmethod
    def names(cls) -> list[str]:
        return list(cls.items.keys())
