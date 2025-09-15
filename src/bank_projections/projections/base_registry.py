from abc import ABC
from collections.abc import Iterable
from typing import Any, ClassVar, TypeVar

from loguru import logger

T = TypeVar("T", bound=ABC)


class BaseRegistry[T](ABC):  # noqa: B024
    items: ClassVar[dict[str, Any]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.items = {}

    @classmethod
    def register(cls, name: str, item: T) -> None:
        name = clean_identifier(name)
        if name in cls.items:
            logger.warning(f"{cls.__name__} '{name}' was already registered.")
        cls.items[name] = item

    @classmethod
    def get(cls, name: str) -> T:
        name = clean_identifier(name)
        if name not in cls.items:
            raise ValueError(f"{cls.__name__} '{name}' is not registered.")
        return cls.items[name]

    @classmethod
    def is_registered(cls, name: str) -> bool:
        return clean_identifier(name) in cls.items


def clean_identifier(identifier: str) -> str:
    return (
        identifier.strip().lower().replace("_", "").replace(" ", "").replace("-", "").replace("/", "").replace("\\", "")
    )


def is_in_identifiers(identifier: str, identifiers: Iterable[str]) -> bool:
    return clean_identifier(identifier) in [clean_identifier(id) for id in identifiers]


def get_identifier(identifier: str, identifiers: Iterable[str]) -> str:
    cleaned = clean_identifier(identifier)
    for id in identifiers:
        if cleaned == clean_identifier(id):
            return id
    raise KeyError(f"{identifier} not found in identifiers")
