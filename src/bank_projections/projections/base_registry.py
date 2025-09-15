from abc import ABC
from typing import TypeVar

from loguru import logger

T = TypeVar("T", bound=ABC)


class BaseRegistry[T](ABC):  # noqa: B024
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.items: dict[str, T] = {}

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


def is_in_identifiers(identifier: str, identifiers: set[str]) -> bool:
    return clean_identifier(identifier) in [clean_identifier(id) for id in identifiers]


def get_identifier(identifier: str, identifiers: set[str]) -> str:
    cleaned = clean_identifier(identifier)
    for id in identifiers:
        if cleaned == clean_identifier(id):
            return id
    raise KeyError(f"{identifier} not found in identifiers")
