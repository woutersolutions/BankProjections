from abc import abstractmethod
from typing import Protocol, TypeVar

T = TypeVar("T", bound="Combinable")


class Combinable(Protocol):
    @abstractmethod
    def combine(self: T, other: T) -> T: ...

    @classmethod
    def combine_list(cls: type[T], items: list[T]) -> T:
        result = items[0]
        for item in items[1:]:
            result = result.combine(item)
        return result
