from abc import ABC

from bank_projections.utils.base_registry import BaseRegistry


class Book(ABC):
    pass


class OldBook(Book):
    pass


class NewBook(Book):
    pass


class BookRegistry(BaseRegistry[Book]):
    pass


BookRegistry.register("old", OldBook())
BookRegistry.register("new", NewBook())
