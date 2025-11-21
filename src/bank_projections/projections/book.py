from abc import ABC

from bank_projections.utils.base_registry import BaseRegistry


class Book(ABC):
    pass


class BackBook(Book):
    pass


class FrontBook(Book):
    pass


class BookRegistry(BaseRegistry[Book]):
    pass


BookRegistry.register("back", BackBook())
BookRegistry.register("front", FrontBook())
