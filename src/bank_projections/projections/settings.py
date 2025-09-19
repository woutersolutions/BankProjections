# Placeholder class, which can be implemented later
from bank_projections.utils.combine import Combinable, T


class Settings(Combinable):
    def combine(self, other: "Settings") -> T:
        return Settings()
