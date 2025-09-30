# Placeholder class, which can be implemented later
from bank_projections.utils.combine import Combinable


class Settings(Combinable):
    def combine(self, other: "Settings") -> "Settings":
        return Settings()
