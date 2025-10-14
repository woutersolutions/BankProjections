import polars as pl

from bank_projections.utils.base_registry import BaseRegistry


class HQLAClass:
    def __init__(self, haircut: float):
        self.haircut = haircut
        self.contribution = 1.0 - haircut


class HQLARegistry(BaseRegistry[HQLAClass]):
    @classmethod
    def hqla_constribution_expression(cls) -> pl.Expr:
        expr = pl.lit(0.0)  # Default, should not be used
        for name, impl in cls.stripped_items.items():
            expr = pl.when(pl.col("HQLAClass") == name).then(impl.contribution).otherwise(expr)
        return expr


HQLARegistry.register("Level 1", HQLAClass(haircut=0.0))
HQLARegistry.register("Level 2A", HQLAClass(haircut=0.15))
HQLARegistry.register("Level 2B corporate", HQLAClass(haircut=0.25))
HQLARegistry.register("Level 2B equity", HQLAClass(haircut=0.50))
HQLARegistry.register("Non-HQLA", HQLAClass(haircut=1.0))
HQLARegistry.register("N/a", HQLAClass(haircut=1.0))
