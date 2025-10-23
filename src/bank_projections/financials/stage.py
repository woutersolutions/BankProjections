import polars as pl

from bank_projections.utils.base_registry import BaseRegistry


class IFRS9Stage:
    def __init__(self, is_default: bool):
        self.is_default = is_default


class IFRS9StageRegistry(BaseRegistry[IFRS9Stage]):
    @classmethod
    def is_default_expr(cls) -> pl.Expr:
        expr = pl.lit(False)
        for name, stage_cls in cls.stripped_items.items():
            expr = pl.when(pl.col("IFRS9Stage") == name).then(pl.lit(stage_cls.is_default)).otherwise(expr)
        return expr


IFRS9StageRegistry.register("1", IFRS9Stage(is_default=False))
IFRS9StageRegistry.register("2", IFRS9Stage(is_default=False))
IFRS9StageRegistry.register("3", IFRS9Stage(is_default=True))
IFRS9StageRegistry.register("Poci", IFRS9Stage(is_default=True))
IFRS9StageRegistry.register("N/a", IFRS9Stage(is_default=False))
