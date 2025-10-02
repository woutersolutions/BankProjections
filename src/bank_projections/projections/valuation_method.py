import datetime
from abc import ABC, abstractmethod

import pandas as pd
import polars as pl

from bank_projections.projections.base_registry import BaseRegistry


class ValuationMethod(ABC):

    @classmethod
    @abstractmethod
    def dirty_price(
        cls, projection_date: datetime.date, zero_rates:pd.DataFrame,
    ) -> pl.Expr:
        pass

class NoValuationMethod(ValuationMethod):

    @classmethod
    def dirty_price(
        cls, projection_date: datetime.date, zero_rates:pd.DataFrame,
    ) -> pl.Expr:
        return pl.lit(None, dtype=pl.Float64)

class BaselineValuationMethod(ValuationMethod):

    @classmethod
    def dirty_price(
        cls, projection_date: datetime.date, zero_rates:pd.DataFrame,
    ) -> pl.Expr:
        return 1.0 + pl.when(pl.col("Quantity") == 0).then(0.0).otherwise(pl.col("AccruedInterest") / pl.col("Quantity"))

class ValuationMethodRegistry(BaseRegistry[ValuationMethod], ValuationMethod):

    @classmethod
    def dirty_price(
        cls, projection_date: datetime.date, zero_rates:pd.DataFrame,
    ) -> pl.Expr:
        expr = pl.lit(None, dtype=pl.Float64)
        for name, impl in cls.items.items():
            expr = (
                pl.when(pl.col("ValuationMethod") == name)
                .then(impl.dirty_price(projection_date, zero_rates))
                .otherwise(expr)
            )
        return expr


ValuationMethodRegistry.register("none", NoValuationMethod())
ValuationMethodRegistry.register("baseline", BaselineValuationMethod())
