import datetime
from abc import ABC, abstractmethod

import polars as pl

from bank_projections.projections.frequency import FrequencyRegistry


class Redemption(ABC):
    @classmethod
    @abstractmethod
    def redemption_factor(
        cls, maturity_date: pl.Expr, interest_rate: pl.Expr, coupon_date: pl.Expr, projection_date: datetime.date
    ) -> pl.Expr:
        pass


class RedemptionRegistry(Redemption):
    _registry: dict[str, type[Redemption]] = {}

    @classmethod
    def register(cls, name: str, redemption: type[Redemption]) -> None:
        if not issubclass(redemption, Redemption):
            raise ValueError(f"Class {redemption} must be a subclass of Redemption")
        cls._registry[name] = redemption

    @classmethod
    def redemption_factor(
        cls, maturity_date: pl.Expr, interest_rate: pl.Expr, coupon_date: pl.Expr, projection_date: datetime.date
    ) -> pl.Expr:
        expr = pl.lit(0.0)
        for name, redemption_cls in cls._registry.items():
            expr = (
                pl.when(pl.col("RedemptionType") == name)
                .then(redemption_cls.redemption_factor(maturity_date, interest_rate, coupon_date, projection_date))
                .otherwise(expr)
            )
        return expr


class BulletRedemption(Redemption):
    @classmethod
    def redemption_factor(
        cls, maturity_date: pl.Expr, interest_rate: pl.Expr, coupon_date: pl.Expr, projection_date: datetime.date
    ) -> pl.Expr:
        return pl.when(maturity_date <= pl.lit(projection_date)).then(pl.lit(1.0)).otherwise(pl.lit(0.0))


class AnnuityRedemption(Redemption):
    @classmethod
    def redemption_factor(
        cls, maturity_date: pl.Expr, interest_rate: pl.Expr, coupon_date: pl.Expr, projection_date: datetime.date
    ) -> pl.Expr:
        payments_left = FrequencyRegistry.number_due(coupon_date, pl.lit(projection_date))
        period_rate = interest_rate * FrequencyRegistry.portion_year()

        return (
            pl.when(maturity_date <= pl.lit(projection_date))
            .then(pl.lit(1.0))
            .when(payments_left <= 0)
            .then(pl.lit(0.0))
            .otherwise(period_rate / (1 - (1 + period_rate).pow(-payments_left)))
        )


class PerpetualRedemption(Redemption):
    @classmethod
    def redemption_factor(
        cls, maturity_date: pl.Expr, interest_rate: pl.Expr, coupon_date: pl.Expr, projection_date: datetime.date
    ) -> pl.Expr:
        return pl.lit(0.0)


class LinearRedemption(Redemption):
    @classmethod
    def redemption_factor(
        cls, maturity_date: pl.Expr, interest_rate: pl.Expr, coupon_date: pl.Expr, projection_date: datetime.date
    ) -> pl.Expr:
        payments_left = FrequencyRegistry.number_due(coupon_date, pl.lit(projection_date))

        return (
            pl.when(maturity_date <= pl.lit(projection_date))
            .then(pl.lit(1.0))
            .when(payments_left <= 0)
            .then(pl.lit(0.0))
            .otherwise(pl.lit(1.0) / payments_left)
        )


# Register all redemption types
RedemptionRegistry.register("bullet", BulletRedemption)
RedemptionRegistry.register("annuity", AnnuityRedemption)
RedemptionRegistry.register("perpetual", PerpetualRedemption)
RedemptionRegistry.register("linear", LinearRedemption)
