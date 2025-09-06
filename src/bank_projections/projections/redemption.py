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

    @classmethod
    def required_columns_validation(cls) -> pl.Expr:
        """Returns an expression that validates all required columns are available and have valid values."""
        pass


class RedemptionRegistry(Redemption):
    _registry: dict[str, type[Redemption]] = {}

    @classmethod
    def register(cls, name: str, redemption: type[Redemption]) -> None:
        if not issubclass(redemption, Redemption):
            raise ValueError(f"Class {redemption} must be a subclass of Redemption")
        cls._registry[name] = redemption

    @classmethod
    def get(cls, item: str) -> type[Redemption]:
        return cls._registry[item]

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

    @classmethod
    def required_columns_validation(cls) -> pl.Expr:
        """Returns an expression that validates all required columns for all registered redemption types."""
        # Base requirement: RedemptionType column must exist and be non-null
        base_validation = pl.col("RedemptionType").is_not_null()

        specific_expr = pl.lit(True)
        for name, redemption_cls in cls._registry.items():
            specific_expr = (
                pl.when(pl.col("RedemptionType") == name)
                .then(redemption_cls.required_columns_validation())
                .otherwise(specific_expr)
            )

        return base_validation & specific_expr


class BulletRedemption(Redemption):
    @classmethod
    def redemption_factor(
        cls, maturity_date: pl.Expr, interest_rate: pl.Expr, coupon_date: pl.Expr, projection_date: datetime.date
    ) -> pl.Expr:
        return pl.when(maturity_date <= pl.lit(projection_date)).then(pl.lit(1.0)).otherwise(pl.lit(0.0))

    @classmethod
    def required_columns_validation(cls) -> pl.Expr:
        """Validates that bullet redemption only needs maturity date."""
        return pl.col("MaturityDate").is_not_null()


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

    @classmethod
    def required_columns_validation(cls) -> pl.Expr:
        """Validates that CouponFrequency column exists and has valid values for annuity redemption."""
        # Check if CouponFrequency column exists and has valid values
        return (
            pl.col("MaturityDate").is_not_null()
            & pl.col("CouponFrequency").is_not_null()
            & pl.col("CouponFrequency").is_in(["Monthly", "Quarterly", "SemiAnnual", "Annual", "Daily", "Weekly"])
            & pl.col("NextCouponDate").is_not_null()
            & pl.col("InterestRate").is_not_null()
        )


class PerpetualRedemption(Redemption):
    @classmethod
    def redemption_factor(
        cls, maturity_date: pl.Expr, interest_rate: pl.Expr, coupon_date: pl.Expr, projection_date: datetime.date
    ) -> pl.Expr:
        return pl.lit(0.0)

    @classmethod
    def required_columns_validation(cls) -> pl.Expr:
        """Validates that perpetual redemption needs no additional columns."""
        return pl.col("MaturityDate").is_null()


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

    @classmethod
    def required_columns_validation(cls) -> pl.Expr:
        """Validates that CouponFrequency column exists and has valid values for linear redemption."""
        return pl.col("MaturityDate").is_not_null()


# Register all redemption types
RedemptionRegistry.register("bullet", BulletRedemption)
RedemptionRegistry.register("annuity", AnnuityRedemption)
RedemptionRegistry.register("perpetual", PerpetualRedemption)
RedemptionRegistry.register("linear", LinearRedemption)
