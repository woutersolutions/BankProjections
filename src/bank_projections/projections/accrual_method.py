import datetime
from abc import ABC, abstractmethod

import polars as pl

from bank_projections.projections.frequency import FrequencyRegistry
from bank_projections.utils.base_registry import BaseRegistry
from bank_projections.utils.daycounting import Actual36525, DaycountFraction


class AccrualMethod(ABC):
    @abstractmethod
    def calculate_accrual(
        self,
        nominal: pl.Expr,
        coupon_rate: pl.Expr,
        previous_coupon_date: pl.Expr,
        next_coupon_date: pl.Expr,
        current_date: datetime.date,
    ) -> pl.Expr:
        pass

    @abstractmethod
    def calculate_current_accrued_interest(
        self,
        nominal: pl.Expr,
        coupon_rate: pl.Expr,
        previous_coupon_date: pl.Expr,
        next_coupon_date: pl.Expr,
        current_date: datetime.date,
    ) -> pl.Expr:
        pass

    @abstractmethod
    def is_accumulating(self) -> bool:
        pass


class RecalculateAccrual(AccrualMethod):
    def __init__(self, day_count: type[DaycountFraction]) -> None:
        self.day_count = day_count

    def calculate_accrual(
        self,
        nominal: pl.Expr,
        coupon_rate: pl.Expr,
        previous_coupon_date: pl.Expr,
        next_coupon_date: pl.Expr,
        current_date: datetime.date,
    ) -> pl.Expr:
        return (
            self.calculate_current_accrued_interest(
                nominal, coupon_rate, previous_coupon_date, next_coupon_date, current_date
            )
            + pl.col("AccruedInterestError")
            - pl.col("AccruedInterest")
        )

    def calculate_current_accrued_interest(
        self,
        nominal: pl.Expr,
        coupon_rate: pl.Expr,
        previous_coupon_date: pl.Expr,
        next_coupon_date: pl.Expr,
        current_date: datetime.date,
    ) -> pl.Expr:
        days_fraction = self.day_count.year_fraction(
            previous_coupon_date, pl.lit(current_date)
        ) / self.day_count.year_fraction(previous_coupon_date, next_coupon_date)

        new_calculated = days_fraction.fill_null(0) * nominal * coupon_rate * FrequencyRegistry.portion_year()
        matured = pl.col("MaturityDate") <= pl.lit(current_date)

        return pl.when(matured).then(0.0).otherwise(new_calculated)

    def is_accumulating(self) -> bool:
        return False


class NoAccrual(AccrualMethod):
    def calculate_current_accrued_interest(
        self,
        nominal: pl.Expr,
        coupon_rate: pl.Expr,
        previous_coupon_date: pl.Expr,
        next_coupon_date: pl.Expr,
        current_date: datetime.date,
    ) -> pl.Expr:
        return pl.lit(0.0)

    def calculate_accrual(
        self,
        nominal: pl.Expr,
        coupon_rate: pl.Expr,
        previous_coupon_date: pl.Expr,
        next_coupon_date: pl.Expr,
        current_date: datetime.date,
    ) -> pl.Expr:
        return pl.lit(0.0)

    def is_accumulating(self) -> bool:
        return False


class DailyAccumulating(AccrualMethod):
    def calculate_current_accrued_interest(
        self,
        nominal: pl.Expr,
        coupon_rate: pl.Expr,
        previous_coupon_date: pl.Expr,
        next_coupon_date: pl.Expr,
        current_date: datetime.date,
    ) -> pl.Expr:
        return pl.lit(0.0)

    def calculate_accrual(
        self,
        nominal: pl.Expr,
        coupon_rate: pl.Expr,
        previous_coupon_date: pl.Expr,
        next_coupon_date: pl.Expr,
        current_date: datetime.date,
    ) -> pl.Expr:
        days_fraction = Actual36525.year_fraction(previous_coupon_date, pl.lit(current_date))
        return days_fraction.fill_null(0) * nominal * coupon_rate

    def is_accumulating(self) -> bool:
        return True


class AccrualMethodRegistry(BaseRegistry[AccrualMethod]):
    @classmethod
    def interest_accrual(
        cls,
        nominal: pl.Expr,
        coupon_rate: pl.Expr,
        previous_coupon_date: pl.Expr,
        next_coupon_date: pl.Expr,
        current_date: datetime.date,
    ) -> pl.Expr:
        expr = pl.lit(None, dtype=pl.Float64)
        for name, accrual_method in cls.stripped_items.items():
            expr = (
                pl.when(pl.col("AccrualMethod") == name)
                .then(
                    accrual_method.calculate_accrual(
                        nominal, coupon_rate, previous_coupon_date, next_coupon_date, current_date
                    )
                )
                .otherwise(expr)
            )

        return expr

    @classmethod
    def current_accrued_interest(
        cls,
        nominal: pl.Expr,
        coupon_rate: pl.Expr,
        previous_coupon_date: pl.Expr,
        next_coupon_date: pl.Expr,
        current_date: datetime.date,
    ) -> pl.Expr:
        expr = pl.lit(None, dtype=pl.Float64)
        for name, accrual_method in cls.stripped_items.items():
            expr = (
                pl.when(pl.col("AccrualMethod") == name)
                .then(
                    accrual_method.calculate_current_accrued_interest(
                        nominal, coupon_rate, previous_coupon_date, next_coupon_date, current_date
                    )
                )
                .otherwise(expr)
            )

        return expr

    @classmethod
    def is_accumulating(cls):
        expr = pl.lit(False)
        for name, accrual_method in cls.stripped_items.items():
            expr = (
                pl.when(pl.col("AccrualMethod") == name).then(pl.lit(accrual_method.is_accumulating())).otherwise(expr)
            )
        return expr


AccrualMethodRegistry.register("Recalculate Actual36525", RecalculateAccrual(Actual36525))
AccrualMethodRegistry.register("Daily Accumulating", DailyAccumulating())
AccrualMethodRegistry.register("None", NoAccrual())
