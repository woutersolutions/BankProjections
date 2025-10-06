import datetime
from abc import ABC, abstractmethod

import polars as pl

from bank_projections.projections.base_registry import BaseRegistry
from bank_projections.utils.date import is_end_of_month


class Frequency(ABC):
    @classmethod
    def previous_coupon_date(
        cls,
        current_date: datetime.date,
        anchor_date: pl.Expr = pl.coalesce("MaturityDate", "NextCouponDate", "PreviousCouponDate", "OriginationDate"),
    ) -> pl.Expr:
        return (
            pl.when(pl.lit(current_date) < pl.col("OriginationDate"))
            .then(pl.lit(None, dtype=pl.Date))
            .otherwise(cls.step_coupon_date(current_date, anchor_date, -1))
        )

    @classmethod
    def next_coupon_date(
        cls,
        current_date: datetime.date,
        anchor_date: pl.Expr = pl.coalesce("MaturityDate", "NextCouponDate", "PreviousCouponDate", "OriginationDate"),
    ) -> pl.Expr:
        return (
            pl.when(pl.lit(current_date) >= pl.col("MaturityDate"))
            .then(pl.lit(None, dtype=pl.Date))
            .otherwise(cls.step_coupon_date(current_date, anchor_date, 0))
        )

    @classmethod
    @abstractmethod
    def step_coupon_date(cls, current_date: datetime.date, anchor_date: pl.Expr, number: int) -> pl.Expr:
        pass

    @classmethod
    @abstractmethod
    def number_due(cls, coupon_date: pl.Expr, projection_date: pl.Expr) -> pl.Expr:
        pass

    @classmethod
    @abstractmethod
    def portion_year(cls) -> pl.Expr:
        pass


class FrequencyRegistry(BaseRegistry[Frequency], Frequency):
    @classmethod
    def step_coupon_date(cls, current_date: datetime.date, anchor_date: pl.Expr, number: int) -> pl.Expr:
        expr = pl.lit(None, dtype=pl.Date)
        for name, freq in cls.items.items():
            expr = (
                pl.when(pl.col("CouponFrequency") == name)
                .then(freq.step_coupon_date(current_date, anchor_date, number))
                .otherwise(expr)
            )
        return expr

    @classmethod
    def number_due(cls, coupon_date: pl.Expr, projection_date: pl.Expr) -> pl.Expr:
        expr = pl.lit(0)
        for name, freq in cls.items.items():
            expr = (
                pl.when(pl.col("CouponFrequency") == name)
                .then(freq.number_due(coupon_date, projection_date))
                .otherwise(expr)
            )
        return expr

    @classmethod
    def portion_year(cls) -> pl.Expr:
        expr = pl.lit(0.0)
        for name, freq in cls.items.items():
            expr = pl.when(pl.col("CouponFrequency") == name).then(freq.portion_year()).otherwise(expr)
        return expr


class MonthlyBase(Frequency):
    number_of_months: int = 0  # Needs to be overridden

    @classmethod
    def step_coupon_date(cls, current_date: datetime.date, anchor_date: pl.Expr, number: int) -> pl.Expr:
        months_to_anchor = (anchor_date.dt.year() - current_date.year) * 12 + (
            anchor_date.dt.month() - current_date.month
        )
        day_passed = pl.lit(True) if is_end_of_month(current_date) else current_date.day >= anchor_date.dt.day()
        payments_to_anchor = (months_to_anchor + pl.when(day_passed).then(0).otherwise(1)) // cls.number_of_months
        return anchor_date.dt.offset_by(by=(-(payments_to_anchor - number) * cls.number_of_months).cast(pl.Utf8) + "mo")

    @classmethod
    def number_due(cls, coupon_date: pl.Expr, projection_date: pl.Expr) -> pl.Expr:
        return pl.when(coupon_date.dt.day() <= projection_date.dt.day()).then(1).otherwise(0) + (
            (projection_date.dt.year() - coupon_date.dt.year()) * 12 // cls.number_of_months
            + (projection_date.dt.month() - coupon_date.dt.month()) // cls.number_of_months
        )

    @classmethod
    def portion_year(cls) -> pl.Expr:
        return pl.lit(cls.number_of_months / 12)


class Monthly(MonthlyBase):
    number_of_months = 1


class Quarterly(MonthlyBase):
    number_of_months = 3


class SemiAnnual(MonthlyBase):
    number_of_months = 6


class Annual(MonthlyBase):
    number_of_months = 12


class DailyBase(Frequency):
    number_of_days: int = 0  # Needs to be overridden

    @classmethod
    def number_due(cls, coupon_date: pl.Expr, projection_date: pl.Expr) -> pl.Expr:
        return ((projection_date - coupon_date).dt.total_days() // cls.number_of_days) + 1

    @classmethod
    def step_coupon_date(cls, current_date: datetime.date, anchor_date: pl.Expr, number: int) -> pl.Expr:
        # Completed d-day steps from anchor up to (and possibly including) current_date
        steps_completed = ((pl.lit(current_date) - anchor_date).dt.total_days()) // pl.lit(cls.number_of_days)

        # Next (> current) has index steps_completed + 1; add 'number' to step forward/backward
        k = steps_completed + pl.lit(1 + number)

        return anchor_date.dt.offset_by(by=((k * pl.lit(cls.number_of_days)).cast(pl.Utf8) + pl.lit("d")))

    @classmethod
    def portion_year(cls) -> pl.Expr:
        return pl.lit(cls.number_of_days / 365.25)


class Daily(DailyBase):
    number_of_days = 1


class Weekly(DailyBase):
    number_of_days = 7


class Never(Frequency):
    @classmethod
    def step_coupon_date(cls, current_date: datetime.date, anchor_date: pl.Expr, number: int) -> pl.Expr:
        return pl.lit(None, dtype=pl.Date)

    @classmethod
    def number_due(cls, coupon_date: pl.Expr, projection_date: pl.Expr) -> pl.Expr:
        return pl.lit(0)

    @classmethod
    def portion_year(cls) -> pl.Expr:
        return pl.lit(0.0)


FrequencyRegistry.register("Monthly", Monthly())
FrequencyRegistry.register("Quarterly", Quarterly())
FrequencyRegistry.register("SemiAnnual", SemiAnnual())
FrequencyRegistry.register("Annual", Annual())
FrequencyRegistry.register("Daily", Daily())
FrequencyRegistry.register("Weekly", Weekly())
FrequencyRegistry.register("Never", Never())


def interest_accrual(
    quantity: pl.Expr,
    interest_rate: pl.Expr,
    previous_coupon_date: pl.Expr,
    next_coupon_date: pl.Expr,
    current_date: datetime.date,
) -> pl.Expr:
    days_fraction = (current_date - previous_coupon_date).dt.total_days() / (
        next_coupon_date - previous_coupon_date
    ).dt.total_days()
    return days_fraction.fill_null(0) * quantity * interest_rate
