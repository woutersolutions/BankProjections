import datetime
from abc import ABC, abstractmethod

import polars as pl


class Frequency(ABC):
    @classmethod
    @abstractmethod
    def advance_next(cls, date: pl.Expr, number: int) -> pl.Expr:
        pass

    @classmethod
    @abstractmethod
    def number_due(cls, coupon_date: pl.Expr, projection_date: pl.Expr) -> pl.Expr:
        pass

    @classmethod
    @abstractmethod
    def portion_passed(cls, next_coupon_date: pl.Expr, projection_date: datetime.date) -> pl.Expr:
        pass

    @classmethod
    @abstractmethod
    def portion_year(cls) -> pl.Expr:
        pass


class FrequencyRegistry(Frequency):
    _registry: dict[str, type[Frequency]] = {}

    @classmethod
    def register(cls, name: str, frequency: type[Frequency]) -> None:
        if not issubclass(frequency, Frequency):
            raise ValueError(f"Class {frequency} must be a subclass of Frequency")
        cls._registry[name] = frequency

    @classmethod
    def advance_next(cls, date: pl.Expr, number: pl.Expr) -> pl.Expr:
        expr = date
        for name, freq in cls._registry.items():
            expr = pl.when(pl.col("CouponFrequency") == name).then(freq.advance_next(date, number)).otherwise(expr)
        return expr

    @classmethod
    def number_due(cls, coupon_date: pl.Expr, projection_date: pl.Expr) -> pl.Expr:
        expr = pl.lit(0)
        for name, freq in cls._registry.items():
            expr = (
                pl.when(pl.col("CouponFrequency") == name)
                .then(freq.number_due(coupon_date, projection_date))
                .otherwise(expr)
            )
        return expr

    @classmethod
    def portion_passed(cls, next_coupon_date: pl.Expr, projection_date: datetime.date) -> pl.Expr:
        expr = pl.lit(0.0)
        for name, freq in cls._registry.items():
            expr = (
                pl.when(pl.col("CouponFrequency") == name)
                .then(freq.portion_passed(next_coupon_date, projection_date))
                .otherwise(expr)
            )
        return expr

    @classmethod
    def portion_year(cls) -> pl.Expr:
        expr = pl.lit(0.0)
        for name, freq in cls._registry.items():
            expr = pl.when(pl.col("CouponFrequency") == name).then(freq.portion_year()).otherwise(expr)
        return expr


class MonthlyBase(Frequency):
    number_of_months = None  # Needs to be overridden

    @classmethod
    def advance_next(cls, date: pl.Expr, number: pl.Expr) -> pl.Expr:
        return date.dt.offset_by((number * cls.number_of_months).cast(pl.Utf8) + "mo")

    @classmethod
    def portion_passed(cls, next_coupon_date: pl.Expr, projection_date: datetime.date) -> pl.Expr:
        return 1 - (next_coupon_date - pl.lit(projection_date)).dt.total_days() / (30 * cls.number_of_months)

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
    number_of_days: int = None  # Needs to be overridden

    @classmethod
    def advance_next(cls, date: pl.Expr, number: pl.Expr) -> pl.Expr:
        return date + pl.duration(days=number * cls.number_of_days)

    @classmethod
    def number_due(cls, coupon_date: pl.Expr, projection_date: pl.Expr) -> pl.Expr:
        return (projection_date - coupon_date).dt.total_days() // cls.number_of_days

    @classmethod
    def portion_passed(cls, next_coupon_date: pl.Expr, projection_date: datetime.date) -> pl.Expr:
        return (cls.number_of_days - (next_coupon_date - projection_date).dt.total_days()) / cls.number_of_days

    @classmethod
    def portion_year(cls) -> pl.Expr:
        return pl.lit(cls.number_of_days / 365.25)


class Daily(DailyBase):
    number_of_days = 1


class Weekly(DailyBase):
    number_of_days = 7


class Never(Frequency):
    @classmethod
    def advance_next(cls, date: pl.Expr, number: pl.Expr) -> pl.Expr:
        return date

    @classmethod
    def number_due(cls, coupon_date: pl.Expr, projection_date: pl.Expr) -> pl.Expr:
        return pl.lit(0)

    @classmethod
    def portion_passed(cls, next_coupon_date: pl.Expr, projection_date: datetime.date) -> pl.Expr:
        return pl.lit(0.0)

    @classmethod
    def portion_year(cls) -> pl.Expr:
        return pl.lit(0.0)


FrequencyRegistry.register("Monthly", Monthly)
FrequencyRegistry.register("Quarterly", Quarterly)
FrequencyRegistry.register("SemiAnnual", SemiAnnual)
FrequencyRegistry.register("Annual", Annual)
FrequencyRegistry.register("Daily", Daily)
FrequencyRegistry.register("Weekly", Weekly)
FrequencyRegistry.register("Never", Never)
