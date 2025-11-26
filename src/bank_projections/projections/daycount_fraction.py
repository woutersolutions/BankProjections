"""Daycount fraction conventions for interest calculations.

This module provides various daycount conventions used in financial calculations
to determine the fraction of a year between two dates.
"""

from abc import ABC, abstractmethod

import polars as pl

from bank_projections.utils.base_registry import BaseRegistry


class DaycountFraction(ABC):
    """Abstract base class for daycount fraction conventions.

    Daycount fractions are used to calculate the portion of a year between two dates
    for interest accrual calculations. Different conventions exist based on market
    practices and instrument types.
    """

    @classmethod
    @abstractmethod
    def year_fraction(cls, start_date: pl.Expr, end_date: pl.Expr) -> pl.Expr:
        """Calculate the year fraction between two dates.

        Args:
            start_date: Polars expression representing the start date.
            end_date: Polars expression representing the end date.

        Returns:
            Polars expression representing the fraction of a year between the dates.
        """
        pass


class DaycountFractionRegistry(BaseRegistry[DaycountFraction], DaycountFraction):
    """Registry for daycount fraction conventions.

    Provides a dispatch mechanism to calculate year fractions based on a
    DaycountBasis column in the data.
    """

    @classmethod
    def year_fraction(cls, start_date: pl.Expr, end_date: pl.Expr) -> pl.Expr:
        """Calculate year fraction using the registered daycount convention.

        Dispatches to the appropriate daycount fraction implementation based
        on the 'DaycountBasis' column in the data.

        Args:
            start_date: Polars expression representing the start date.
            end_date: Polars expression representing the end date.

        Returns:
            Polars expression representing the year fraction.
        """
        expr = pl.lit(0.0)
        for name, daycount in cls.stripped_items.items():
            expr = pl.when(pl.col("DaycountBasis") == name).then(daycount.year_fraction(start_date, end_date)).otherwise(
                expr
            )
        return expr


class Actual360(DaycountFraction):
    """Actual/360 daycount convention.

    Calculates year fraction as: actual days / 360.
    Commonly used for money market instruments.
    """

    @classmethod
    def year_fraction(cls, start_date: pl.Expr, end_date: pl.Expr) -> pl.Expr:
        actual_days = (end_date - start_date).dt.total_days()
        return actual_days / 360.0


class Actual365Fixed(DaycountFraction):
    """Actual/365 Fixed daycount convention.

    Calculates year fraction as: actual days / 365.
    Used for many government bonds and some corporate bonds.
    """

    @classmethod
    def year_fraction(cls, start_date: pl.Expr, end_date: pl.Expr) -> pl.Expr:
        actual_days = (end_date - start_date).dt.total_days()
        return actual_days / 365.0


class Actual36525(DaycountFraction):
    """Actual/365.25 daycount convention.

    Calculates year fraction as: actual days / 365.25.
    Accounts for leap years on average.
    """

    @classmethod
    def year_fraction(cls, start_date: pl.Expr, end_date: pl.Expr) -> pl.Expr:
        actual_days = (end_date - start_date).dt.total_days()
        return actual_days / 365.25


class ActualActualISDA(DaycountFraction):
    """Actual/Actual ISDA daycount convention.

    Calculates year fraction considering actual days in each year,
    accounting for leap years. The fraction is the sum of:
    - days in non-leap years / 365
    - days in leap years / 366

    This is a simplified implementation that approximates by checking
    if the period spans a leap year.
    """

    @classmethod
    def year_fraction(cls, start_date: pl.Expr, end_date: pl.Expr) -> pl.Expr:
        actual_days = (end_date - start_date).dt.total_days()
        # Use the end date's year to determine leap year status for simplicity
        # A more precise implementation would split days across year boundaries
        year = end_date.dt.year()
        is_leap_year = ((year % 4 == 0) & (year % 100 != 0)) | (year % 400 == 0)
        days_in_year = pl.when(is_leap_year).then(366.0).otherwise(365.0)
        return actual_days / days_in_year


class Thirty360BondBasis(DaycountFraction):
    """30/360 Bond Basis (US) daycount convention.

    Assumes 30 days per month and 360 days per year.
    US convention: if end day is 31 and start day < 30, end day becomes 31;
    otherwise, if start day is 31, it becomes 30, and if end day is 31, it becomes 30.
    """

    @classmethod
    def year_fraction(cls, start_date: pl.Expr, end_date: pl.Expr) -> pl.Expr:
        # Cast to i32 to avoid overflow (dt.day() and dt.month() return i8)
        start_day = start_date.dt.day().cast(pl.Int32)
        start_month = start_date.dt.month().cast(pl.Int32)
        start_year = start_date.dt.year()

        end_day = end_date.dt.day().cast(pl.Int32)
        end_month = end_date.dt.month().cast(pl.Int32)
        end_year = end_date.dt.year()

        # Adjust start day: if 31, becomes 30
        adjusted_start_day = pl.when(start_day == 31).then(30).otherwise(start_day)

        # Adjust end day: if 31 and start day >= 30, becomes 30
        adjusted_end_day = pl.when((end_day == 31) & (adjusted_start_day >= 30)).then(30).otherwise(end_day)

        day_diff = adjusted_end_day - adjusted_start_day
        month_diff = (end_month - start_month) * 30
        year_diff = (end_year - start_year) * 360

        return (year_diff + month_diff + day_diff) / 360.0


class Thirty360European(DaycountFraction):
    """30E/360 (Eurobond Basis) daycount convention.

    Assumes 30 days per month and 360 days per year.
    European convention: both start and end day of 31 become 30.
    """

    @classmethod
    def year_fraction(cls, start_date: pl.Expr, end_date: pl.Expr) -> pl.Expr:
        # Cast to i32 to avoid overflow (dt.day() and dt.month() return i8)
        start_day = start_date.dt.day().cast(pl.Int32)
        start_month = start_date.dt.month().cast(pl.Int32)
        start_year = start_date.dt.year()

        end_day = end_date.dt.day().cast(pl.Int32)
        end_month = end_date.dt.month().cast(pl.Int32)
        end_year = end_date.dt.year()

        # Adjust days: if 31, becomes 30
        adjusted_start_day = pl.when(start_day == 31).then(30).otherwise(start_day)
        adjusted_end_day = pl.when(end_day == 31).then(30).otherwise(end_day)

        day_diff = adjusted_end_day - adjusted_start_day
        month_diff = (end_month - start_month) * 30
        year_diff = (end_year - start_year) * 360

        return (year_diff + month_diff + day_diff) / 360.0


class Thirty360ISDA(DaycountFraction):
    """30E/360 ISDA daycount convention.

    Assumes 30 days per month and 360 days per year.
    ISDA convention: day 31 becomes 30, and the last day of February
    is also treated as day 30.
    """

    @classmethod
    def year_fraction(cls, start_date: pl.Expr, end_date: pl.Expr) -> pl.Expr:
        # Cast to i32 to avoid overflow (dt.day() and dt.month() return i8)
        start_day = start_date.dt.day().cast(pl.Int32)
        start_month = start_date.dt.month().cast(pl.Int32)
        start_year = start_date.dt.year()

        end_day = end_date.dt.day().cast(pl.Int32)
        end_month = end_date.dt.month().cast(pl.Int32)
        end_year = end_date.dt.year()

        # Check if date is last day of February (month_end gives last day of month)
        start_is_last_day_of_feb = (start_month == 2) & (start_date == start_date.dt.month_end())
        end_is_last_day_of_feb = (end_month == 2) & (end_date == end_date.dt.month_end())

        # Adjust days: if 31 or last day of February, becomes 30
        adjusted_start_day = pl.when((start_day == 31) | start_is_last_day_of_feb).then(30).otherwise(start_day)
        adjusted_end_day = pl.when((end_day == 31) | end_is_last_day_of_feb).then(30).otherwise(end_day)

        day_diff = adjusted_end_day - adjusted_start_day
        month_diff = (end_month - start_month) * 30
        year_diff = (end_year - start_year) * 360

        return (year_diff + month_diff + day_diff) / 360.0


# Register all daycount fractions
DaycountFractionRegistry.register("actual360", Actual360())
DaycountFractionRegistry.register("actual365fixed", Actual365Fixed())
DaycountFractionRegistry.register("actual36525", Actual36525())
DaycountFractionRegistry.register("actualactual", ActualActualISDA())
DaycountFractionRegistry.register("30360", Thirty360BondBasis())
DaycountFractionRegistry.register("30e360", Thirty360European())
DaycountFractionRegistry.register("30e360isda", Thirty360ISDA())
