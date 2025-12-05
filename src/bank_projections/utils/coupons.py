import datetime

import polars as pl

from bank_projections.projections.frequency import FrequencyRegistry
from bank_projections.utils.daycounting import Actual36525, DaycountFraction

DAYCOUNT_ACCRUAL: DaycountFraction = Actual36525


def interest_accrual(
    nominal: pl.Expr,
    coupon_rate: pl.Expr,
    previous_coupon_date: pl.Expr,
    next_coupon_date: pl.Expr,
    current_date: datetime.date,
) -> pl.Expr:
    days_fraction = DAYCOUNT_ACCRUAL.year_fraction(
        previous_coupon_date, pl.lit(current_date)
    ) / DAYCOUNT_ACCRUAL.year_fraction(previous_coupon_date, next_coupon_date)
    return days_fraction.fill_null(0) * coupon_payment(nominal, coupon_rate)


def coupon_payment(
    nominal: pl.Expr,
    coupon_rate: pl.Expr,
) -> pl.Expr:
    return nominal * coupon_rate * FrequencyRegistry.portion_year()
