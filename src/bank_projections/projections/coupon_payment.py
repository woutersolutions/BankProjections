import polars as pl

from bank_projections.financials.balance_sheet import BalanceSheet, MutationReason
from bank_projections.financials.balance_sheet_category import BalanceSheetCategoryRegistry
from bank_projections.financials.balance_sheet_item import BalanceSheetItem
from bank_projections.financials.market_data import MarketRates
from bank_projections.projections.accrual_method import AccrualMethodRegistry
from bank_projections.projections.coupon_type import CouponTypeRegistry
from bank_projections.projections.frequency import FrequencyRegistry
from bank_projections.projections.rule import Rule
from bank_projections.utils.time import TimeIncrement


class CouponPayment(Rule):
    def apply(self, bs: BalanceSheet, increment: TimeIncrement, market_rates: MarketRates) -> BalanceSheet:
        if increment.from_date == increment.to_date:  # No time passed
            return bs

        matured = pl.col("MaturityDate") <= pl.lit(increment.to_date)
        number_of_payments = (
            pl.when(pl.col("NextCouponDate").is_null())
            .then(0)
            .otherwise(
                FrequencyRegistry.number_due(
                    pl.col("NextCouponDate"), pl.min_horizontal(pl.col("MaturityDate"), pl.lit(increment.to_date))
                )
            )
        )

        previous_coupon_date = (
            pl.when(matured)
            .then(pl.col("MaturityDate"))
            .otherwise(FrequencyRegistry.previous_coupon_date(increment.to_date))
        )
        new_coupon_date = pl.when(matured).then(None).otherwise(FrequencyRegistry.next_coupon_date(increment.to_date))
        coupon_payments = coupon_payment(pl.col("Nominal"), pl.col("InterestRate")) * number_of_payments
        floating_rates = market_rates.curves.floating_rate_expr()
        new_interest_rates = (
            pl.when(number_of_payments > 0)
            .then(CouponTypeRegistry.coupon_rate(floating_rates))
            .otherwise(pl.col("InterestRate"))
        )

        cashflow = pl.when(AccrualMethodRegistry.is_accumulating()).then(0.0).otherwise(coupon_payments)
        signs = BalanceSheetCategoryRegistry.book_value_sign()

        bs.mutate(
            BalanceSheetItem(),
            cashflows={
                MutationReason(module="Runoff", rule="Coupon payment"): signs * cashflow,
            },
            AccruedInterest=pl.col("AccruedInterest") - cashflow,
            PreviousCouponDate=previous_coupon_date,
            NextCouponDate=new_coupon_date,
            FloatingRate=floating_rates,
            InterestRate=new_interest_rates,
        )

        return bs


def coupon_payment(
    nominal: pl.Expr,
    coupon_rate: pl.Expr,
) -> pl.Expr:
    return nominal * coupon_rate * FrequencyRegistry.portion_year()
