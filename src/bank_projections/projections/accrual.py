import polars as pl

from bank_projections.financials.balance_sheet import BalanceSheet, MutationReason
from bank_projections.financials.balance_sheet_category import BalanceSheetCategoryRegistry
from bank_projections.financials.balance_sheet_item import BalanceSheetItem
from bank_projections.financials.market_data import MarketRates
from bank_projections.projections.accrual_method import AccrualMethodRegistry
from bank_projections.projections.rule import Rule
from bank_projections.utils.time import TimeIncrement


class Accrual(Rule):
    def apply(self, bs: BalanceSheet, increment: TimeIncrement, market_rates: MarketRates) -> BalanceSheet:
        if increment.from_date == increment.to_date:  # No time passed
            return bs

        accrual = AccrualMethodRegistry.interest_accrual(
            pl.col("Nominal") + pl.col("Notional"),
            pl.col("InterestRate"),
            pl.col("PreviousCouponDate"),
            pl.col("NextCouponDate"),
            increment.to_date,
        )
        is_accumulating = AccrualMethodRegistry.is_accumulating()
        signs = BalanceSheetCategoryRegistry.book_value_sign()

        bs.mutate(
            BalanceSheetItem(),
            pnls={
                MutationReason(module="Runoff", rule="Accrual"): signs * accrual,
            },
            Nominal=pl.col("Nominal") + pl.when(is_accumulating).then(accrual).otherwise(0.0),
            AccruedInterest=pl.col("AccruedInterest") + pl.when(is_accumulating).then(0.0).otherwise(accrual),
            Accrual=accrual,
        )

        return bs
