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

        new_accrual = AccrualMethodRegistry.interest_accrual(
            pl.col("Nominal") + pl.col("Notional"),
            pl.col("InterestRate"),
            pl.col("PreviousCouponDate"),
            pl.col("NextCouponDate"),
            increment.to_date,
        ) + pl.col("AccruedInterestError")

        matured = pl.col("MaturityDate") <= pl.lit(increment.to_date)
        new_accrual = pl.when(matured).then(0.0).otherwise(new_accrual)

        signs = BalanceSheetCategoryRegistry.book_value_sign()

        bs.mutate(
            BalanceSheetItem(),
            pnls={
                MutationReason(module="Runoff", rule="Accrual"): signs * (new_accrual - pl.col("AccruedInterest")),
            },
            AccruedInterest=new_accrual,
        )

        return bs
