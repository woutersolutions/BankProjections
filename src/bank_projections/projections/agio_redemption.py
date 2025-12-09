import polars as pl

from bank_projections.financials.balance_sheet import BalanceSheet, MutationReason
from bank_projections.financials.balance_sheet_category import BalanceSheetCategoryRegistry
from bank_projections.financials.balance_sheet_item import BalanceSheetItem
from bank_projections.financials.market_data import MarketRates
from bank_projections.projections.rule import Rule
from bank_projections.utils.time import TimeIncrement


class AgioRedemption(Rule):
    def apply(self, bs: BalanceSheet, increment: TimeIncrement, market_rates: MarketRates) -> BalanceSheet:
        if increment.from_date == increment.to_date:  # No time passed
            return bs

        matured = pl.col("MaturityDate") <= pl.lit(increment.to_date)

        # For now, assume agio decreases linear
        new_agio = (
            pl.when(pl.col("MaturityDate").is_null())
            .then(pl.col("Agio"))
            .when(matured)
            .then(0.0)
            .otherwise(
                pl.col("Agio")
                * (pl.col("MaturityDate") - increment.to_date).dt.total_days()
                / (pl.col("MaturityDate") - increment.from_date).dt.total_days()
            )
        )

        signs = BalanceSheetCategoryRegistry.book_value_sign()

        bs.mutate(
            BalanceSheetItem(),
            pnls={
                MutationReason(module="Runoff", rule="Agio"): signs * (new_agio - pl.col("Agio")),
            },
            Agio=new_agio,
        )

        return bs
