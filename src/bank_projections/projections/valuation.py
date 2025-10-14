import polars as pl

from bank_projections.financials.balance_sheet import BalanceSheet, MutationReason
from bank_projections.financials.balance_sheet_item import BalanceSheetItem
from bank_projections.financials.metrics import SMALL_NUMBER
from bank_projections.projections.market_data import MarketRates
from bank_projections.projections.rule import Rule
from bank_projections.projections.time import TimeIncrement
from bank_projections.projections.valuation_method import ValuationMethodRegistry


class Valuation(Rule):
    def apply(self, bs: BalanceSheet, increment: TimeIncrement, market_rates: MarketRates) -> BalanceSheet:
        if increment.from_date == increment.to_date:  # No time passed
            return bs

        # Apply runoff to all instruments that origination before the current projection date # TODO: refine
        item = BalanceSheetItem(
            expr=(
                (pl.col("OriginationDate").is_null() | (pl.col("OriginationDate") < increment.to_date))
                & (pl.col("AccountingMethod") == "fairvaluethroughoci")
            )
        )

        zero_rates = market_rates.curves.get_zero_rates()
        # TODO: Find a way not to use the _.data here
        bs._data = ValuationMethodRegistry.corrected_dirty_price(
            bs._data, increment.to_date, zero_rates, "NewDirtyPrice"
        )
        new_clean_prices = pl.col("NewDirtyPrice") - pl.col("AccruedInterest") / (pl.col("Quantity") + SMALL_NUMBER)

        # TODO: Consider doing revaluation before runoff
        bs.mutate(
            item,
            pnls={
                MutationReason(module="Valuation", rule="Net gains fair value through P&L"): pl.when(
                    pl.col("AccountingMethod") == "fairvaluethroughpnl"
                )
                .then((new_clean_prices - pl.col("CleanPrice")) * pl.col("Quantity"))
                .otherwise(0.0),
            },
            ocis={
                MutationReason(module="Valuation", rule="Net gains fair value through OCI"): pl.when(
                    pl.col("AccountingMethod") == "fairvaluethroughoci"
                )
                .then((new_clean_prices - pl.col("CleanPrice")) * pl.col("Quantity"))
                .otherwise(0.0),
            },
            CleanPrice=new_clean_prices,
        )

        bs._data = bs._data.drop("NewDirtyPrice")

        bs.validate()

        return bs
