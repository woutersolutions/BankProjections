import polars as pl

from bank_projections.financials.balance_sheet import BalanceSheet, MutationReason
from bank_projections.financials.balance_sheet_category import BalanceSheetCategoryRegistry
from bank_projections.financials.balance_sheet_item import BalanceSheetItem
from bank_projections.financials.balance_sheet_metrics import SMALL_NUMBER
from bank_projections.financials.market_data import MarketRates
from bank_projections.projections.rule import Rule
from bank_projections.projections.valuation_method import ValuationMethodRegistry
from bank_projections.utils.time import TimeIncrement


class Valuation(Rule):
    def apply(self, bs: BalanceSheet, increment: TimeIncrement, market_rates: MarketRates) -> BalanceSheet:
        if increment.from_date == increment.to_date:  # No time passed
            return bs

        zero_rates = market_rates.curves.get_zero_rates()
        # TODO: Find a way not to use the _.data here
        bs._data = ValuationMethodRegistry.corrected_dirty_price(
            bs._data, increment.to_date, zero_rates, "NewDirtyPrice"
        )
        new_clean_prices = pl.col("NewDirtyPrice") - pl.col("AccruedInterest") / (pl.col("Quantity") + SMALL_NUMBER)

        signs = BalanceSheetCategoryRegistry.book_value_sign()

        bs.mutate(
            BalanceSheetItem(),
            pnls={
                MutationReason(module="Valuation", rule="Net gains fair value through P&L"): signs
                * pl.when(pl.col("AccountingMethod") == "fairvaluethroughpnl")
                .then((new_clean_prices - pl.col("CleanPrice")) * pl.col("Quantity"))
                .otherwise(0.0),
            },
            ocis={
                MutationReason(module="Valuation", rule="Net gains fair value through OCI"): signs
                * pl.when(pl.col("AccountingMethod") == "fairvaluethroughoci")
                .then((new_clean_prices - pl.col("CleanPrice")) * pl.col("Quantity"))
                .otherwise(0.0),
            },
            CleanPrice=new_clean_prices,
        )

        bs._data = bs._data.drop("NewDirtyPrice")

        return bs
