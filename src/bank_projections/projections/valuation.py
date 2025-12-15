import polars as pl

from bank_projections.financials.balance_sheet import BalanceSheet, MutationReason
from bank_projections.financials.balance_sheet_category import BalanceSheetCategoryRegistry
from bank_projections.financials.balance_sheet_item import BalanceSheetItem
from bank_projections.projections.projectionrule import ProjectionRule
from bank_projections.projections.valuation_method import ValuationMethodRegistry
from bank_projections.scenarios.scenario import ScenarioSnapShot
from bank_projections.utils.time import TimeIncrement


class Valuation(ProjectionRule):
    def apply(self, bs: BalanceSheet, increment: TimeIncrement, scenario: ScenarioSnapShot) -> BalanceSheet:
        if increment.from_date == increment.to_date:  # No time passed
            return bs

        zero_rates = scenario.curves.get_zero_rates()
        # TODO: Find a way not to use the _.data here
        bs._data = ValuationMethodRegistry.corrected_dirty_price(
            bs._data, increment.to_date, zero_rates, "NewDirtyPrice"
        )
        new_fair_value_adjustment = (
            pl.col("NewDirtyPrice") * (pl.col("Nominal") + pl.col("Notional"))
            - pl.col("AccruedInterest")
            - pl.col("Nominal")
            - pl.col("Impairment")
        )
        signs = BalanceSheetCategoryRegistry.book_value_sign()
        fair_value_change = new_fair_value_adjustment - pl.col("FairValueAdjustment")

        bs.mutate(
            BalanceSheetItem(AccountingMethod="fairvaluethroughpnl")
            | BalanceSheetItem(AccountingMethod="fairvaluethroughoci"),
            pnls={
                MutationReason(module="Valuation", rule="Net gains fair value through P&L"): pl.when(
                    pl.col("AccountingMethod") == "fairvaluethroughpnl"
                )
                .then(signs * fair_value_change)
                .otherwise(0.0),
            },
            ocis={
                MutationReason(module="Valuation", rule="Net gains fair value through OCI"): pl.when(
                    pl.col("AccountingMethod") == "fairvaluethroughoci"
                )
                .then(signs * fair_value_change)
                .otherwise(0.0),
            },
            FairValueAdjustment=new_fair_value_adjustment,
            DirtyPrice=pl.col("NewDirtyPrice"),
            FairValueChange=fair_value_change,
        )

        bs._data = bs._data.drop("NewDirtyPrice")

        return bs
