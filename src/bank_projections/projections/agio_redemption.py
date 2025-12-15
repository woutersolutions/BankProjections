import polars as pl

from bank_projections.financials.balance_sheet import BalanceSheet, MutationReason
from bank_projections.financials.balance_sheet_category import BalanceSheetCategoryRegistry
from bank_projections.financials.balance_sheet_item import BalanceSheetItem
from bank_projections.projections.projectionrule import ProjectionRule
from bank_projections.scenarios.scenario import ScenarioSnapShot
from bank_projections.utils.time import TimeIncrement


class AgioRedemption(ProjectionRule):
    def apply(self, bs: BalanceSheet, increment: TimeIncrement, scenario: ScenarioSnapShot) -> BalanceSheet:
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
        agio_redemption = new_agio - pl.col("Agio")

        bs.mutate(
            BalanceSheetItem(),
            pnls={
                MutationReason(module="Runoff", rule="Agio"): signs * agio_redemption,
            },
            Agio=new_agio,
            AgioRedemption=agio_redemption,
        )

        return bs
