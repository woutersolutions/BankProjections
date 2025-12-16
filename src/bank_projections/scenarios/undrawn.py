import polars as pl

from bank_projections.financials.balance_sheet import BalanceSheet, MutationReason
from bank_projections.financials.balance_sheet_category import BalanceSheetCategoryRegistry
from bank_projections.financials.balance_sheet_item import BalanceSheetItem
from bank_projections.projections.projectionrule import ProjectionRule
from bank_projections.scenarios.scenario import ScenarioSnapShot
from bank_projections.utils.time import TimeIncrement


class DrawDownRule(ProjectionRule):
    def apply(self, bs: BalanceSheet, increment: TimeIncrement, scenario: ScenarioSnapShot) -> BalanceSheet:
        if increment.from_date == increment.to_date:  # No time passed
            return bs

        draw_downs = pl.col("CCF").fill_null(0.0) * increment.portion_year * pl.col("Undrawn")
        for mutation in [mutation for mutation in scenario.mutations if mutation.metric == "drawdown"]:
            draw_downs = (
                pl.when(mutation.item.filter_expression)
                .then(
                    pl.lit(mutation.amount)
                    / (pl.col("Nominal") * mutation.item.filter_expression).sum()
                    * pl.col("Nominal")
                )
                .otherwise(draw_downs)
            )
        for mutation in [mutation for mutation in scenario.mutations if mutation.metric == "drawdownrate"]:
            draw_downs = (
                pl.when(mutation.item.filter_expression)
                .then(pl.col("Undrawn") * pl.lit(mutation.amount))
                .otherwise(draw_downs)
            )

        top_ups = pl.lit(0.0)
        for mutation in [mutation for mutation in scenario.mutations if mutation.metric == "topup"]:
            top_ups = (
                pl.when(mutation.item.filter_expression)
                .then(
                    pl.col("Nominal")
                    * pl.lit(mutation.amount)
                    / (pl.col("Nominal") * mutation.item.filter_expression).sum()
                )
                .otherwise(top_ups)
            )
        for mutation in [mutation for mutation in scenario.mutations if mutation.metric == "topuprate"]:
            top_ups = (
                pl.when(mutation.item.filter_expression)
                .then(pl.col("Nominal") * pl.lit(mutation.amount))
                .otherwise(top_ups)
            )

        signs = BalanceSheetCategoryRegistry.book_value_sign()

        bs.mutate(
            BalanceSheetItem(),
            cashflows={
                MutationReason(module="DrawDown", rule="Draw downs"): -signs * draw_downs,
            },
            Nominal=pl.col("Nominal") + draw_downs,
            Undrawn=pl.col("Undrawn") - draw_downs + top_ups,
        )

        return bs
