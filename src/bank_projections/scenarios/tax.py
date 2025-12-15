import polars as pl

from bank_projections.financials.balance_sheet import BalanceSheet, MutationReason
from bank_projections.projections.projectionrule import ProjectionRule
from bank_projections.scenarios.scenario import ScenarioSnapShot
from bank_projections.utils.time import TimeIncrement


class TaxProjectionRule(ProjectionRule):
    def apply(self, bs: BalanceSheet, increment: TimeIncrement, scenario: ScenarioSnapShot) -> BalanceSheet:
        income, expense = bs.pnls.select(
            income=pl.when(pl.col("Amount") < 0).then(pl.col("Amount")).otherwise(0).sum()
            * -pl.lit(scenario.tax.tax_rate),
            expense=pl.when(pl.col("Amount") > 0).then(pl.col("Amount")).otherwise(0).sum()
            * -pl.lit(scenario.tax.tax_rate),
        ).row(0)

        bs.add_single_pnl(expense, MutationReason(module="Tax", rule="Tax expense"), offset_liquidity=True)
        bs.add_single_pnl(income, MutationReason(module="Tax", rule="Tax benefit"), offset_liquidity=True)

        return bs
