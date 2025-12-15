from typing import Any

import polars as pl

from bank_projections.financials.balance_sheet import BalanceSheet, MutationReason
from bank_projections.financials.balance_sheet_metric_registry import BalanceSheetMetrics
from bank_projections.projections.projectionrule import ProjectionRule
from bank_projections.scenarios.scenario import ScenarioSnapShot
from bank_projections.scenarios.scenario_input_type import ProductionInputItem
from bank_projections.utils.date import add_months
from bank_projections.utils.time import TimeIncrement


class ProductionRule(ProjectionRule):
    def apply(self, bs: BalanceSheet, increment: TimeIncrement, scenario: ScenarioSnapShot) -> BalanceSheet:
        for production_item in scenario.production:
            bs = self.apply_item(bs, production_item)

        return bs

    def apply_item(self, bs: BalanceSheet, production_item: ProductionInputItem) -> BalanceSheet:
        if production_item.reference_item is None:
            # TODO: Implement production without reference item
            raise NotImplementedError("Production without reference item not yet implement")
        else:
            reason = MutationReason(module="Production", rule="Production")

            if production_item.date is None:
                raise ValueError("Date must be specified for production")
            maturity_date = (
                None
                if production_item.maturity is None
                else add_months(production_item.date, 12 * production_item.maturity)
            )

            sign = bs.get_item_book_value_sign(production_item.reference_item)

            bs.add_item(
                production_item.reference_item,
                labels=production_item.labels,
                metrics=production_item.metrics,
                origination_date=production_item.date,
                maturity_date=maturity_date,
                pnls={reason: sign * pl.col("Impairment")},
                cashflows={reason: -sign * (pl.col("Nominal") + pl.col("AccruedInterest") + pl.col("Agio"))},
            )

        return bs

    @staticmethod
    def _process_metric(data: pl.DataFrame, metrics: dict[str, Any] | float, metric_name: str) -> pl.DataFrame:
        metric = BalanceSheetMetrics.get(metric_name)
        metric_value = metrics if isinstance(metrics, float) else metrics.pop(metric_name)
        data = data.with_columns(metric.mutation_expression(metric_value, pl.lit(True)).alias(metric.mutation_column))

        return data
