from bank_projections.financials.balance_sheet import BalanceSheet
from bank_projections.financials.balance_sheet_metrics import BalanceSheetMetric
from bank_projections.projections.projectionrule import ProjectionRule
from bank_projections.scenarios.scenario import ScenarioSnapShot
from bank_projections.scenarios.scenario_input_type import BalanceSheetMutationInputItem
from bank_projections.utils.time import TimeIncrement


class BalanceSheetMutationRule(ProjectionRule):
    def apply(self, bs: BalanceSheet, increment: TimeIncrement, scenario: ScenarioSnapShot) -> BalanceSheet:
        for mutation_item in scenario.mutations:
            if isinstance(mutation_item.metric, BalanceSheetMetric):
                bs = self.apply_item(bs, increment, mutation_item)
        return bs

    def apply_item(
        self, bs: BalanceSheet, increment: TimeIncrement, mutation_item: BalanceSheetMutationInputItem
    ) -> BalanceSheet:
        # Implement the logic to apply the mutation to the balance sheet based on rule_input

        item = mutation_item.item.add_cohort_expressions(mutation_item.cohorts, increment.to_date)

        if mutation_item.date is None or increment.contains(mutation_item.date):
            bs.mutate_metric(
                item,
                mutation_item.metric,
                mutation_item.amount,
                mutation_item.reason,
                mutation_item.relative,
                mutation_item.multiplicative,
                mutation_item.offset_liquidity,
                mutation_item.offset_pnl,
                mutation_item.counter_item,
            )

        return bs
