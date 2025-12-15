import polars as pl

from bank_projections.financials.balance_sheet import BalanceSheet, MutationReason
from bank_projections.financials.balance_sheet_item import BalanceSheetItemRegistry
from bank_projections.financials.balance_sheet_metric_registry import BalanceSheetMetrics
from bank_projections.projections.projectionrule import ProjectionRule
from bank_projections.scenarios.scenario import ScenarioSnapShot
from bank_projections.utils.date import add_months, is_end_of_month
from bank_projections.utils.time import TimeIncrement


class AuditRule(ProjectionRule):
    def apply(self, bs: BalanceSheet, increment: TimeIncrement, scenario: ScenarioSnapShot) -> BalanceSheet:
        # Implement the logic to apply the mutation to the balance sheet based on rule_input
        # This is a placeholder implementation

        audit_input = scenario.audit

        # See when (and if) in the increment an audit should be done
        current_date = increment.to_date
        while current_date > increment.from_date:
            if current_date.month == audit_input.audit_month and is_end_of_month(current_date):
                audit_date = current_date
                break
            current_date = add_months(current_date, -1, make_end_of_month=True)
        else:
            # No audit in this increment
            return bs

        closing_date = add_months(
            current_date, -((audit_input.audit_month - audit_input.closing_month) % 12), make_end_of_month=True
        )
        item = BalanceSheetItemRegistry.get("pnl account").add_condition(
            (pl.col("OriginationDate") <= closing_date) | pl.col("OriginationDate").is_null()
        )
        counter_item = BalanceSheetItemRegistry.get("Retained earnings")
        reason = MutationReason(module="Audit", rule=f"Audit as of {audit_date}", date=audit_date)
        bs.mutate_metric(item, BalanceSheetMetrics.get("nominal"), 0, reason, relative=False, counter_item=counter_item)

        return bs
