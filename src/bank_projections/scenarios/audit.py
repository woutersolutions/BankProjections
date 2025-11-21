from typing import Any

import polars as pl

from bank_projections.financials.balance_sheet import BalanceSheet, MutationReason
from bank_projections.financials.balance_sheet_item import BalanceSheetItem, BalanceSheetItemRegistry
from bank_projections.financials.balance_sheet_metric_registry import BalanceSheetMetrics
from bank_projections.financials.market_data import MarketRates
from bank_projections.scenarios.template import KeyValueRuleBase
from bank_projections.utils.date import add_months, is_end_of_month
from bank_projections.utils.parsing import strip_identifier
from bank_projections.utils.time import TimeIncrement


class AuditRule(KeyValueRuleBase):
    def __init__(self, rule_input: dict[str, Any]):
        self.target = BalanceSheetItem()
        for key, value in rule_input.items():
            match strip_identifier(key):
                case "closingmonth":
                    self.closing_month = int(value)
                case "auditmonth":
                    self.audit_month = int(value)
                case _ if (stripped := strip_identifier(key)) is not None and stripped.startswith("target"):
                    label = strip_identifier(key[len("target") :])
                    if label is not None:
                        self.target = self.target.add_identifier(label, value)
                case _:
                    raise KeyError(f"{key} not recognized in AuditRule")

    def apply(self, bs: BalanceSheet, increment: TimeIncrement, market_rates: MarketRates) -> BalanceSheet:
        # Implement the logic to apply the mutation to the balance sheet based on rule_input
        # This is a placeholder implementation

        # See when (and if) in the increment an audit should be done
        current_date = increment.to_date
        while current_date > increment.from_date:
            if current_date.month == self.audit_month and is_end_of_month(current_date):
                audit_date = current_date
                break
            current_date = add_months(current_date, -1, make_end_of_month=True)
        else:
            # No audit in this increment
            return bs

        closing_date = add_months(current_date, -((self.audit_month - self.closing_month) % 12), make_end_of_month=True)
        item = BalanceSheetItemRegistry.get("pnl account").add_condition(
            (pl.col("OriginationDate") <= closing_date) | pl.col("OriginationDate").is_null()
        )
        counter_item = BalanceSheetItemRegistry.get("Retained earnings")
        reason = MutationReason(module="Audit", rule=f"Audit as of {audit_date}", date=audit_date)
        bs.mutate_metric(
            item, BalanceSheetMetrics.get("quantity"), 0, reason, relative=False, counter_item=counter_item
        )

        return bs
