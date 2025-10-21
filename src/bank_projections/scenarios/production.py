import datetime
from typing import Any

import pandas as pd
import polars as pl

from bank_projections.config import Config
from bank_projections.financials.balance_sheet import BalanceSheet, MutationReason
from bank_projections.financials.balance_sheet_item import BalanceSheetItem, BalanceSheetItemRegistry
from bank_projections.financials.balance_sheet_metrics import BalanceSheetMetrics
from bank_projections.financials.market_data import MarketRates
from bank_projections.scenarios.template import AmountRuleBase
from bank_projections.utils.date import add_months
from bank_projections.utils.parsing import get_identifier, is_in_identifiers, read_bool, read_date, strip_identifier
from bank_projections.utils.time import TimeIncrement


class ProductionRule(AmountRuleBase):
    def __init__(self, rule_input: dict[str, Any]):
        self.multiplicative = False
        self.reason = MutationReason(rule="Production")
        self.date: datetime.date | None = None
        self.metrics: dict[str, Any] = {}
        self.labels: dict[str, Any] = {}
        self.maturity: int | None = None

        if is_in_identifiers("reference item", list(rule_input.keys())):
            value = rule_input[get_identifier("reference item", list(rule_input.keys()))]
            if pd.isna(value) or value == "":
                self.reference_item = None
            else:
                self.reference_item = BalanceSheetItemRegistry.get(value)
        else:
            self.reference_item = None

        for key, value in rule_input.items():
            match strip_identifier(key):
                case _ if pd.isna(value) or value == "":
                    pass
                case "referenceitem":
                    pass
                case _ if is_in_identifiers(key, list(BalanceSheetMetrics.stripped_names())):
                    stripped_key = strip_identifier(key)
                    if stripped_key is not None:
                        self.metrics[stripped_key] = value
                case _ if (stripped := strip_identifier(key)) is not None and stripped.startswith("reference"):
                    stripped_key = strip_identifier(key)
                    if stripped_key is not None:
                        label = get_identifier(stripped_key.replace("reference", ""), Config.label_columns())
                        if self.reference_item is None:
                            self.reference_item = BalanceSheetItem(**{label: value})
                        else:
                            self.reference_item = self.reference_item.add_identifier(label, value)
                case _ if is_in_identifiers(key, Config.label_columns()):
                    self.labels[get_identifier(key, Config.label_columns())] = value
                case "multiplicative":
                    self.multiplicative = read_bool(value)
                case "date":
                    self.date = read_date(value)
                case "maturity":
                    self.maturity = value
                case _:
                    raise KeyError(f"{key} not recognized in BalanceSheetMutationRule")

    def apply(self, bs: BalanceSheet, increment: TimeIncrement, market_rates: MarketRates) -> BalanceSheet:
        # Implement the logic to apply the mutation to the balance sheet based on rule_input
        # This is a placeholder implementation

        if self.date is None or increment.contains(self.date):
            if self.reference_item is None:
                # TODO: Implement production without reference item
                raise NotImplementedError("Production without reference item not yet implement")
            else:
                reason = MutationReason(module="Production", rule="Production")

                if self.date is None:
                    raise ValueError("Date must be specified for production")
                maturity_date = None if self.maturity is None else add_months(self.date, 12 * self.maturity)

                bs.add_item(
                    self.reference_item,
                    labels=self.labels,
                    metrics=self.metrics,
                    origination_date=self.date,
                    maturity_date=maturity_date,
                    pnls={reason: pl.col("Impairment")},
                    cashflows={reason: -pl.col("Quantity") - pl.col("AccruedInterest") - pl.col("Agio")},
                )

        return bs

    @staticmethod
    def _process_metric(data: pl.DataFrame, metrics: dict[str, Any] | float, metric_name: str) -> pl.DataFrame:
        metric = BalanceSheetMetrics.get(metric_name)
        metric_value = metrics if isinstance(metrics, float) else metrics.pop(metric_name)
        data = data.with_columns(metric.mutation_expression(metric_value, pl.lit(True)).alias(metric.mutation_column))

        return data
