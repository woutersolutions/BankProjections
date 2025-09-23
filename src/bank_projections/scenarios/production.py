import datetime
from typing import Any

import pandas as pd
import polars as pl

from bank_projections.config import Config
from bank_projections.financials.balance_sheet import BalanceSheet, MutationReason
from bank_projections.financials.balance_sheet_item import BalanceSheetItem, BalanceSheetItemRegistry
from bank_projections.financials.metrics import BalanceSheetMetrics
from bank_projections.projections.frequency import FrequencyRegistry
from bank_projections.projections.time import TimeIncrement
from bank_projections.scenarios.template import AmountRuleBase
from bank_projections.utils.date import add_months
from bank_projections.utils.parsing import clean_identifier, get_identifier, is_in_identifiers, read_bool, read_date


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
            match clean_identifier(key):
                case _ if pd.isna(value) or value == "":
                    pass
                case "referenceitem":
                    pass
                case _ if is_in_identifiers(key, list(BalanceSheetMetrics.items.keys())):
                    self.metrics[clean_identifier(key)] = value
                case _ if clean_identifier(key).startswith("reference"):
                    label = get_identifier(clean_identifier(key).replace("reference", ""), Config.label_columns())
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

    def apply(self, bs: BalanceSheet, increment: TimeIncrement, market_rates) -> BalanceSheet:
        # Implement the logic to apply the mutation to the balance sheet based on rule_input
        # This is a placeholder implementation

        if self.date is None or increment.contains(self.date):
            if self.reference_item is None:
                # TODO: Implement production without reference item
                raise NotImplementedError("Production without reference item not yet implement")
            else:
                # TODO: Restructure balance sheet and positions class to avoid direct access to _data

                # Find unique labels for non-numeric columns
                n_uniques = bs._data.filter(self.reference_item.filter_expression).select(
                    [pl.col(col).n_unique().alias(col) for col in Config.BALANCE_SHEET_LABELS]
                )
                constant_cols = [c for c in Config.BALANCE_SHEET_LABELS if n_uniques[0, c] == 1]

                data = (
                    bs._data.filter(self.reference_item.filter_expression)
                    .with_columns(**self.labels)
                    .group_by(
                        set(constant_cols)
                        | set(Config.BALANCE_SHEET_AGGREGATION_LABELS)
                        | set(Config.CLASSIFICATIONS.keys())
                        | set(self.labels.keys())
                    )
                    .agg(
                        [
                            metric.aggregation_expression.alias(metric.column)
                            for name, metric in BalanceSheetMetrics.items.items()
                            if metric.is_stored
                        ]
                    )
                )

                metrics = self.metrics.copy()
                if "quantity" in self.metrics:
                    data = self._process_metric(data, metrics, "quantity")
                else:
                    raise ValueError("Must specify quantity in ProductionRule")

                if "impairment" in self.metrics and "coveragerate" in self.metrics:
                    raise ValueError("Cannot specify both impairment and coveragerate in ProductionRule")
                if "impairment" in self.metrics:
                    data = self._process_metric(data, metrics, "impairment")
                if "coveragerate" in self.metrics:
                    data = self._process_metric(data, metrics, "coveragerate")
                else:
                    # Assume the same coverage rate if not specified
                    coverage_rate = bs.get_amount(self.reference_item, "coveragerate")
                    data = self._process_metric(data, coverage_rate, "coveragerate")

                if "agio" in self.metrics and "agioweight" in self.metrics:
                    raise ValueError("Cannot specify both agio and agio weight in ProductionRule")
                if "agio" in self.metrics:
                    data = self._process_metric(data, metrics, "agio")
                if "agioweight" in self.metrics:
                    data = self._process_metric(data, metrics, "agioweight")
                else:
                    # Assume no agio if not specified
                    data = self._process_metric(data, 0.0, "agio")

                # Determine the dates
                if self.maturity is None:
                    data = data.with_columns(
                        OriginationDate=pl.lit(self.date),
                        MaturityDate=pl.lit(None),
                        NextCouponDate=pl.lit(None),
                        AccruedInterest=pl.lit(0.0),
                    )
                else:
                    maturity_date = add_months(self.date, 12 * self.maturity)
                    data = data.with_columns(
                        OriginationDate=pl.lit(self.date),
                        MaturityDate=pl.lit(maturity_date),
                        NextCouponDate=FrequencyRegistry.advance_next(pl.lit(self.date), pl.lit(1)),
                        AccruedInterest=pl.lit(0.0),  # TODO
                    )

                for metric_name, metric_value in self.metrics.items():
                    metric = BalanceSheetMetrics.get(metric_name)
                    data = data.with_columns(
                        metric.mutation_expression(metric_value, pl.lit(True)).alias(metric.mutation_column)
                    )

                missing_columns = set(bs._data.columns) - set(data.columns)
                if missing_columns:
                    raise ValueError(f"Missing columns in production data: {missing_columns}")
                extra_columns = set(data.columns) - set(bs._data.columns)
                if extra_columns:
                    raise ValueError(f"Extra columns in production data: {extra_columns}")

                bs._data = pl.concat([bs._data, data], how="diagonal")

                # Offset the production in cash and pnl
                reason = MutationReason(module="Production", rule="Production")
                bs.add_liquidity(data, -pl.col("Quantity") - pl.col("AccruedInterest") - pl.col("Agio"), reason)
                bs.add_pnl(data, pl.col("Impairment"), reason)

        return bs

    @staticmethod
    def _process_metric(data: pl.DataFrame, metrics: dict[str, Any] | float, metric_name: str) -> pl.DataFrame:
        metric = BalanceSheetMetrics.get(metric_name)
        if isinstance(metrics, float):
            metric_value = metrics
        else:
            metric_value = metrics.pop(metric_name)
        data = data.with_columns(metric.mutation_expression(metric_value, pl.lit(True)).alias(metric.mutation_column))

        return data
