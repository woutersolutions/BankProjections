import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd
import polars as pl
from dateutil.relativedelta import relativedelta

from bank_projections.config import Config
from bank_projections.financials.balance_sheet import BalanceSheet, MutationReason
from bank_projections.financials.balance_sheet_item import BalanceSheetItem, BalanceSheetItemRegistry
from bank_projections.financials.balance_sheet_metrics import BalanceSheetMetrics
from bank_projections.financials.market_data import MarketRates
from bank_projections.scenarios.template import AmountRuleBase
from bank_projections.utils.parsing import (
    get_identifier,
    is_in_identifiers,
    read_bool,
    read_date,
    read_int,
    strip_identifier,
)
from bank_projections.utils.time import TimeIncrement


class BalanceSheetMutationRule(AmountRuleBase):
    def __init__(self, rule_input: dict[str, Any], amount: float):
        self.amount = amount
        self.relative = True
        self.multiplicative = False
        self.offset_liquidity = False
        self.offset_pnl = False
        self.reason = MutationReason(rule="BalanceSheetMutationRule")
        self.date: datetime.date | None = None
        self.cohorts: list[Cohort] = []

        if is_in_identifiers("item", list(rule_input.keys())):
            value = rule_input[get_identifier("item", list(rule_input.keys()))]
            if pd.isna(value) or value == "":
                self.item = BalanceSheetItem()
            else:
                self.item = BalanceSheetItemRegistry.get(value)
        else:
            self.item = BalanceSheetItem()
        if is_in_identifiers("counter item", list(rule_input.keys())):
            value = rule_input[get_identifier("counter item", list(rule_input.keys()))]
            if pd.isna(value) or value == "":
                self.counter_item = None
            else:
                self.counter_item = BalanceSheetItemRegistry.get(value)
        else:
            self.counter_item = None

        for key, value in rule_input.items():
            # TODO: Use walrus operator
            match strip_identifier(key):
                case _ if value in ["", np.nan, None]:
                    pass
                case "item" | "counteritem":
                    pass
                case "metric":
                    self.metric = BalanceSheetMetrics.get(value)
                case _ if key.startswith("counter"):
                    label = strip_identifier(key[len("counter") :])
                    if label is not None and is_in_identifiers(label, Config.label_columns()):
                        if self.counter_item is None:
                            self.counter_item = BalanceSheetItem(**{label: value})
                        else:
                            self.counter_item = self.counter_item.add_identifier(label, value)
                    else:
                        raise KeyError(f"{key} not recognized as valid balance sheet label")
                case _ if is_in_identifiers(key, Config.label_columns()):
                    self.item = self.item.add_identifier(key, value)
                case _ if strip_identifier(key).startswith(("age", "minage", "maxage")):
                    cohort = Cohort.from_string(strip_identifier(key), read_int(value))
                    self.cohorts.append(cohort)
                case "relative":
                    self.relative = read_bool(value)
                case "multiplicative":
                    self.multiplicative = read_bool(value)
                case "offsetliquidity":
                    self.offset_liquidity = read_bool(value)
                case "offsetpnl":
                    self.offset_pnl = read_bool(value)
                case "date":
                    self.date = read_date(value)
                case _:
                    raise KeyError(f"{key} not recognized in BalanceSheetMutationRule")

    def apply(self, bs: BalanceSheet, increment: TimeIncrement, market_rates: MarketRates) -> BalanceSheet:
        # Implement the logic to apply the mutation to the balance sheet based on rule_input
        # This is a placeholder implementation

        item = self.item
        for cohort in self.cohorts:
            expr = cohort.get_expression(increment.to_date)
            item = item.add_condition(expr)

        if self.date is None or increment.contains(self.date):
            bs.mutate_metric(
                item,
                self.metric,
                self.amount,
                self.reason,
                self.relative,
                self.multiplicative,
                self.offset_liquidity,
                self.offset_pnl,
                self.counter_item,
            )

        bs.validate()

        return bs


class Cohort:
    def __init__(
        self, age: int, unit: Literal["days", "months", "years"], minimum: bool = False, maximum: bool = False
    ) -> None:
        self.age = age
        self.unit = unit
        self.minimum = minimum
        self.maximum = maximum
        assert not (minimum and maximum), "Cohort cannot be both minimum and maximum"

    @staticmethod
    def from_string(label: str, value: int) -> "Cohort":
        if label.startswith("minage"):
            minimum = True
            maximum = False
            unit = label[len("minage") :]
        elif label.startswith("maxage"):
            minimum = False
            maximum = True
            unit = label[len("maxage") :]
        elif label.startswith("age"):
            minimum = False
            maximum = False
            unit = label[len("age") :]
        else:
            raise ValueError(f"Cohort string '{label}' must start with 'age', 'minage', or 'maxage'")

        return Cohort(age=value, unit=unit, minimum=minimum, maximum=maximum)

    def get_expression(self, reference_date: datetime.date) -> pl.Expr:
        offset_date = reference_date - relativedelta(**{self.unit: self.age})

        if self.minimum:
            return pl.col("OriginationDate") >= pl.lit(offset_date)
        elif self.maximum:
            return pl.col("OriginationDate") <= pl.lit(offset_date)
        else:
            offset_date2 = reference_date - relativedelta(**{self.unit: self.age + 1})
            return (pl.col("OriginationDate") > pl.lit(offset_date2)) & (
                pl.col("OriginationDate") <= pl.lit(offset_date)
            )
