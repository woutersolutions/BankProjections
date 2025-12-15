import datetime
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd
import polars as pl
from dateutil.relativedelta import relativedelta

from bank_projections.app_config import Config
from bank_projections.financials.balance_sheet_category import BalanceSheetCategoryRegistry
from bank_projections.utils.base_registry import BaseRegistry
from bank_projections.utils.parsing import get_identifier, is_in_identifiers, read_date, strip_identifier


@dataclass
class BalanceSheetItem:
    identifiers: dict[str, Any] = field(default_factory=dict)

    def __init__(self, expr: pl.Expr | None = None, **identifiers: Any) -> None:
        self.identifiers = {}
        for key, value in identifiers.items():
            self._add_identifier(self.identifiers, key, value)

        self.expr = expr

    @staticmethod
    def _add_identifier(identifiers: dict[str, Any], key: str, value: Any) -> None:
        if pd.isna(value) or value == "":
            raise ValueError(f"BalanceSheetItem {key} cannot be '{value}'")
        elif is_in_identifiers(key, Config.balance_sheet_labels()):
            key = get_identifier(key, Config.balance_sheet_labels())
        elif is_in_identifiers(key, Config.date_columns()):
            key = get_identifier(key, Config.date_columns())
            value = read_date(value)
        elif is_in_identifiers(key, Config.get_classifications()):
            key = get_identifier(key, Config.get_classifications())
            value = strip_identifier(value)
        else:
            raise ValueError(
                f"Invalid identifier '{key}' for BalanceSheetItem. Valid identifiers are: {Config.label_columns()}"
            )
        identifiers[key] = value

    def add_identifier(self, key: str, value: Any) -> "BalanceSheetItem":
        identifiers = self.identifiers.copy()
        self._add_identifier(identifiers, key, value)
        return BalanceSheetItem(expr=self.expr, **identifiers)

    def add_condition(self, expr: pl.Expr) -> "BalanceSheetItem":
        new_expr = expr if self.expr is None else self.expr & expr
        return BalanceSheetItem(expr=new_expr, **self.identifiers)

    def remove_identifier(self, identifier: str) -> "BalanceSheetItem":
        identifiers = self.identifiers.copy()
        del identifiers[identifier]
        return BalanceSheetItem(**identifiers)

    def copy(self) -> "BalanceSheetItem":
        return BalanceSheetItem(**self.identifiers.copy())

    @property
    def filter_expression(self) -> pl.Expr:
        expr = pl.all_horizontal(
            ([pl.lit(True)] if self.expr is None else [self.expr])
            + [pl.col(col) == val for col, val in self.identifiers.items()]
        )
        return expr

    def __and__(self, other: "BalanceSheetItem") -> "BalanceSheetItem":
        # Check for conflicting identifiers
        for key in self.identifiers.keys() & other.identifiers.keys():
            if self.identifiers[key] != other.identifiers[key]:
                raise ValueError(f"Conflicting identifiers for BalanceSheetItem: {key}")

        # Combine identifiers
        combined_identifiers = {**self.identifiers, **other.identifiers}
        if self.expr is None and other.expr is None:
            combined_expr = None
        elif self.expr is None:
            combined_expr = other.expr
        elif other.expr is None:
            combined_expr = self.expr
        else:
            combined_expr = self.expr & other.expr

        return BalanceSheetItem(expr=combined_expr, **combined_identifiers)

    def __or__(self, other: "BalanceSheetItem") -> "BalanceSheetItem":
        return BalanceSheetItem(expr=self.filter_expression | other.filter_expression)


class BalanceSheetItemRegistry(BaseRegistry[BalanceSheetItem]):
    pass


BalanceSheetItemRegistry.register("cash account", BalanceSheetItem(ItemType="Cash"))
BalanceSheetItemRegistry.register("pnl account", BalanceSheetItem(ItemType="Unaudited earnings"))
BalanceSheetItemRegistry.register("retained earnings", BalanceSheetItem(SubItemType="Retained earnings"))
BalanceSheetItemRegistry.register("dividend", BalanceSheetItem(ItemType="Dividends payable"))
BalanceSheetItemRegistry.register("oci", BalanceSheetItem(SubItemType="Other comprehensive income"))

BalanceSheetItemRegistry.register("derivatives", BalanceSheetItem(BalanceSheetCategory="Derivatives"))
BalanceSheetItemRegistry.register(
    "assets",
    BalanceSheetItem(expr=BalanceSheetCategoryRegistry.is_asset_side_expr()),
)
BalanceSheetItemRegistry.register(
    "liabilities",
    BalanceSheetItem(
        expr=~BalanceSheetCategoryRegistry.is_asset_side_expr() & ~pl.col("BalanceSheetCategory").eq("Equity")
    ),
)
BalanceSheetItemRegistry.register("equity", BalanceSheetItem(BalanceSheetCategory="Equity"))
BalanceSheetItemRegistry.register(
    "funding",
    BalanceSheetItem(expr=~BalanceSheetCategoryRegistry.is_asset_side_expr()),
)


class Cohort:
    def __init__(
        self, age: int, unit: Literal["days", "months", "years"], minimum: bool = False, maximum: bool = False
    ) -> None:
        self.age = age
        self.unit = unit
        self.minimum = minimum
        self.maximum = maximum
        assert not (minimum and maximum), "Cohort cannot be both minimum and maximum"
        # Validate unit is one of the expected values
        if unit not in ("days", "months", "years"):
            raise ValueError(f"Unit '{unit}' must be 'days', 'months', or 'years'")

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
