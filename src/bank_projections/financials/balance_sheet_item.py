from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import polars as pl

from bank_projections.config import Config
from bank_projections.projections.base_registry import BaseRegistry
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
        elif is_in_identifiers(key, Config.BALANCE_SHEET_LABELS):
            key = get_identifier(key, Config.BALANCE_SHEET_LABELS)
        elif is_in_identifiers(key, Config.DATE_COLUMNS):
            key = get_identifier(key, Config.DATE_COLUMNS)
            value = read_date(value)
        elif is_in_identifiers(key, Config.CLASSIFICATIONS):
            key = get_identifier(key, Config.CLASSIFICATIONS)
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


class BalanceSheetItemRegistry(BaseRegistry[BalanceSheetItem]):
    pass


BalanceSheetItemRegistry.register("cash account", BalanceSheetItem(ItemType="Cash"))
BalanceSheetItemRegistry.register("pnl account", BalanceSheetItem(ItemType="Unaudited earnings"))
BalanceSheetItemRegistry.register("retained earnings", BalanceSheetItem(SubItemType="Retained earnings"))
BalanceSheetItemRegistry.register("dividend", BalanceSheetItem(ItemType="Dividends payable"))
BalanceSheetItemRegistry.register("oci", BalanceSheetItem(SubItemType="Other comprehensive income"))
