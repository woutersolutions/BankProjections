from dataclasses import dataclass, field
from typing import Any

import polars as pl

from src.bank_projections.financials.metrics import (
    BalanceSheetMetric,
    BalanceSheetMetrics,
)


@dataclass
class BalanceSheetItem:
    """
    Represents a balance sheet item defined by a set of identifiers.

    Used to filter and identify specific rows in balance sheet data based on
    column values (e.g., asset_type='loan', currency='EUR').
    """

    identifiers: dict[str, Any] = field(default_factory=dict)

    def __init__(self, **identifiers: Any) -> None:
        self.identifiers = identifiers

    def remove_identifier(self, identifier: str) -> "BalanceSheetItem":
        identifiers = self.identifiers.copy()
        del identifiers[identifier]
        return BalanceSheetItem(**identifiers)

    @property
    def filter_expression(self) -> pl.Expr:
        if not self.identifiers:
            return pl.lit(True)
        expr = pl.all_horizontal([pl.col(col) == val for col, val in self.identifiers.items()])
        return expr


class Positions:
    def __init__(self, data: pl.DataFrame) -> None:
        self._data = data
        self.validate()

    def validate(self) -> None:
        if len(self) == 0:
            raise ValueError("Positions data cannot be empty")

    def get_amount(self, item: BalanceSheetItem, metric: BalanceSheetMetric) -> float:
        result = self._data.filter(item.filter_expression).select(metric.aggregation_expression).item()
        return float(result)

    def set_amount(
        self,
        item: BalanceSheetItem,
        metric: BalanceSheetMetric,
        amount: float,
        relative: bool = False,
    ) -> None:
        if relative:
            expr = metric.mutation_expression(amount, item.filter_expression) + pl.col(metric.mutation_column)
        else:
            expr = metric.mutation_expression(amount, item.filter_expression)

        self._data = self._data.with_columns(
            pl.when(item.filter_expression)
            .then(expr)
            .otherwise(pl.col(metric.mutation_column))
            .alias(metric.mutation_column)
        )

    def __len__(self) -> int:
        return len(self._data)

    @classmethod
    def combine(cls, *position_sets: "Positions") -> "Positions":
        """Combine multiple position sets into a single Positions object."""
        if not position_sets:
            raise ValueError("At least one position set must be provided")

        combined_data = pl.concat([pos._data for pos in position_sets], how="vertical")
        return cls(combined_data)


class BalanceSheet(Positions):
    def validate(self) -> None:
        super().validate()

        total_book_value = self.get_amount(BalanceSheetItem(), BalanceSheetMetrics.book_value)
        if abs(total_book_value) > 0.01:
            raise ValueError(
                f"Balance sheet does not balance: total book value is {total_book_value:.4f}, "
                f"expected 0.00 (assets should equal funding within 0.01 tolerance)"
            )
