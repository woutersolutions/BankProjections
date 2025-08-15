from typing import Any

import polars as pl

from src.bank_projections.financials.metrics import (
    BalanceSheetMetric,
    BalanceSheetMetrics,
)


class BalanceSheetItem:

    def __init__(self, **identifiers: Any) -> None:
        self.identifiers = identifiers

    def remove_identifier(self, identifier: str):

        identifiers = self.identifiers.copy()
        del identifiers[identifier]
        return BalanceSheetItem(**identifiers)

    @property
    def filter_expression(self):
        expr = pl.all_horizontal(
            [pl.col(col) == val for col, val in self.identifiers.items()]
        )
        return expr


class Positions:

    def __init__(self, data: pl.DataFrame) -> None:

        self._data = data
        self.validate()

    def validate(self):
        assert len(self) > 0

    def get_amount(self, item: BalanceSheetItem, metric: BalanceSheetMetric) -> float:

        return (
            self._data.filter(item.filter_expression)
            .select(metric.aggregation_expression)
            .item()
        )

    def set_amount(
        self,
        item: BalanceSheetItem,
        metric: BalanceSheetMetric,
        amount: float,
        relative: bool = False,
    ) -> None:

        mutation_amounts = self._data.select(
            amount * metric.mutation_expression()
        ).item()
        if relative:
            expr = pl.col(metric.mutation_column) + mutation_amounts
        else:
            expr = mutation_amounts

        self._data = self._data.with_columns(
            pl.when(item.filter_expression, expr)
            .then(expr)
            .otherwise(pl.col(metric.mutation_column))
            .alias(metric.mutation_column)
        )

    def __len__(self) -> int:
        return len(self._data)


class BalanceSheet(Positions):

    def validate(self):
        super().validate()

        total_book_value = self.get_amount(
            BalanceSheetItem(), BalanceSheetMetrics.book_value
        )
        if abs(total_book_value) > 0.01:
            raise ValueError(
                f"Total book value is not zero, so assets do not equal funding"
            )
