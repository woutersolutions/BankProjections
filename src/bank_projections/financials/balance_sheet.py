from dataclasses import dataclass, field
from typing import Any

import polars as pl

from src.bank_projections.config import CASHFLOW_AGGREGATION_LABELS
from src.bank_projections.financials.metrics import (
    BalanceSheetMetric,
    BalanceSheetMetrics,
)


@dataclass
class BalanceSheetItem:
    identifiers: dict[str, Any] = field(default_factory=dict)

    def __init__(self, **identifiers: Any) -> None:
        self.identifiers = identifiers

    def add_identifier(self, key: str, value: Any) -> "BalanceSheetItem":
        identifiers = self.identifiers.copy()
        identifiers[key] = value
        return BalanceSheetItem(**identifiers)

    def remove_identifier(self, identifier: str) -> "BalanceSheetItem":
        identifiers = self.identifiers.copy()
        del identifiers[identifier]
        return BalanceSheetItem(**identifiers)

    def copy(self):
        return BalanceSheetItem(**self.identifiers.copy())

    @property
    def filter_expression(self) -> pl.Expr:
        if not self.identifiers:
            return pl.lit(True)
        expr = pl.all_horizontal([pl.col(col) == val for col, val in self.identifiers.items()])
        return expr


class Positions:
    def __init__(self, data: pl.DataFrame):
        self._data = data

    def validate(self) -> None:
        if len(self) == 0:
            raise ValueError("Positions data cannot be empty")

    def __len__(self) -> int:
        return len(self._data)

    def get_amount(self, item: BalanceSheetItem, metric: BalanceSheetMetric) -> float:
        result = self._data.filter(item.filter_expression).select(metric.aggregation_expression).item()
        return float(result)

    @staticmethod
    def combine(*positions: "Positions") -> "Positions":
        if len(positions) < 1:
            raise ValueError("At least one position is required")

        # Concatenate all position data
        combined_data = pl.concat([pos._data for pos in positions])

        return Positions(combined_data)


class BalanceSheet(Positions):
    def __init__(self, data: pl.DataFrame, cash_account: BalanceSheetItem, pnl_account: BalanceSheetItem):
        super().__init__(data)
        self.cash_account = cash_account
        self.pnl_account = pnl_account

        self.cashflows = pl.DataFrame()
        self.pnls = pl.DataFrame()

        self.validate()

    def validate(self) -> None:
        super().validate()

        total_book_value = self.get_amount(BalanceSheetItem(), BalanceSheetMetrics.book_value)
        if abs(total_book_value) > 0.01:
            raise ValueError(
                f"Balance sheet does not balance: total book value is {total_book_value:.4f}, "
                f"expected 0.00 (assets should equal funding within 0.01 tolerance)"
            )

    def mutate_metric(
        self,
        item: BalanceSheetItem,
        metric: BalanceSheetMetric,
        amount: float,
        relative: bool = False,
        offset_liquidity: bool = False,
        offset_pnl: bool = False,
    ) -> None:
        if relative:
            expr = metric.mutation_expression(amount, item.filter_expression) + pl.col(metric.mutation_column)
        else:
            expr = metric.mutation_expression(amount, item.filter_expression)

        new_data = self._data.with_columns(
            pl.when(item.filter_expression)
            .then(expr)
            .otherwise(pl.col(metric.mutation_column))
            .alias(metric.mutation_column),
            BalanceSheetMetrics.book_value.get_expression.alias("BookValueBefore"),
        ).with_columns(
            (BalanceSheetMetrics.book_value.get_expression - pl.col("BookValueBefore")).alias("BookValueImpact")
        )

        if offset_liquidity and offset_pnl:
            raise ValueError("Cannot offset with both cash and pnl")
        if offset_liquidity:
            cashflows = (
                new_data.filter(item.filter_expression)
                .group_by(CASHFLOW_AGGREGATION_LABELS)
                .agg(Amount=-pl.col("BookValueImpact").sum())
            )
            self.cashflows = pl.concat([self.cashflows, cashflows])
        if offset_pnl:
            pnls = (
                new_data.filter(item.filter_expression)
                .group_by(CASHFLOW_AGGREGATION_LABELS)
                .agg(Amount=pl.col("BookValueImpact").sum())
            )
            self.pnls = pl.concat([self.pnls, pnls])

        # Update the balance sheet data with the mutations
        self._data = new_data.drop("BookValueBefore", "BookValueImpact")

    def add_pnl(self, amount: float):
        # TODO: Add origination date
        self.mutate_metric(self.pnl_account, BalanceSheetMetrics.quantity, amount, True)

    def add_liquidity(self, amount: float):
        self.mutate_metric(self.cash_account, BalanceSheetMetrics.quantity, amount, True)

    def copy(self):
        return BalanceSheet(
            self._data.clone(), cash_account=self.cash_account.copy(), pnl_account=self.pnl_account.copy()
        )
