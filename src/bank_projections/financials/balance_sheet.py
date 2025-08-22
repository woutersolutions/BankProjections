from dataclasses import dataclass, field
from typing import Any

import polars as pl

from src.bank_projections.config import MUTATION_AGGREGATION_LABELS
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

        self.validate()

    def validate(self) -> None:
        super().validate()

        total_book_value = self.get_amount(BalanceSheetItem(), BalanceSheetMetrics.book_value)
        if abs(total_book_value) > 0.01:
            raise ValueError(
                f"Balance sheet does not balance: total book value is {total_book_value:.4f}, "
                f"expected 0.00 (assets should equal funding within 0.01 tolerance)"
            )

    def mutate(
        self,
        item: BalanceSheetItem,
        metric: BalanceSheetMetric,
        amount: float,
        relative: bool = False,
        offset_liquidity: bool = False,
        offset_pnl: bool = False,
    ) -> pl.DataFrame:
        if offset_liquidity and offset_pnl:
            raise ValueError("Cannot offset with both cash and pnl")

        if relative:
            expr = metric.mutation_expression(amount, item.filter_expression) + pl.col(metric.mutation_column)
        else:
            expr = metric.mutation_expression(amount, item.filter_expression)

        new_data = self._data.with_columns(
            pl.when(item.filter_expression)
            .then(expr)
            .otherwise(pl.col(metric.mutation_column))
            .alias(metric.mutation_column)
        )

        # Calculate book value impact
        impact_metrics = {"Quantity": BalanceSheetMetrics.quantity, "BookValue": BalanceSheetMetrics.book_value}

        def agg_bs(df: pl.DataFrame, suffix: str) -> pl.DataFrame:
            return (
                df.filter(item.filter_expression)
                .group_by(MUTATION_AGGREGATION_LABELS)
                .agg(*[metric.aggregation_expression.alias(name + suffix) for name, metric in impact_metrics.items()])
            )

        diffs = (
            agg_bs(new_data, "New")
            .join(agg_bs(self._data, "Old"), on=MUTATION_AGGREGATION_LABELS, how="full", coalesce=True)
            .fill_null(0.0)
        )

        mutations = diffs.with_columns(
            *[(pl.col(name + "New") - pl.col(name + "Old")).alias(name) for name in impact_metrics],
            -(pl.col("BookValueNew") - pl.col("BookValueOld")).alias("Liquidity")
            if offset_liquidity
            else pl.lit(0.0).alias("Liquidity"),
            (pl.col("BookValueNew") - pl.col("BookValueOld")).alias("PnL") if offset_pnl else pl.lit(0.0).alias("PnL"),
        ).select(MUTATION_AGGREGATION_LABELS + list(impact_metrics.keys()) + ["Liquidity", "PnL"])

        # Update the balance sheet data with the mutations
        self._data = new_data

        if offset_pnl or offset_liquidity:
            # Calculate the total book value change for offsetting
            total_book_value_change = mutations.select(pl.col("BookValue").sum()).item()

            if offset_pnl:
                self.add_pnl(total_book_value_change)
            if offset_liquidity:
                self.add_liquidity(-total_book_value_change)

        return mutations

    def add_pnl(self, amount: float):
        # TODO: Add origination date
        self.mutate(self.pnl_account, BalanceSheetMetrics.quantity, amount, True)

    def add_liquidity(self, amount: float):
        self.mutate(self.cash_account, BalanceSheetMetrics.quantity, amount, True)

    def copy(self):
        return BalanceSheet(
            self._data.clone(), cash_account=self.cash_account.copy(), pnl_account=self.pnl_account.copy()
        )
