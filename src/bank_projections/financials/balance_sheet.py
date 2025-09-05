from dataclasses import dataclass, field
from typing import Any, Optional

import polars as pl

from bank_projections.config import CASHFLOW_AGGREGATION_LABELS, PNL_AGGREGATION_LABELS
from bank_projections.financials.metrics import (
    BalanceSheetMetric,
    BalanceSheetMetrics,
)


@dataclass
class MutationReason:
    def __init__(self, **kwargs: Any) -> None:
        self.reasons = kwargs

    def add_to_df(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(**{k: pl.lit(v) for k, v in self.reasons.items()})

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.reasons.items())))


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
        reason: MutationReason,
        relative: bool = False,
        offset_liquidity: bool = False,
        offset_pnl: bool = False,
    ) -> None:
        if relative:
            expr = metric.mutation_expression(amount, item.filter_expression) + pl.col(metric.mutation_column)
        else:
            expr = metric.mutation_expression(amount, item.filter_expression)

        offset_pnl_reason = reason if offset_pnl else None
        offset_liquidity_reason = reason if offset_liquidity else None
        self.mutate(
            item,
            offset_pnl=offset_pnl_reason,
            offset_liquidity=offset_liquidity_reason,
            **{metric.mutation_column: expr},
        )

    def mutate(
        self,
        item: BalanceSheetItem,
        pnls: Optional[dict[MutationReason, pl.Expr]] = None,
        cashflows: Optional[dict[MutationReason, pl.Expr]] = None,
        offset_pnl: Optional[MutationReason] = None,
        offset_liquidity: Optional[MutationReason] = None,
        **exprs: pl.Expr,
    ) -> None:
        # Assert all exprs keys are valid columns
        valid_columns = set(self._data.columns)
        for k in exprs:
            if k not in valid_columns:
                raise ValueError(f"Invalid column '{k}' for mutation. Valid columns are: {valid_columns}")

        calculations = {k: pl.when(item.filter_expression).then(v).otherwise(pl.col(k)) for k, v in exprs.items()}

        if pnls is not None:
            for i, (_mut_reason, pnl_expr) in enumerate(pnls.items()):
                pnl_col = f"pnl_{i}"
                calculations[pnl_col] = pl.when(item.filter_expression).then(pnl_expr).otherwise(0.0).alias(pnl_col)

        if cashflows is not None:
            for i, (_mut_reason, cashflow_expr) in enumerate(cashflows.items()):
                cashflow_col = f"cashflow_{i}"
                calculations[cashflow_col] = (
                    pl.when(item.filter_expression).then(cashflow_expr).otherwise(0.0).alias(cashflow_col)
                )

        if offset_liquidity is not None or offset_pnl is not None:
            calculations["BookValueBefore"] = BalanceSheetMetrics.book_value.get_expression.alias("BookValueBefore")

        self._data = self._data.with_columns(**calculations)

        # Process PnL mutations
        if pnls is not None:
            for i, (mut_reason, _) in enumerate(pnls.items()):
                pnl_col = f"pnl_{i}"
                self.add_pnl(self._data.filter(item.filter_expression), pl.col(pnl_col), mut_reason)
                self._data.drop(pnl_col)

        # Process cashflow mutations
        if cashflows is not None:
            for i, (mut_reason, _) in enumerate(cashflows.items()):
                cashflow_col = f"cashflow_{i}"
                self.add_liquidity(self._data.filter(item.filter_expression), pl.col(cashflow_col), mut_reason)
                self._data.drop(cashflow_col)

        if offset_pnl is not None and offset_liquidity is not None:
            raise ValueError("Cannot offset with both cash and pnl")
        if offset_pnl is not None:
            self.add_pnl(
                self._data.filter(item.filter_expression),
                BalanceSheetMetrics.book_value.get_expression - pl.col("BookValueBefore"),
                offset_pnl,
            )
            self._data = self._data.drop("BookValueBefore")
        if offset_liquidity is not None:
            self.add_liquidity(
                self._data.filter(item.filter_expression),
                -(BalanceSheetMetrics.book_value.get_expression - pl.col("BookValueBefore")),
                offset_liquidity,
            )
            self._data = self._data.drop("BookValueBefore")

    def add_pnl(self, data: pl.DataFrame, expr: pl.Expr, reason: MutationReason):
        pnls = data.group_by(PNL_AGGREGATION_LABELS).agg(Amount=expr.sum()).pipe(reason.add_to_df)

        self.pnls = pl.concat([self.pnls, pnls])
        self.mutate_metric(self.pnl_account, BalanceSheetMetrics.quantity, -pnls["Amount"].sum(), reason, relative=True)

    def add_liquidity(self, data: pl.DataFrame, expr: pl.Expr, reason: MutationReason):
        cashflows = data.group_by(CASHFLOW_AGGREGATION_LABELS).agg(Amount=expr.sum()).pipe(reason.add_to_df)

        self.cashflows = pl.concat([self.cashflows, cashflows])
        self.mutate_metric(
            self.cash_account, BalanceSheetMetrics.quantity, cashflows["Amount"].sum(), reason, relative=True
        )

    def copy(self):
        return BalanceSheet(
            self._data.clone(), cash_account=self.cash_account.copy(), pnl_account=self.pnl_account.copy()
        )
