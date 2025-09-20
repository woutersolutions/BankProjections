import copy
import datetime
from dataclasses import dataclass
from typing import Any

import pandas as pd
import polars as pl

from bank_projections.config import (
    BALANCE_SHEET_AGGREGATION_LABELS,
    CASHFLOW_AGGREGATION_LABELS,
    PNL_AGGREGATION_LABELS,
)
from bank_projections.financials.balance_sheet_item import BalanceSheetItem, BalanceSheetItemRegistry
from bank_projections.financials.metrics import (
    BalanceSheetMetric,
    BalanceSheetMetrics,
)
from bank_projections.projections.redemption import RedemptionRegistry


@dataclass
class MutationReason:
    def __init__(self, **kwargs: Any) -> None:
        self.reasons = kwargs

    def add_to_df(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(**{k: pl.lit(v) for k, v in self.reasons.items()})

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.reasons.items())))


class Positions:
    def __init__(self, data: pl.DataFrame):
        self._data = data

    def validate(self) -> None:
        if len(self) == 0:
            raise ValueError("Positions data cannot be empty")

        if not self._data.select(RedemptionRegistry.required_columns_validation().all()).item():
            raise ValueError("Positions data is missing required columns for registered redemption types")

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
    def __init__(self, data: pl.DataFrame, date: datetime.date):
        super().__init__(data)
        self.add_item(BalanceSheetItemRegistry.get("cash account"), OriginationDate=date)
        self.add_item(BalanceSheetItemRegistry.get("pnl account"), OriginationDate=date)

        self.cashflows = pl.DataFrame(schema={"Amount": pl.Float64})
        self.pnls = pl.DataFrame(schema={"Amount": pl.Float64})
        self.date = date

        self.validate()

    def initialize_new_date(self, date: datetime.date) -> "BalanceSheet":
        return BalanceSheet(self._data, date)

    def validate(self) -> None:
        super().validate()

        total_book_value = self.get_amount(BalanceSheetItem(), BalanceSheetMetrics.get("book_value"))
        if abs(total_book_value) > 0.01:
            raise ValueError(
                f"Balance sheet does not balance: total book value is {total_book_value:.4f}, "
                f"expected 0.00 (assets should equal funding within 0.01 tolerance)"
            )

        if self._data["Quantity"].is_null().any():
            raise ValueError("Quantity column contains null values after adding new item.")

        # Check total pnl in balance sheet and pnl table are the same
        total_pnl_bs = self.get_amount(
            BalanceSheetItemRegistry.get("pnl account").add_identifier("OriginationDate", self.date),
            BalanceSheetMetrics.get("book_value"),
        )
        total_pnl_table = self.pnls["Amount"].sum() if len(self.pnls) > 0 else 0.0
        if abs(total_pnl_bs + total_pnl_table) > 0.01:
            raise ValueError(
                f"PnL in balance sheet and PnL table do not match: {total_pnl_bs:.4f} vs {-total_pnl_table:.4f}"
            )

        # Check total cash in balance sheet and cashflow table are the same
        total_cash_bs = self.get_amount(
            BalanceSheetItemRegistry.get("cash account").add_identifier("OriginationDate", self.date),
            BalanceSheetMetrics.get("book_value"),
        )
        total_cash_table = self.cashflows["Amount"].sum() if len(self.cashflows) > 0 else 0.0
        if abs(total_cash_bs - total_cash_table) > 0.01:
            raise ValueError(
                f"Cash in balance sheet and cashflow table do not match: {total_cash_bs:.4f} vs {total_cash_table:.4f}"
            )

    def add_item(
        self,
        based_on_item: BalanceSheetItem,
        quantity: float = 0.0,
        accrued_interest: float = 0.0,
        impairment: float = 0.0,
        dirty_price: float = 0.0,
        **new_values: Any,
    ) -> None:
        # Find the number of rows and total quantity of the based_on_item
        based_on_quantity, based_on_count = (
            self._data.filter(based_on_item.filter_expression).select(
                [pl.col("Quantity").sum().alias("total"), pl.col("Quantity").count().alias("count")]
            )
        ).row(0)
        if based_on_count == 0:
            raise ValueError(f"No item found on balance sheet matching: {based_on_item}")
        if based_on_quantity == 0:
            raise ValueError(f"Cannot base new item on zero quantity item: {based_on_item}")

        n_rows = len(self._data.filter(based_on_item.filter_expression))

        # Find unique labels for non-numeric columns
        non_numeric_cols = self._data.select([pl.col(pl.Utf8), pl.col(pl.Boolean)]).columns
        n_uniques = self._data.filter(based_on_item.filter_expression).select(
            [pl.col(col).n_unique().alias(col) for col in non_numeric_cols]
        )
        constant_cols = [c for c in non_numeric_cols if n_uniques[0, c] == 1]

        new_row = (
            self._data.filter(based_on_item.filter_expression)
            .select(
                [pl.col(col).first().alias(col) for col in constant_cols]
                + [
                    metric.aggregation_expression.alias(metric.column)
                    for name, metric in BalanceSheetMetrics.items.items()
                    if metric.is_stored
                ],
            )
            .with_columns(
                [pl.lit(value).alias(column) for column, value in new_values.items()],
                Quantity=quantity,
                AccruedInterest=accrued_interest,
                Impairment=impairment,
                CleanPrice=dirty_price,
            )
        )

        # Check missing columns #TODO: Properly handle missing columns
        missing_cols = set(self._data.columns) - set(new_row.columns)

        self._data = pl.concat([self._data, new_row], how="diagonal")

    def mutate_metric(
        self,
        item: BalanceSheetItem,
        metric: BalanceSheetMetric,
        amount: float,
        reason: MutationReason | None = None,
        relative: bool = False,
        multiplicative: bool = False,
        offset_liquidity: bool = False,
        offset_pnl: bool = False,
        counter_item: BalanceSheetItem | None = None,
    ) -> None:
        if multiplicative:
            if relative:
                expr = pl.col(metric.mutation_column) * (1 + amount)
            else:
                expr = pl.col(metric.mutation_column) * amount
        else:
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
            counter_item=counter_item,
            **{metric.mutation_column: expr},
        )

    def mutate(
        self,
        item: BalanceSheetItem,
        pnls: dict[MutationReason, pl.Expr] | None = None,
        cashflows: dict[MutationReason, pl.Expr] | None = None,
        offset_pnl: MutationReason | None = None,
        offset_liquidity: MutationReason | None = None,
        counter_item: BalanceSheetItem | None = None,
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

        number_of_offsets = sum([offset_pnl is not None, offset_liquidity is not None, counter_item is not None])
        if number_of_offsets > 1:
            raise ValueError("Can offset with 1 thing only (pnl, cash, or counter balance sheet item")

        if number_of_offsets > 0:
            calculations["BookValueBefore"] = BalanceSheetMetrics.get("book_value").get_expression.alias(
                "BookValueBefore"
            )

        if self._data.filter(item.filter_expression).is_empty():
            raise ValueError(f"No item found on balance sheet matching: {item}")

        self._data = self._data.with_columns(**calculations)

        # Process PnL mutations
        if pnls is not None:
            for i, (mut_reason, _) in enumerate(pnls.items()):
                pnl_col = f"pnl_{i}"
                self.add_pnl(self._data.filter(item.filter_expression), pl.col(pnl_col), mut_reason)
                self._data = self._data.drop(pnl_col)

        # Process cashflow mutations
        if cashflows is not None:
            for i, (mut_reason, _) in enumerate(cashflows.items()):
                cashflow_col = f"cashflow_{i}"
                self.add_liquidity(self._data.filter(item.filter_expression), pl.col(cashflow_col), mut_reason)
                self._data = self._data.drop(cashflow_col)

        if offset_pnl is not None:
            self.add_pnl(
                self._data.filter(item.filter_expression),
                BalanceSheetMetrics.get("book_value").get_expression - pl.col("BookValueBefore"),
                offset_pnl,
            )
        if offset_liquidity is not None:
            self.add_liquidity(
                self._data.filter(item.filter_expression),
                -(BalanceSheetMetrics.get("book_value").get_expression - pl.col("BookValueBefore")),
                offset_liquidity,
            )

        if counter_item is not None:
            book_value_change = (
                self._data.filter(item.filter_expression)
                .select((BalanceSheetMetrics.get("book_value").get_expression - pl.col("BookValueBefore")).sum())
                .item()
            )

            self.mutate_metric(counter_item, BalanceSheetMetrics.get("quantity"), -book_value_change, relative=True)

        if number_of_offsets > 0:
            self._data = self._data.drop("BookValueBefore")

    def add_pnl(self, data: pl.DataFrame, expr: pl.Expr, reason: MutationReason) -> None:
        pnls = data.group_by(PNL_AGGREGATION_LABELS).agg(Amount=expr.sum()).pipe(reason.add_to_df)

        self.pnls = pl.concat([self.pnls, pnls], how="diagonal")
        self.mutate_metric(
            BalanceSheetItemRegistry.get("pnl account").add_identifier("OriginationDate", self.date),
            BalanceSheetMetrics.get("quantity"),
            -pnls["Amount"].sum(),
            reason,
            relative=True,
        )

    def add_single_pnl(self, amount: float, reason: MutationReason, offset_liquidity: bool = False) -> None:
        pnls = pl.DataFrame({"Amount": [amount]}).pipe(reason.add_to_df)

        self.pnls = pl.concat([self.pnls, pnls], how="diagonal")
        self.mutate_metric(
            BalanceSheetItemRegistry.get("pnl account").add_identifier("OriginationDate", self.date),
            BalanceSheetMetrics.get("quantity"),
            -amount,
            reason,
            relative=True,
        )

        if offset_liquidity:
            self.add_single_liquidity(amount, reason)

    def add_liquidity(self, data: pl.DataFrame, expr: pl.Expr, reason: MutationReason) -> None:
        cashflows = data.group_by(CASHFLOW_AGGREGATION_LABELS).agg(Amount=expr.sum()).pipe(reason.add_to_df)

        self.cashflows = pl.concat([self.cashflows, cashflows], how="diagonal")
        self.mutate_metric(
            BalanceSheetItemRegistry.get("cash account").add_identifier("OriginationDate", self.date),
            BalanceSheetMetrics.get("quantity"),
            cashflows["Amount"].sum(),
            reason,
            relative=True,
        )

    def add_single_liquidity(self, amount: float, reason: MutationReason, offset_pnl: bool = False) -> None:
        cashflows = pl.DataFrame({"Amount": [amount]}).pipe(reason.add_to_df)

        self.cashflows = pl.concat([self.cashflows, cashflows], how="diagonal")
        self.mutate_metric(
            BalanceSheetItemRegistry.get("cash account").add_identifier("OriginationDate", self.date),
            BalanceSheetMetrics.get("quantity"),
            amount,
            reason,
            relative=True,
        )

        if offset_pnl:
            self.add_single_pnl(-amount, reason)

    def copy(self) -> "BalanceSheet":
        return copy.deepcopy(self)

    def aggregate(
        self, group_columns: list[str] = BALANCE_SHEET_AGGREGATION_LABELS
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        return (
            (
                self._data.group_by(group_columns)
                .agg([metric.aggregation_expression.alias(name) for name, metric in BalanceSheetMetrics.items.items()])
                .sort(by=group_columns)
            ),
            self.pnls,
            self.cashflows,
        )

    @classmethod
    def get_differences(cls, bs1: "BalanceSheet", bs2: "BalanceSheet") -> pl.DataFrame:
        numeric_cols = [c for c, dt in zip(bs1._data.columns, bs2._data.dtypes, strict=False) if dt.is_numeric()]

        # Compute differences only on numeric cols
        diff_df = bs1._data.select([(pl.col(c) - bs2._data[c]).alias(f"Delta_{c}") for c in numeric_cols])

        return diff_df

    @classmethod
    def debug(cls, bs1: "BalanceSheet", bs2: "BalanceSheet") -> dict[str, pd.DataFrame]:
        return {
            "bs1": bs1._data.to_pandas(),
            "bs2": bs2._data.to_pandas(),
            "diff": cls.get_differences(bs1, bs2).to_pandas(),
            "pnl": bs2.pnls.to_pandas(),
            "cash": bs2.cashflows.to_pandas(),
        }
