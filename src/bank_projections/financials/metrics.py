from abc import ABC, abstractmethod

import polars as pl

from bank_projections.projections.base_registry import BaseRegistry

SMALL_NUMBER = 1e-12


class BalanceSheetMetric(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def get_expression(self) -> pl.Expr:
        pass

    @abstractmethod
    def set_expression(self, amounts: pl.Expr) -> pl.Expr:
        pass

    @property
    @abstractmethod
    def aggregation_expression(self) -> pl.Expr:
        pass

    @abstractmethod
    def mutation_expression(self, amount: float, filter_expression: pl.Expr) -> pl.Expr:
        pass

    @property
    @abstractmethod
    def mutation_column(self) -> str:
        pass

    @property
    @abstractmethod
    def is_stored(self) -> bool:
        pass


class StoredColumn(BalanceSheetMetric, ABC):
    is_stored = True

    def __init__(self, column: str):
        self.column = column

    @property
    def name(self) -> str:
        return self.column

    @property
    def get_expression(self) -> pl.Expr:
        return pl.col(self.column)

    def set_expression(self, amounts: pl.Expr) -> pl.Expr:
        return pl.lit(amounts)

    @property
    def mutation_column(self) -> str:
        return self.column


class StoredAmount(StoredColumn):
    def __init__(self, column: str, allocation_col: str = "Quantity"):
        super().__init__(column)
        self.allocation_expr = pl.col(allocation_col) + pl.lit(SMALL_NUMBER)  # Prevent division by zero

    @property
    def aggregation_expression(self) -> pl.Expr:
        return self.get_expression.sum()

    def mutation_expression(self, amount: float, filter_expression: pl.Expr) -> pl.Expr:
        return pl.lit(amount) * self.allocation_expr / (filter_expression * self.allocation_expr).sum()


class StoredWeight(StoredColumn):
    def __init__(self, column: str, weight_expr: pl.Expr = pl.col("Quantity")):
        super().__init__(column)
        self.weight_expr = weight_expr + pl.lit(SMALL_NUMBER)  # Prevent division by zero

    @property
    def aggregation_expression(self) -> pl.Expr:
        return (pl.col(self.column) * self.weight_expr).sum() / self.weight_expr.sum()

    def mutation_expression(self, amount: float, filter_expression: pl.Expr) -> pl.Expr:
        return pl.lit(amount)


class DerivedMetric(BalanceSheetMetric, ABC):
    is_stored: bool = False

    def set_expression(self, amounts: pl.Expr) -> pl.Expr:
        raise NotImplementedError("Derived metric cannot be modified")

    def mutation_expression(self, amount: float, filter_expression: pl.Expr) -> pl.Expr:
        raise NotImplementedError("Derived metric cannot be modified")

    @property
    def mutation_column(self) -> str:
        raise NotImplementedError("Derived metric cannot be modified")


class DirtyPrice(DerivedMetric):
    name = "DirtyPrice"

    @property
    def get_expression(self) -> pl.Expr:
        return pl.col("CleanPrice") + pl.col("AccruedInterest") / (pl.col("Quantity") + pl.lit(SMALL_NUMBER))

    @property
    def aggregation_expression(self) -> pl.Expr:
        return (pl.col("CleanPrice") * pl.col("Quantity") + pl.col("AccruedInterest")).sum() / (pl.col("Quantity").sum() + pl.lit(SMALL_NUMBER))

class MarketValue(DerivedMetric):
    name = "MarketValue"

    @property
    def get_expression(self) -> pl.Expr:
        return pl.col("CleanPrice") * pl.col("Quantity") + pl.col("AccruedInterest")

    @property
    def aggregation_expression(self) -> pl.Expr:
        return self.get_expression.sum()

class DerivedAmount(DerivedMetric):
    def __init__(self, column: str, weight_column: str, allocation_expr: pl.Expr = pl.col("Quantity")):
        self.weight_column = weight_column
        self.allocation_expr = allocation_expr + pl.lit(SMALL_NUMBER)  # Prevent division by zero
        self.column = column

    @property
    def name(self) -> str:
        return self.column

    @property
    def get_expression(self) -> pl.Expr:
        return pl.col(self.weight_column) * self.allocation_expr

    def set_expression(self, amounts: pl.Expr) -> pl.Expr:
        return amounts * self.allocation_expr

    @property
    def aggregation_expression(self) -> pl.Expr:
        return self.get_expression.sum()

    def mutation_expression(self, amount: float, filter_expression: pl.Expr) -> pl.Expr:
        return amount * self.allocation_expr / (filter_expression * self.allocation_expr).sum()

    @property
    def mutation_column(self) -> str:
        return self.weight_column


class DerivedWeight(DerivedMetric):
    def __init__(self, column: str, amount_column: str, weight_expr: pl.Expr = pl.col("Quantity")):
        self.amount_column = amount_column
        self.weight_expr = weight_expr + pl.lit(SMALL_NUMBER)  # Prevent division by zero
        self.column = column

    @property
    def name(self) -> str:
        return self.column

    @property
    def get_expression(self) -> pl.Expr:
        return pl.col(self.amount_column) / self.weight_expr

    def set_expression(self, amounts: pl.Expr) -> pl.Expr:
        return amounts * self.weight_expr

    @property
    def aggregation_expression(self) -> pl.Expr:
        return pl.col(self.amount_column).sum() / self.weight_expr.sum()

    def mutation_expression(self, amount: float, filter_expression: pl.Expr) -> pl.Expr:
        return self.weight_expr * amount

    @property
    def mutation_column(self) -> str:
        return self.amount_column


# TODO: Determine exposure for fair value items
class Exposure(DerivedMetric):
    name = "Exposure"

    @property
    def get_expression(self) -> pl.Expr:
        return pl.col("Quantity") + pl.col("OffBalance")

    @property
    def aggregation_expression(self) -> pl.Expr:
        return self.get_expression.sum()


class BookValue(DerivedMetric):
    name = "BookValue"

    @property
    def get_expression(self) -> pl.Expr:
        return (
            pl.when(pl.col("AccountingMethod") == "amortizedcost")
            .then(pl.col("Quantity") + pl.col("Agio") + pl.col("AccruedInterest") + pl.col("Impairment"))
            .otherwise(pl.col("Quantity") * pl.col("CleanPrice") + pl.col("AccruedInterest") + pl.col("Agio"))
        )

    @property
    def aggregation_expression(self) -> pl.Expr:
        return self.get_expression.sum()


class BalanceSheetMetrics(BaseRegistry[BalanceSheetMetric]):
    @classmethod
    def stored_columns(cls) -> list[str]:
        return [metric.column for name, metric in cls.items.items() if metric.is_stored]


BalanceSheetMetrics.register("quantity", StoredAmount("Quantity"))
BalanceSheetMetrics.register("impairment", StoredAmount("Impairment"))
BalanceSheetMetrics.register("accrued_interest", StoredAmount("AccruedInterest"))
BalanceSheetMetrics.register("agio", StoredAmount("Agio"))
BalanceSheetMetrics.register("clean_price", StoredWeight("CleanPrice"))
BalanceSheetMetrics.register("off_balance", StoredWeight("OffBalance"))

BalanceSheetMetrics.register("dirty_price", DirtyPrice())
BalanceSheetMetrics.register("market_value", MarketValue())

BalanceSheetMetrics.register("coverage_rate", DerivedWeight("CoverageRate", "Impairment"))
BalanceSheetMetrics.register("accrued_interest_rate", DerivedWeight("AccruedInterestRate", "AccruedInterest"))
BalanceSheetMetrics.register("agio_weight", DerivedWeight("AgioWeight", "Agio"))

BalanceSheetMetrics.register("book_value", BookValue())
BalanceSheetMetrics.register("exposure", Exposure())

BalanceSheetMetrics.register("floating_rate", StoredWeight("FloatingRate"))
BalanceSheetMetrics.register("spread", StoredWeight("Spread"))
BalanceSheetMetrics.register("interest_rate", StoredWeight("InterestRate"))
BalanceSheetMetrics.register("prepayment_rate", StoredWeight("PrepaymentRate"))


BalanceSheetMetrics.register("trea_weight", StoredWeight("TREAWeight", Exposure().get_expression))
BalanceSheetMetrics.register("trea", DerivedAmount("TREA", "TREAWeight", Exposure().get_expression))
