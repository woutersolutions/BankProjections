from abc import ABC, abstractmethod

import polars as pl

from bank_projections.financials.hqla_class import HQLARegistry
from bank_projections.utils.base_registry import BaseRegistry

SMALL_NUMBER = 1e-12


class BalanceSheetMetric(ABC):
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


class StoredColumn(BalanceSheetMetric, ABC):
    def __init__(self, column: str):
        self.column = column

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
    def set_expression(self, amounts: pl.Expr) -> pl.Expr:
        raise NotImplementedError("Derived metric cannot be modified")

    def mutation_expression(self, amount: float, filter_expression: pl.Expr) -> pl.Expr:
        raise NotImplementedError("Derived metric cannot be modified")

    @property
    def mutation_column(self) -> str:
        raise NotImplementedError("Derived metric cannot be modified")


class DirtyPrice(DerivedMetric):
    @property
    def get_expression(self) -> pl.Expr:
        return pl.col("CleanPrice") + pl.col("AccruedInterest") / (pl.col("Quantity") + pl.lit(SMALL_NUMBER))

    @property
    def aggregation_expression(self) -> pl.Expr:
        return (pl.col("CleanPrice") * pl.col("Quantity") + pl.col("AccruedInterest")).sum() / (
            pl.col("Quantity").sum() + pl.lit(SMALL_NUMBER)
        )


class MarketValue(DerivedMetric):
    @property
    def get_expression(self) -> pl.Expr:
        return pl.col("CleanPrice") * pl.col("Quantity") + pl.col("AccruedInterest")

    @property
    def aggregation_expression(self) -> pl.Expr:
        return self.get_expression.sum()


class DerivedAmount(DerivedMetric):
    def __init__(self, weight_column: str, allocation_expr: pl.Expr = pl.col("Quantity")):
        self.weight_column = weight_column
        self.allocation_expr = allocation_expr + pl.lit(SMALL_NUMBER)  # Prevent division by zero

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
    def __init__(self, amount_column: str, weight_expr: pl.Expr = pl.col("Quantity")):
        self.amount_column = amount_column
        self.weight_expr = weight_expr + pl.lit(SMALL_NUMBER)  # Prevent division by zero

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
    @property
    def get_expression(self) -> pl.Expr:
        return pl.col("Quantity") + pl.col("CCF") * pl.col("Undrawn") + pl.col("OffBalance")

    @property
    def aggregation_expression(self) -> pl.Expr:
        return self.get_expression.sum()


class BookValue(DerivedMetric):
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


class HQLA(DerivedMetric):
    @property
    def get_expression(self) -> pl.Expr:
        return HQLARegistry.hqla_constribution_expression() * BookValue().get_expression

    @property
    def aggregation_expression(self) -> pl.Expr:
        return self.get_expression.sum()


class EncumberedHQLA(DerivedMetric):
    @property
    def get_expression(self) -> pl.Expr:
        return pl.col("EncumberedWeight") * HQLA().get_expression

    @property
    def aggregation_expression(self) -> pl.Expr:
        return self.get_expression.sum()


class UnencumberedHQLA(DerivedMetric):
    @property
    def get_expression(self) -> pl.Expr:
        return (1 - pl.col("EncumberedWeight")) * HQLA().get_expression

    @property
    def aggregation_expression(self) -> pl.Expr:
        return self.get_expression.sum()


class BalanceSheetMetrics(BaseRegistry[BalanceSheetMetric]):
    @classmethod
    def stored_columns(cls) -> list[str]:
        return [metric.column for metric in cls.values() if isinstance(metric, StoredColumn)]


BalanceSheetMetrics.register("Quantity", StoredAmount("Quantity"))
BalanceSheetMetrics.register("Impairment", StoredAmount("Impairment"))
BalanceSheetMetrics.register("AccruedInterest", StoredAmount("AccruedInterest"))
BalanceSheetMetrics.register("Undrawn", StoredAmount("Undrawn"))
BalanceSheetMetrics.register("Agio", StoredAmount("Agio"))
BalanceSheetMetrics.register("CleanPrice", StoredWeight("CleanPrice"))
BalanceSheetMetrics.register("Off-Balance", StoredWeight("OffBalance"))

BalanceSheetMetrics.register("DirtyPrice", DirtyPrice())
BalanceSheetMetrics.register("ValuationError", StoredWeight("ValuationError"))
BalanceSheetMetrics.register("MarketValue", MarketValue())

BalanceSheetMetrics.register("CoverageRate", DerivedWeight("Impairment"))
BalanceSheetMetrics.register("AccruedInterestWeight", DerivedWeight("AccruedInterest"))
BalanceSheetMetrics.register("AgioWeight", DerivedWeight("Agio"))
BalanceSheetMetrics.register("UndrawnPortion", DerivedWeight("Undrawn"))


BalanceSheetMetrics.register("BookValue", BookValue())
BalanceSheetMetrics.register("Exposure", Exposure())

BalanceSheetMetrics.register("FloatingRate", StoredWeight("FloatingRate"))
BalanceSheetMetrics.register("Spread", StoredWeight("Spread"))
BalanceSheetMetrics.register("InterestRate", StoredWeight("InterestRate"))
BalanceSheetMetrics.register("PrepaymentRate", StoredWeight("PrepaymentRate"))
BalanceSheetMetrics.register("CCF", StoredWeight("CCF", pl.col("Undrawn")))


BalanceSheetMetrics.register("TREAWeight", StoredWeight("TREAWeight", Exposure().get_expression))
BalanceSheetMetrics.register("TREA", DerivedAmount("TREAWeight", Exposure().get_expression))

BalanceSheetMetrics.register("EncumberedWeight", StoredWeight("EncumberedWeight"))
BalanceSheetMetrics.register("Encumbered", DerivedAmount("EncumberedWeight"))

BalanceSheetMetrics.register("StableFundingWeight", StoredWeight("StableFundingWeight"))
BalanceSheetMetrics.register("StableFunding", DerivedAmount("StableFundingWeight"))

BalanceSheetMetrics.register("StressedOutflowWeight", StoredWeight("StressedOutflowWeight"))
BalanceSheetMetrics.register("StressedOutflow", DerivedAmount("StressedOutflowWeight"))

BalanceSheetMetrics.register("HQLA", HQLA())
BalanceSheetMetrics.register("EncumberedHQLA", EncumberedHQLA())
BalanceSheetMetrics.register("UnencumberedHQLA", UnencumberedHQLA())
