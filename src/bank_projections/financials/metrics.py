from abc import ABC, abstractmethod

import polars as pl


class BalanceSheetMetric(ABC):
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


class CoreAmount(BalanceSheetMetric):
    is_stored = True

    def __init__(self, column: str):
        self.column = column

    @property
    def aggregation_expression(self) -> pl.Expr:
        return pl.col(self.column).sum()

    def mutation_expression(self, amount: float, filter_expression: pl.Expr) -> pl.Expr:
        return pl.lit(amount) * pl.col("Quantity") / (filter_expression * pl.col("Quantity")).sum()

    @property
    def mutation_column(self) -> str:
        return self.column


class CleanPrice(BalanceSheetMetric):
    is_stored: bool = True

    def __init__(self, column: str):
        self.column = column

    @property
    def aggregation_expression(self) -> pl.Expr:
        return (pl.col(self.column) * pl.col("Quantity")).sum() / pl.col("Quantity").sum()

    def mutation_expression(self, amount: float, filter_expression: pl.Expr) -> pl.Expr:
        return pl.lit(amount)

    @property
    def mutation_column(self) -> str:
        return self.column


class DerivedMetric(BalanceSheetMetric, ABC):
    is_stored: bool = False

    def mutation_expression(self, amount: float, filter_expression: pl.Expr) -> pl.Expr:
        raise NotImplementedError("Derived metric cannot be modified")

    @property
    def mutation_column(self) -> str:
        raise NotImplementedError("Derived metric cannot be modified")


class DirtyPrice(DerivedMetric):
    @property
    def aggregation_expression(self) -> pl.Expr:
        return pl.col("Quantity") * pl.col("CleanPrice") + pl.col("AccruedInterestRate")


class DerivedWeight(DerivedMetric):
    def __init__(self, amount_column: str):
        self.amount_column = amount_column

    @property
    def aggregation_expression(self) -> pl.Expr:
        return pl.col(self.amount_column).sum() / pl.col("Quantity")

    def mutation_expression(self, amount: float, filter_expression: pl.Expr) -> pl.Expr:
        return pl.col("Quantity") * amount

    @property
    def mutation_column(self) -> str:
        return self.amount_column


class WeightMetric(BalanceSheetMetric):
    is_stored: bool = True

    def __init__(self, weight_column: str) -> None:
        self.weight_column = weight_column

    @property
    def aggregation_expression(self) -> pl.Expr:
        return (pl.col(self.weight_column) * pl.col("Quantity")).sum() / pl.col("Quantity").sum()

    def mutation_expression(self, amount: float, filter_expression: pl.Expr) -> pl.Expr:
        return pl.lit(amount)

    @property
    def mutation_column(self) -> str:
        return self.weight_column


class BookValue(DerivedMetric):
    @property
    def aggregation_expression(self) -> pl.Expr:
        return (
            pl.when(pl.col("ValuationMethod") == "amortized cost")
            .then(pl.col("Quantity") + pl.col("Agio") + pl.col("AccruedInterest") + pl.col("Impairment"))
            .when(pl.col("ValuationMethod") == "fair value")
            .then(pl.col("Quantity") * pl.col("CleanPrice") + pl.col("AccruedInterest") + pl.col("Agio"))
            .sum()
        )


class BalanceSheetMetrics:
    quantity = CoreAmount("Quantity")
    impairment = CoreAmount("Impairment")
    accrued_interest = CoreAmount("AccruedInterest")
    agio = CoreAmount("AgioWeight")
    clean_price = CleanPrice("CleanPrice")

    dirty_price = DirtyPrice()

    coverage_rate = DerivedWeight("Impairment")
    accrued_interest_rate = DerivedWeight("AccruedInterest")
    agio_weight = DerivedWeight("AgioWeight")

    book_value = BookValue()
