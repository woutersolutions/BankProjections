from abc import ABC, abstractmethod

import polars as pl


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

    @property
    @abstractmethod
    def is_stored(self) -> bool:
        pass


class StoredColumn(BalanceSheetMetric, ABC):
    is_stored = True

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


class CoreAmount(StoredColumn):
    @property
    def aggregation_expression(self) -> pl.Expr:
        return self.get_expression.sum()

    def mutation_expression(self, amount: float, filter_expression: pl.Expr) -> pl.Expr:
        return pl.lit(amount) * pl.col("Quantity") / (filter_expression * pl.col("Quantity")).sum()


class CleanPrice(StoredColumn):
    @property
    def aggregation_expression(self) -> pl.Expr:
        return (pl.col(self.column) * pl.col("Quantity")).sum() / pl.col("Quantity").sum()

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
    @property
    def get_expression(self) -> pl.Expr:
        return pl.col("Quantity") * pl.col("CleanPrice") + pl.col("AccruedInterestRate")

    @property
    def aggregation_expression(self) -> pl.Expr:
        return self.get_expression.sum()


class DerivedWeight(DerivedMetric):
    def __init__(self, amount_column: str):
        self.amount_column = amount_column

    @property
    def get_expression(self) -> pl.Expr:
        return pl.col(self.amount_column) / pl.col("Quantity")

    def set_expression(self, amounts: pl.Expr) -> pl.Expr:
        return amounts * pl.col("Quantity")

    @property
    def aggregation_expression(self) -> pl.Expr:
        return pl.col(self.amount_column).sum() / pl.col("Quantity").sum()

    def mutation_expression(self, amount: float, filter_expression: pl.Expr) -> pl.Expr:
        return pl.col("Quantity") * amount

    @property
    def mutation_column(self) -> str:
        return self.amount_column


class BookValue(DerivedMetric):
    @property
    def get_expression(self) -> pl.Expr:
        return (
            pl.when(pl.col("ValuationMethod") == "amortized cost")
            .then(pl.col("Quantity") + pl.col("Agio") + pl.col("AccruedInterest") + pl.col("Impairment"))
            .when(pl.col("ValuationMethod") == "fair value")
            .then(pl.col("Quantity") * pl.col("CleanPrice") + pl.col("AccruedInterest") + pl.col("Agio"))
        )

    @property
    def aggregation_expression(self) -> pl.Expr:
        return self.get_expression.sum()


class BalanceSheetMetrics:
    quantity = CoreAmount("Quantity")
    impairment = CoreAmount("Impairment")
    accrued_interest = CoreAmount("AccruedInterest")
    agio = CoreAmount("Agio")
    clean_price = CleanPrice("CleanPrice")

    dirty_price = DirtyPrice()

    coverage_rate = DerivedWeight("Impairment")
    accrued_interest_rate = DerivedWeight("AccruedInterest")
    agio_weight = DerivedWeight("Agio")

    book_value = BookValue()
