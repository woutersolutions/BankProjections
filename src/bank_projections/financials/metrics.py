from abc import ABC, abstractmethod

import polars as pl


class BalanceSheetMetric(ABC):

    @abstractmethod
    @property
    def aggregation_expression(self) -> pl.expression.Expression:
        pass

    @abstractmethod
    @property
    def mutation_expression(self) -> pl.expression.Expression:
        pass

    @abstractmethod
    @property
    def mutation_column(self) -> str:
        pass

    @abstractmethod
    @property
    def is_stored(self) -> bool:
        pass


class Quantity(BalanceSheetMetric):

    aggregation_expression = pl.col("Quantity").sum()
    mutation_expression = pl.expression.Expression
    mutation_column = "Quantity"
    is_stored = True


class WeightMetric(BalanceSheetMetric):

    is_stored = True

    def __init__(self, weight_column: str) -> None:
        self.weight_column = weight_column

    @property
    def aggregation_expression(self) -> pl.expression.Expression:
        return (pl.col(self.weight_column) * pl.col("Quantity")).sum() / pl.col(
            "Quantity"
        ).sum()

    @property
    def mutation_expression(self) -> pl.expression.Expression:
        return 1.0

    @property
    def mutation_column(self) -> str:
        return self.weight_column


class AmountMetric(BalanceSheetMetric):

    is_stored = False

    def __init__(self, weight_column: str) -> None:
        self.weight_column = weight_column

    @property
    def aggregation_expression(self) -> pl.expression.Expression:
        return (pl.col(self.weight_column) * pl.col("Quantity")).sum()

    @property
    def mutation_expression(self) -> pl.expression.Expression:
        return 1.0 / pl.col("Quantity").sum()

    @property
    def mutation_column(self) -> str:
        return self.weight_column


class DerivedMetric(BalanceSheetMetric, ABC):

    is_stored = False

    @property
    def mutation_expression(self) -> pl.expression.Expression:
        raise not NotImplementedError("Derived metric cannot be modified")

    @property
    def mutation_column(self) -> str:
        raise not NotImplementedError("Derived metric cannot be modified")


class BookValue(DerivedMetric):

    @property
    def aggregation_expression(self) -> pl.expression.Expression:
        return (
            pl.when(pl.col("ValuationMethod") == "amortized cost")
            .then(
                (
                    pl.col("Quantity")
                    * (
                        1
                        + pl.col("AgioWeight")
                        + pl.col("AccruedInterestWeight")
                        - pl.col("CoverageRate")
                    )
                )
            )
            .when(pl.col("ValuationMethod") == "fair value")
            .then(
                (pl.col("Quantity") * (1 + pl.col("DirtyPrice") + pl.col("AgioWeight")))
            )
            .sum()
        )


class CleanPrice(DerivedMetric):

    @property
    def aggregation_expression(self) -> pl.expression.Expression:
        return pl.col("Quantity") * (
            pl.col("DirtyPrice") - pl.col("AccruedInterestRate")
        )


class BalanceSheetMetrics:

    quantity = (Quantity(),)
    impairment = AmountMetric("CoverageRate")
    coverage_rate = WeightMetric("CoverageRate")
    accrued_interest = AmountMetric("AccruedInterestRate")
    accrued_interest_rate = WeightMetric("AccruedInterestRate")
    agio = AmountMetric("AgioWeight")
    agio_weight = WeightMetric("AgioWeight")
    dirty_price = WeightMetric("AgioWeight")
    clean_price = CleanPrice()
    book_value = BookValue()
