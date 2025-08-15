from abc import ABC, abstractmethod

import polars as pl


class BalanceSheetMetric(ABC):
    @property
    @abstractmethod
    def aggregation_expression(self) -> pl.Expr:
        pass

    @property
    @abstractmethod
    def mutation_expression(self) -> pl.Expr:
        pass

    @property
    @abstractmethod
    def mutation_column(self) -> str:
        pass

    @property
    @abstractmethod
    def is_stored(self) -> bool:
        pass


class Quantity(BalanceSheetMetric):
    @property
    def aggregation_expression(self) -> pl.Expr:
        return pl.col("Quantity").sum()

    @property
    def mutation_expression(self) -> pl.Expr:
        return pl.lit(1.0)

    @property
    def mutation_column(self) -> str:
        return "Quantity"

    is_stored: bool = True


class WeightMetric(BalanceSheetMetric):
    is_stored: bool = True

    def __init__(self, weight_column: str) -> None:
        self.weight_column = weight_column

    @property
    def aggregation_expression(self) -> pl.Expr:
        return (pl.col(self.weight_column) * pl.col("Quantity")).sum() / pl.col("Quantity").sum()

    @property
    def mutation_expression(self) -> pl.Expr:
        return pl.lit(1.0)

    @property
    def mutation_column(self) -> str:
        return self.weight_column


class AmountMetric(BalanceSheetMetric):
    is_stored: bool = False

    def __init__(self, weight_column: str) -> None:
        self.weight_column = weight_column

    @property
    def aggregation_expression(self) -> pl.Expr:
        return (pl.col(self.weight_column) * pl.col("Quantity")).sum()

    @property
    def mutation_expression(self) -> pl.Expr:
        return pl.lit(1.0) / pl.col("Quantity").sum()

    @property
    def mutation_column(self) -> str:
        return self.weight_column


class DerivedMetric(BalanceSheetMetric, ABC):
    is_stored: bool = False

    @property
    def mutation_expression(self) -> pl.Expr:
        raise NotImplementedError("Derived metric cannot be modified")

    @property
    def mutation_column(self) -> str:
        raise NotImplementedError("Derived metric cannot be modified")


class BookValue(DerivedMetric):
    """Calculate book value based on valuation method (amortized cost vs fair value)."""

    @property
    def aggregation_expression(self) -> pl.Expr:
        """
        Calculate total book value across all positions based on valuation method.

        For amortized cost: Quantity * (1 + AgioWeight + AccruedInterestWeight - CoverageRate)
        For fair value: Quantity * (1 + DirtyPrice + AgioWeight)

        Returns:
            pl.Expr: Polars expression to calculate aggregated book value
        """
        return (
            pl.when(pl.col("ValuationMethod") == "amortized cost")
            .then(
                pl.col("Quantity")
                * (1 + pl.col("AgioWeight") + pl.col("AccruedInterestWeight") - pl.col("CoverageRate"))
            )
            .when(pl.col("ValuationMethod") == "fair value")
            .then(pl.col("Quantity") * (1 + pl.col("DirtyPrice") + pl.col("AgioWeight")))
            .sum()
        )


class CleanPrice(DerivedMetric):
    @property
    def aggregation_expression(self) -> pl.Expr:
        return pl.col("Quantity") * (pl.col("DirtyPrice") - pl.col("AccruedInterestRate"))


class BalanceSheetMetrics:
    quantity = Quantity()
    impairment = AmountMetric("CoverageRate")
    coverage_rate = WeightMetric("CoverageRate")
    accrued_interest = AmountMetric("AccruedInterestRate")
    accrued_interest_rate = WeightMetric("AccruedInterestRate")
    agio = AmountMetric("AgioWeight")
    agio_weight = WeightMetric("AgioWeight")
    dirty_price = WeightMetric("AgioWeight")
    clean_price = CleanPrice()
    book_value = BookValue()
