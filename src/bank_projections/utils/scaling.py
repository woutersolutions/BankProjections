from abc import ABC, abstractmethod

import polars as pl


class ScalingMethod(ABC):
    @abstractmethod
    def scale(self, amount: pl.Expr, factor: pl.Expr) -> pl.Expr:
        pass

    @abstractmethod
    def scaling_factor(self, original_amount: pl.Expr, new_amount: pl.Expr) -> pl.Expr:
        pass


class AdditiveScaling(ScalingMethod):
    def scale(self, amount: pl.Expr, factor: pl.Expr) -> pl.Expr:
        return amount + factor

    def scaling_factor(self, original_amount: pl.Expr, new_amount: pl.Expr) -> pl.Expr:
        return new_amount - original_amount


class MultiplicativeScaling(ScalingMethod):
    def scale(self, amount: pl.Expr, factor: pl.Expr) -> pl.Expr:
        return amount * (factor + 1.0)

    def scaling_factor(self, original_amount: pl.Expr, new_amount: pl.Expr) -> pl.Expr:
        return new_amount / original_amount - 1.0


class NoScaling(ScalingMethod):
    def scale(self, amount: pl.Expr, factor: pl.Expr) -> pl.Expr:
        return amount

    def scaling_factor(self, original_amount: pl.Expr, new_amount: pl.Expr) -> pl.Expr:
        return pl.lit(0.0)
