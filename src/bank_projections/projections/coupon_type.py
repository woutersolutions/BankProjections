from abc import ABC, abstractmethod

import polars as pl

from bank_projections.projections.base_registry import BaseRegistry


class CouponType(ABC):
    @classmethod
    @abstractmethod
    def coupon_rate(cls, floating_rate: pl.Expr) -> pl.Expr:
        raise NotImplementedError("Subclasses must implement this method")


class Fixed(CouponType):
    @classmethod
    def coupon_rate(cls, floating_rate: pl.Expr) -> pl.Expr:
        return pl.col("InterestRate")


class Floating(CouponType):
    @classmethod
    def coupon_rate(cls, floating_rate: pl.Expr) -> pl.Expr:
        return floating_rate + pl.col("Spread")


class Zero(CouponType):
    @classmethod
    def coupon_rate(cls, floating_rate: pl.Expr) -> pl.Expr:
        return pl.lit(0.0)


class CouponTypeRegistry(BaseRegistry[CouponType], CouponType):
    @classmethod
    def coupon_rate(cls, floating_rate: pl.Expr) -> pl.Expr:
        expr = pl.col("InterestRate")  # Default, should not be used
        for name, impl in cls.items.items():
            expr = pl.when(pl.col("CouponType") == name).then(impl.coupon_rate(floating_rate)).otherwise(expr)

        return expr


CouponTypeRegistry.register("fixed", Fixed())
CouponTypeRegistry.register("floating", Floating())
CouponTypeRegistry.register("zero", Zero())
CouponTypeRegistry.register("none", Zero())
