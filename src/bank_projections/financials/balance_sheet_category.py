from abc import ABC, abstractmethod

import polars as pl

from bank_projections.financials.balance_sheet_metrics import MarketValue
from bank_projections.utils.base_registry import BaseRegistry


class BalanceSheetCategory(ABC):
    @property
    @abstractmethod
    def book_value_reversed(self) -> bool:
        pass

    @property
    @abstractmethod
    def is_asset_side(self) -> pl.Expr:
        pass


class AssetSide(BalanceSheetCategory):
    @property
    def book_value_reversed(self) -> bool:
        return False

    @property
    def is_asset_side(self) -> pl.Expr:
        return pl.lit(True)


class FundingSide(BalanceSheetCategory):
    @property
    def book_value_reversed(self) -> bool:
        return True

    @property
    def is_asset_side(self) -> pl.Expr:
        return pl.lit(False)


class SideDependsOnMarketValue(BalanceSheetCategory):
    @property
    def book_value_reversed(self) -> bool:
        return False

    @property
    def is_asset_side(self) -> pl.Expr:
        return MarketValue().aggregation_expression >= 0


class SideDependsOnQuantity(BalanceSheetCategory):
    @property
    def book_value_reversed(self) -> bool:
        return False

    @property
    def is_asset_side(self) -> pl.Expr:
        return pl.col("Quantity").sum() >= 0


class BalanceSheetCategoryRegistry(BaseRegistry[BalanceSheetCategory]):
    @classmethod
    def book_value_sign(cls) -> pl.Expr:
        expr = pl.lit(1)
        for name, category_cls in cls.stripped_items.items():
            sign = -1 if category_cls.book_value_reversed else 1
            expr = pl.when(pl.col("BalanceSheetCategory") == name).then(pl.lit(sign)).otherwise(expr)
        return expr


BalanceSheetCategoryRegistry.register("Assets", AssetSide())
BalanceSheetCategoryRegistry.register("Liabilities", FundingSide())
BalanceSheetCategoryRegistry.register("Equity", FundingSide())
BalanceSheetCategoryRegistry.register("Derivatives", SideDependsOnMarketValue())
BalanceSheetCategoryRegistry.register("Collateral", SideDependsOnQuantity())
