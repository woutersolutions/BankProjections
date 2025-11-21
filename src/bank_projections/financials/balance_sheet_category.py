from abc import ABC, abstractmethod

import polars as pl

from bank_projections.financials.balance_sheet_metrics import BalanceSheetMetrics
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
        return BalanceSheetMetrics.get("marketvalue").aggregation_expression >= 0


class SideDependsOnQuantity(BalanceSheetCategory):
    @property
    def book_value_reversed(self) -> bool:
        return False

    @property
    def is_asset_side(self) -> pl.Expr:
        return BalanceSheetMetrics.get("quantity").aggregation_expression >= 0


class BalanceSheetCategoryRegistry(BaseRegistry[BalanceSheetCategory]):
    pass


BalanceSheetCategoryRegistry.register("Assets", AssetSide())
BalanceSheetCategoryRegistry.register("Liabilities", FundingSide())
BalanceSheetCategoryRegistry.register("Equity", FundingSide())
BalanceSheetCategoryRegistry.register("Derivatives", SideDependsOnMarketValue())
BalanceSheetCategoryRegistry.register("Collateral", SideDependsOnQuantity())
