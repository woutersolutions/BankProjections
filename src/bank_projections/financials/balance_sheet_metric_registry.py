import polars as pl

from bank_projections.financials.balance_sheet_category import BalanceSheetCategoryRegistry
from bank_projections.financials.balance_sheet_metrics import (
    HQLA,
    BalanceSheetMetric,
    BaselExposure,
    BookValue,
    DerivedAmount,
    DerivedMetric,
    DerivedWeight,
    DirtyPrice,
    EncumberedHQLA,
    LeverageExposure,
    MarketValue,
    OffBalanceExposure,
    OnBalanceExposure,
    StoredAmount,
    StoredColumn,
    StoredWeight,
    UnencumberedHQLA,
)
from bank_projections.utils.base_registry import BaseRegistry


class BalanceSheetMetrics(BaseRegistry[BalanceSheetMetric]):
    @classmethod
    def stored_columns(cls) -> list[str]:
        return [metric.column for metric in cls.values() if isinstance(metric, StoredColumn)]


BalanceSheetMetrics.register("Quantity", StoredAmount("Quantity"))
BalanceSheetMetrics.register("Impairment", StoredAmount("Impairment"))
BalanceSheetMetrics.register("AccruedInterest", StoredAmount("AccruedInterest"))
BalanceSheetMetrics.register("FairValueAdjustment", StoredAmount("FairValueAdjustment"))
BalanceSheetMetrics.register("Undrawn", StoredAmount("Undrawn"))
BalanceSheetMetrics.register("Agio", StoredAmount("Agio"))
BalanceSheetMetrics.register("CleanPrice", StoredWeight("CleanPrice"))
BalanceSheetMetrics.register("OtherOffBalanceWeight", StoredWeight("OtherOffBalanceWeight"))

BalanceSheetMetrics.register("DirtyPrice", DirtyPrice())
BalanceSheetMetrics.register("ValuationError", StoredWeight("ValuationError"))
BalanceSheetMetrics.register("MarketValue", MarketValue())

BalanceSheetMetrics.register("CoverageRate", DerivedWeight("Impairment"))
BalanceSheetMetrics.register("AccruedInterestWeight", DerivedWeight("AccruedInterest"))
BalanceSheetMetrics.register("AgioWeight", DerivedWeight("Agio"))
BalanceSheetMetrics.register("UndrawnPortion", DerivedWeight("Undrawn"))

BalanceSheetMetrics.register("BookValue", BookValue())


class BookValueSigned(DerivedMetric):
    @property
    def get_expression(self) -> pl.Expr:
        return BalanceSheetCategoryRegistry.book_value_sign() * (
            pl.when(pl.col("AccountingMethod") == "amortizedcost")
            .then(pl.col("Quantity") + pl.col("Agio") + pl.col("AccruedInterest") + pl.col("Impairment"))
            .otherwise(pl.col("Quantity") * pl.col("CleanPrice") + pl.col("AccruedInterest") + pl.col("Agio"))
        )

    @property
    def aggregation_expression(self) -> pl.Expr:
        return self.get_expression.sum()


BalanceSheetMetrics.register("BookValueSigned", BookValueSigned())

BalanceSheetMetrics.register("OtherOffBalance", DerivedAmount("OtherOffBalanceWeight"))
BalanceSheetMetrics.register("OnBalanceExposure", OnBalanceExposure())
BalanceSheetMetrics.register("OffBalanceExposure", OffBalanceExposure())
BalanceSheetMetrics.register("BaselExposure", BaselExposure())
BalanceSheetMetrics.register("LeverageExposure", LeverageExposure())

BalanceSheetMetrics.register("FloatingRate", StoredWeight("FloatingRate"))
BalanceSheetMetrics.register("Spread", StoredWeight("Spread"))
BalanceSheetMetrics.register("InterestRate", StoredWeight("InterestRate"))
BalanceSheetMetrics.register("PrepaymentRate", StoredWeight("PrepaymentRate"))
BalanceSheetMetrics.register("CCF", StoredWeight("CCF", pl.col("Undrawn")))


BalanceSheetMetrics.register("TREAWeight", StoredWeight("TREAWeight", BaselExposure().get_expression))
BalanceSheetMetrics.register("TREA", DerivedAmount("TREAWeight", BaselExposure().get_expression))

BalanceSheetMetrics.register("EncumberedWeight", StoredWeight("EncumberedWeight"))
BalanceSheetMetrics.register("Encumbered", DerivedAmount("EncumberedWeight"))

BalanceSheetMetrics.register("StableFundingWeight", StoredWeight("StableFundingWeight"))
BalanceSheetMetrics.register("StableFunding", DerivedAmount("StableFundingWeight"))

BalanceSheetMetrics.register("StressedOutflowWeight", StoredWeight("StressedOutflowWeight"))
BalanceSheetMetrics.register("StressedOutflow", DerivedAmount("StressedOutflowWeight"))

BalanceSheetMetrics.register("HQLA", HQLA())
BalanceSheetMetrics.register("EncumberedHQLA", EncumberedHQLA())
BalanceSheetMetrics.register("UnencumberedHQLA", UnencumberedHQLA())
