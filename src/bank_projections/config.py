# TODO: Arrange these configurations into a config file
from typing import Any

import polars as pl
from pydantic import BaseModel

from bank_projections.financials.accounting_method import AccountingMethodRegistry
from bank_projections.financials.balance_sheet_category import BalanceSheetCategoryRegistry
from bank_projections.financials.balance_sheet_metric_registry import BalanceSheetMetrics
from bank_projections.financials.hqla_class import HQLARegistry
from bank_projections.financials.stage import IFRS9StageRegistry
from bank_projections.projections.book import BookRegistry
from bank_projections.projections.coupon_type import CouponTypeRegistry
from bank_projections.projections.frequency import FrequencyRegistry
from bank_projections.projections.redemption import RedemptionRegistry
from bank_projections.projections.valuation_method import ValuationMethodRegistry
from bank_projections.utils.base_registry import BaseRegistry
from bank_projections.utils.time import TimeHorizonConfig


class ScenarioConfig(BaseModel):
    rule_paths: list[str]
    time_horizon: TimeHorizonConfig


class AggregationConfig(BaseModel):
    balance_sheet: list[str] | None = None
    pnl: list[str] | None = None
    cashflow: list[str] | None = None
    oci: list[str] | None = None


class OutputConfig(BaseModel):
    output_folder: str
    output_file: str
    aggregation: AggregationConfig


class Config:
    BALANCE_SHEET_LABELS = [
        "ItemType",
        "SubItemType",
        "Currency",
        "ReferenceRate",
        "ValuationCurve",
        "IsAccumulating",
    ]
    CASHFLOW_LABELS = ["ItemType", "SubItemType", "module", "rule"]
    PNL_LABELS = ["ItemType", "SubItemType", "module", "rule"]
    OCI_LABELS = ["ItemType", "SubItemType", "module", "rule"]

    DATE_COLUMNS = ["OriginationDate", "MaturityDate", "PreviousCouponDate", "NextCouponDate"]

    CLASSIFICATIONS: dict[str, type[BaseRegistry[Any]]] = {
        "BalanceSheetCategory": BalanceSheetCategoryRegistry,
        "AccountingMethod": AccountingMethodRegistry,
        "ValuationMethod": ValuationMethodRegistry,
        "CouponFrequency": FrequencyRegistry,
        "RedemptionType": RedemptionRegistry,
        "CouponType": CouponTypeRegistry,
        "IFRS9Stage": IFRS9StageRegistry,
        "HQLAClass": HQLARegistry,
        "Book": BookRegistry,
    }

    PROFITABILITY_OUTLOOKS: list[str] = ["Monthly", "Quarterly", "Annual"]

    @classmethod
    def label_columns(cls) -> list[str]:
        return Config.BALANCE_SHEET_LABELS + Config.DATE_COLUMNS + list(Config.CLASSIFICATIONS.keys())

    @classmethod
    def required_columns(cls) -> list[str]:
        return (
            cls.BALANCE_SHEET_LABELS
            + list(cls.CLASSIFICATIONS.keys())
            + BalanceSheetMetrics.stored_columns()
            + cls.DATE_COLUMNS
        )

    @classmethod
    def non_null_columns(cls) -> list[str]:
        return list(cls.CLASSIFICATIONS.keys()) + ["Quantity", "Impairment", "AccruedInterest", "Agio"]

    @classmethod
    def cast_columns(cls, df: pl.DataFrame) -> pl.DataFrame:
        # TODO: Also cast other columns
        return df.with_columns(
            [pl.col(name).cast(pl.Enum(registry.stripped_names())) for name, registry in cls.CLASSIFICATIONS.items()]
        )
