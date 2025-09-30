# TODO: Properly arrange these configurations into a config file or environment variables
from typing import Any

from bank_projections.financials.metrics import BalanceSheetMetrics
from bank_projections.projections.base_registry import BaseRegistry
from bank_projections.projections.coupon_type import CouponTypeRegistry
from bank_projections.projections.frequency import FrequencyRegistry
from bank_projections.projections.redemption import RedemptionRegistry
from bank_projections.projections.valuation import ValuationRegistry


class Config:
    CASHFLOW_AGGREGATION_LABELS = ["ItemType"]
    PNL_AGGREGATION_LABELS = ["ItemType"]
    BALANCE_SHEET_LABELS = [
        "BalanceSheetSide",
        "ItemType",
        "Currency",
        "ReferenceRate",
        "IsAccumulating",
    ]
    DATE_COLUMNS = ["OriginationDate", "MaturityDate", "PreviousCouponDate", "NextCouponDate"]
    BALANCE_SHEET_AGGREGATION_LABELS = ["BalanceSheetSide", "ItemType"]

    CLASSIFICATIONS: dict[str, type[BaseRegistry[Any]]] = {
        "ValuationMethod": ValuationRegistry,
        "CouponFrequency": FrequencyRegistry,
        "RedemptionType": RedemptionRegistry,
        "CouponType": CouponTypeRegistry,
    }

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
