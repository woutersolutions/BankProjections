# TODO: Properly arrange these configurations into a config file or environment variables
from typing import Any

from bank_projections.financials.accounting_method import AccountingMethodRegistry
from bank_projections.financials.balance_sheet_metrics import BalanceSheetMetrics
from bank_projections.financials.hqla_class import HQLARegistry
from bank_projections.financials.stage import IFRS9StageRegistry
from bank_projections.projections.coupon_type import CouponTypeRegistry
from bank_projections.projections.frequency import FrequencyRegistry
from bank_projections.projections.redemption import RedemptionRegistry
from bank_projections.projections.valuation_method import ValuationMethodRegistry
from bank_projections.utils.base_registry import BaseRegistry


class Config:
    CASHFLOW_AGGREGATION_LABELS = ["ItemType", "SubItemType"]
    PNL_AGGREGATION_LABELS = ["ItemType", "SubItemType"]
    OCI_AGGREGATION_LABELS = ["ItemType", "SubItemType"]
    BALANCE_SHEET_LABELS = [
        "BalanceSheetSide",
        "ItemType",
        "SubItemType",
        "Currency",
        "ReferenceRate",
        "ValuationCurve",
        "IsAccumulating",
    ]
    DATE_COLUMNS = ["OriginationDate", "MaturityDate", "PreviousCouponDate", "NextCouponDate"]

    CLASSIFICATIONS: dict[str, type[BaseRegistry[Any]]] = {
        "AccountingMethod": AccountingMethodRegistry,
        "ValuationMethod": ValuationMethodRegistry,
        "CouponFrequency": FrequencyRegistry,
        "RedemptionType": RedemptionRegistry,
        "CouponType": CouponTypeRegistry,
        "IFRS9Stage": IFRS9StageRegistry,
        "HQLAClass": HQLARegistry,
    }

    BALANCE_SHEET_AGGREGATION_LABELS = ["BalanceSheetSide", "ItemType", "SubItemType"] + list(CLASSIFICATIONS.keys())

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
