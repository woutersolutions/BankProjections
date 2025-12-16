import datetime
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from bank_projections.app_config import Config
from bank_projections.financials.balance_sheet import MutationReason
from bank_projections.financials.balance_sheet_item import BalanceSheetItem, BalanceSheetItemRegistry, Cohort
from bank_projections.financials.balance_sheet_metric_registry import BalanceSheetMetrics
from bank_projections.financials.balance_sheet_metrics import BalanceSheetMetric  # noqa: TC001
from bank_projections.financials.market_data import Curves, parse_tenor
from bank_projections.scenarios.excel_sheet_format import ExcelInput, KeyValueInput
from bank_projections.utils.parsing import (
    get_identifier,
    is_in_identifiers,
    read_bool,
    read_date,
    read_int,
    strip_identifier,
)
from bank_projections.utils.time import TimeIncrement


class ScenarioInput(ABC):
    @abstractmethod
    def __init__(self, excel_inputs: list[ExcelInput]) -> None:
        pass

    @abstractmethod
    def filter_on_date_snapshot(self, increment: TimeIncrement) -> "ScenarioInput":
        pass


class TaxInput(ScenarioInput):
    def __init__(self, excel_inputs: list[ExcelInput]) -> None:
        assert len(excel_inputs) == 1, "TaxInput requires exactly one ExcelInput."

        excel_input = excel_inputs[0]

        if not isinstance(excel_input, KeyValueInput):
            raise TypeError("TaxInput requires KeyValueInput format.")

        self.tax_rate = excel_input.general_tags["Tax Rate"]

    def filter_on_date_snapshot(self, increment: TimeIncrement) -> "TaxInput":
        return self


class AuditInput(ScenarioInput):
    def __init__(self, excel_inputs: list[ExcelInput]) -> None:
        assert len(excel_inputs) == 1, "AuditInput requires exactly one ExcelInput."

        excel_input = excel_inputs[0]
        if not isinstance(excel_input, KeyValueInput):
            raise TypeError("AuditInput requires KeyValueInput format.")

        self.target = BalanceSheetItem()
        for key, value in excel_input.general_tags.items():
            match strip_identifier(key):
                case "closingmonth":
                    self.closing_month = int(value)
                case "auditmonth":
                    self.audit_month = int(value)
                case _ if (stripped := strip_identifier(key)) is not None and stripped.startswith("target"):
                    label = strip_identifier(key[len("target") :])
                    if label is not None:
                        self.target = self.target.add_identifier(label, value)
                case _:
                    raise KeyError(f"{key} not recognized in AuditRule")

    def filter_on_date_snapshot(self, increment: TimeIncrement) -> "AuditInput":
        return self


class CurveInput(ScenarioInput):
    def __init__(self, excel_inputs: list[ExcelInput]) -> None:
        self.data = pd.concat([excel_input.to_dataframe() for excel_input in excel_inputs], ignore_index=True)
        self._enforce_schema()

    @staticmethod
    def _empty_frame() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Date": pd.Series(dtype="datetime64[ns]"),
                "Name": pd.Series(dtype="string"),
                "Type": pd.Series(dtype="string"),
                "Tenor": pd.Series(dtype="string"),
                "Maturity": pd.Series(dtype="string"),
                "Rate": pd.Series(dtype="float64"),
                "MaturityYears": pd.Series(dtype="float64"),
            }
        )

    def _enforce_schema(self) -> None:
        # Ensure all required columns exist
        template = self._empty_frame()
        for col in template.columns:
            if col not in self.data.columns:
                self.data[col] = template[col]

        # Coerce types
        self.data["Date"] = pd.to_datetime(self.data["Date"], errors="coerce")

        for col in ["Name", "Type", "Tenor", "Maturity"]:
            self.data[col] = self.data[col].astype("string").str.strip().str.lower()

        self.data["Rate"] = pd.to_numeric(self.data["Rate"], errors="coerce").astype("float64")

        # (Re)compute MaturityYears from Maturity ensuring float dtype
        self.data["MaturityYears"] = self.data["Maturity"].map(parse_tenor).astype("float64")

        # Column order normalization (optional but keeps consistency)
        self.data = self.data[template.columns]

    def filter_on_date_snapshot(self, increment: TimeIncrement) -> Curves:
        latest_date = self.data[self.data["Date"] <= pd.Timestamp(increment.to_date)]["Date"].max()
        if pd.isna(latest_date):
            raise ValueError(f"No curve data available for date {increment.to_date}")
        filtered_data = self.data.loc[self.data["Date"] == latest_date]

        # TODO: Interpolation between dates

        return Curves(filtered_data)


class ProductionInputItem:
    def __init__(self, **rule_input: Any) -> None:
        # Initialize defaults
        self.multiplicative = False
        self.reason = MutationReason(rule="Production")
        self.date: datetime.date | None = None
        self.metrics: dict[str, Any] = {}
        self.labels: dict[str, Any] = {}
        self.maturity: int | None = None

        if is_in_identifiers("reference item", list(rule_input.keys())):
            value = get_identifier("reference item", list(rule_input.keys()))
            if pd.isna(value) or value == "":
                self.reference_item = None
            else:
                self.reference_item = BalanceSheetItemRegistry.get(value)
        else:
            self.reference_item = None

        for key, value in rule_input.items():
            match strip_identifier(key):
                case _ if pd.isna(value) or value == "":
                    pass
                case "referenceitem":
                    pass
                case _ if is_in_identifiers(key, list(BalanceSheetMetrics.stripped_names())):
                    stripped_key = strip_identifier(key)
                    if stripped_key is not None:
                        self.metrics[stripped_key] = value
                case _ if (stripped := strip_identifier(key)) is not None and stripped.startswith("reference"):
                    stripped_key = strip_identifier(key)
                    if stripped_key is not None:
                        label = get_identifier(stripped_key.replace("reference", ""), Config.label_columns())
                        if self.reference_item is None:
                            self.reference_item = BalanceSheetItem(**{label: value})
                        else:
                            self.reference_item = self.reference_item.add_identifier(label, value)
                case _ if is_in_identifiers(key, Config.label_columns()):
                    self.labels[get_identifier(key, Config.label_columns())] = value
                case "multiplicative":
                    self.multiplicative = read_bool(value)
                case "date":
                    self.date = read_date(value)
                case "maturity":
                    self.maturity = value
                case _:
                    raise KeyError(f"{key} not recognized in ProductionRule")


class ProductionInput(ScenarioInput):
    def __init__(self, excel_inputs: list[ExcelInput]) -> None:
        self.production_items: list[ProductionInputItem] = [
            ProductionInputItem(**items) for excel_input in excel_inputs for items in excel_input.to_dict_list()
        ]

    def filter_on_date_snapshot(self, increment: TimeIncrement) -> list[ProductionInputItem]:
        return [item for item in self.production_items if item.date is None or increment.contains(item.date)]


class BalanceSheetMutationInputItem:
    def __init__(self, rule_input: dict[str, Any]):
        self.amount = rule_input["Amount"]
        self.relative = True
        self.multiplicative = False
        self.offset_liquidity = False
        self.offset_pnl = False
        self.reason = MutationReason(rule="BalanceSheetMutationRule", module="Mutation")
        self.date: datetime.date | None = None
        self.cohorts: list[Cohort] = []
        self.metric: str | BalanceSheetMetric | None = None

        if is_in_identifiers("item", list(rule_input.keys())):
            value = rule_input[get_identifier("item", list(rule_input.keys()))]
            if pd.isna(value) or value == "":
                self.item = BalanceSheetItem()
            else:
                self.item = BalanceSheetItemRegistry.get(value)
        else:
            self.item = BalanceSheetItem()
        if is_in_identifiers("counter item", list(rule_input.keys())):
            value = rule_input[get_identifier("counter item", list(rule_input.keys()))]
            if pd.isna(value) or value == "":
                self.counter_item = None
            else:
                self.counter_item = BalanceSheetItemRegistry.get(value)
        else:
            self.counter_item = None

        for key, value in rule_input.items():
            match strip_identifier(key):
                case _ if value in ["", np.nan, None]:
                    pass
                case "item" | "counteritem" | "amount":
                    pass
                case "metric":
                    if strip_identifier(value) in [
                        "repaymentrate",
                        "repayment",
                        "prepayment",
                        "drawdown",
                        "drawdownrate",
                        "topup",
                        "topuprate",
                    ]:
                        self.metric = strip_identifier(value)
                    else:
                        self.metric = BalanceSheetMetrics.get(value)
                case _ if key.startswith("counter"):
                    label = strip_identifier(key[len("counter") :])
                    if label is not None and is_in_identifiers(label, Config.label_columns()):
                        if self.counter_item is None:
                            self.counter_item = BalanceSheetItem(**{label: value})
                        else:
                            self.counter_item = self.counter_item.add_identifier(label, value)
                    else:
                        raise KeyError(f"{key} not recognized as valid balance sheet label")
                case _ if is_in_identifiers(key, Config.label_columns()):
                    self.item = self.item.add_identifier(key, value)
                case _ if (stripped_key := strip_identifier(key)) is not None and stripped_key.startswith(
                    ("age", "minage", "maxage")
                ):
                    cohort = Cohort.from_string(stripped_key, read_int(value))
                    self.cohorts.append(cohort)
                case "relative":
                    self.relative = read_bool(value)
                case "multiplicative":
                    self.multiplicative = read_bool(value)
                case "offsetliquidity":
                    self.offset_liquidity = read_bool(value)
                case "offsetpnl":
                    self.offset_pnl = read_bool(value)
                case "date":
                    self.date = read_date(value)
                case _:
                    raise KeyError(f"{key} not recognized in BalanceSheetMutationRule")


class BalanceSheetMutationInput(ScenarioInput):
    def __init__(self, excel_inputs: list[ExcelInput]) -> None:
        self.mutation_items: list[BalanceSheetMutationInputItem] = [
            BalanceSheetMutationInputItem(items) for excel_input in excel_inputs for items in excel_input.to_dict_list()
        ]

    def filter_on_date_snapshot(self, increment: TimeIncrement) -> list[BalanceSheetMutationInputItem]:
        return [item for item in self.mutation_items if item.date is None or increment.contains(item.date)]


class CostIncomeInputItem:
    def __init__(self, **rule_input: Any) -> None:
        self.amount: float | None = None
        self.reason = MutationReason(module="Cost/Income")
        self.cashflow_date: datetime.date | None = None
        self.pnl_start: datetime.date | None = None
        self.pnl_end: datetime.date | None = None

        for key, value in rule_input.items():
            match strip_identifier(key):
                case _ if pd.isna(value) or value == "":
                    pass
                case _ if is_in_identifiers(key, Config.cashflow_labels() + Config.pnl_labels()):
                    label = get_identifier(key, Config.cashflow_labels() + Config.pnl_labels())
                    self.reason = self.reason.add_identifier(label, value)
                case "date":
                    self.cashflow_date = read_date(value)
                case "pnlstart":
                    self.pnl_start = read_date(value)
                case "pnlend":
                    self.pnl_end = read_date(value)
                case "amount":
                    self.amount = float(value)
                case _:
                    raise KeyError(f"{key} not recognized in CostIncomeRule")

        if self.cashflow_date is None:
            raise ValueError("Date must be specified in CostIncomeRule")
        if self.amount is None:
            raise ValueError("Amount must be specified in CostIncomeRule")
        if (self.pnl_start is None) != (self.pnl_end is None):
            raise ValueError("Both pnlstart and pnlend must be specified together in CostIncomeRule")

        if self.pnl_start is None:
            self.bs_item = None
        else:
            # At this point we know both pnl_start and pnl_end are not None (validated above)
            assert self.pnl_end is not None
            if self.pnl_start > self.pnl_end:
                raise ValueError("pnlstart must be before or equal to pnlend in CostIncomeRule")


class CostIncomeInput(ScenarioInput):
    def __init__(self, excel_inputs: list[ExcelInput]) -> None:
        self.cost_income_items: list[CostIncomeInputItem] = [
            CostIncomeInputItem(**items) for excel_input in excel_inputs for items in excel_input.to_dict_list()
        ]

    def filter_on_date_snapshot(self, increment: TimeIncrement) -> list[CostIncomeInputItem]:
        return [
            item
            for item in self.cost_income_items
            if item.cashflow_date is None or increment.contains(item.cashflow_date)
        ]
