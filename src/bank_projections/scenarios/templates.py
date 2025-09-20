import datetime
import os
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from bank_projections.config import BALANCE_SHEET_LABELS
from bank_projections.financials.balance_sheet import BalanceSheet, MutationReason
from bank_projections.financials.balance_sheet_item import BalanceSheetItem, BalanceSheetItemRegistry
from bank_projections.financials.metrics import BalanceSheetMetrics
from bank_projections.projections.base_registry import BaseRegistry, clean_identifier, get_identifier, is_in_identifiers
from bank_projections.projections.market_data import CurveData, MarketData
from bank_projections.projections.rule import Rule
from bank_projections.projections.time import TimeIncrement
from bank_projections.scenarios.scenario import Scenario
from bank_projections.utils.date import add_months, is_end_of_month


class ScenarioTemplate(ABC):
    @abstractmethod
    def load_excel_sheet(self, file_path: str, sheet_name: str) -> Scenario:
        pass


class CurveTemplate(ScenarioTemplate):
    def load_excel_sheet(self, file_path: str, sheet_name: str) -> Scenario:
        data = pd.read_excel(file_path, sheet_name=sheet_name, header=1)
        curve_data = CurveData(data)
        return Scenario(market_data=MarketData(curve_data))


class MultiHeaderTemplate(ScenarioTemplate):
    def __init__(self, rule_type: type["AmountRuleBase"]):
        self.rule_type = rule_type

    def load_excel_sheet(self, file_path: str, sheet_name: str) -> Scenario:
        df_raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

        # The first row must indicate the template name (later we can have multiple templates)
        if clean_identifier(str(df_raw.iloc[0, 0])) != "template":
            raise ValueError(f"First cell must be 'Template', found {df_raw.iloc[0, 0]}")

        # Find cell with '*' in it
        star_row, star_col = df_raw[df_raw.map(lambda x: isinstance(x, str) and "*" in x)].stack().index[0]

        # Find the first row with non-empty cells from the third column
        col_header_start_row = df_raw.iloc[:, 2:].apply(lambda row: row[2:].notna().any(), axis=1).idxmax()
        assert col_header_start_row <= star_row
        col_headers = (
            df_raw.iloc[col_header_start_row : (star_row + 1), star_col:].set_index(star_col).T.reset_index(drop=True)
        )
        col_headers.columns = [str(col).split("*")[-1] for idx, col in enumerate(col_headers.columns)]

        row_headers = df_raw.iloc[(star_row + 1) :, : (star_col + 1)]
        row_headers.columns = df_raw.iloc[star_row, : (star_col + 1)].values
        row_headers = row_headers.rename(columns={row_headers.columns[-1]: str(row_headers.columns[-1]).split("*")[0]})

        # Read the table
        content = pd.read_excel(
            file_path,
            sheet_name=sheet_name,
            header=None,
            skiprows=star_row + 1,
            usecols=range(star_col + 1, df_raw.shape[1]),
        )

        # Read the tags above the start row (key in A and value in B)
        general_tags = {}
        for idx in range(1, col_header_start_row):
            key = str(df_raw.iloc[idx, 0]).strip()
            value = str(df_raw.iloc[idx, 1]).strip()
            if key and value:
                general_tags[key] = value

        return Scenario(
            rules={sheet_name: MultiHeaderRule(content, col_headers, row_headers, general_tags, self.rule_type)}
        )


class KeyValueTemplate(ScenarioTemplate):
    def __init__(self, rule_type: type["KeyValueRuleBase"]):
        self.rule_type = rule_type

    def load_excel_sheet(self, file_path: str, sheet_name: str) -> Scenario:
        df_raw = pd.read_excel(file_path, usecols="A:B", sheet_name=sheet_name, header=None)
        rule_input = dict(zip(df_raw.iloc[1:, 0], df_raw.iloc[1:, 1], strict=False))
        rule = self.rule_type(rule_input)
        return Scenario(rules={sheet_name: rule})


class KeyValueRuleBase(Rule):
    @abstractmethod
    def __init__(self, rule_input: dict[str, Any]):
        pass


class AuditRule(KeyValueRuleBase):
    def __init__(self, rule_input: dict[str, Any]):
        self.target = BalanceSheetItem()
        for key, value in rule_input.items():
            match clean_identifier(key):
                case "closingmonth":
                    self.closing_month = int(value)
                case "auditmonth":
                    self.audit_month = int(value)
                case _ if clean_identifier(key).startswith("target"):
                    label = clean_identifier(key[len("target") :])
                    self.target = self.target.add_identifier(label, value)
                case _:
                    raise KeyError(f"{key} not recognized in AuditRule")

    def apply(self, bs: BalanceSheet, increment: TimeIncrement, market_rates) -> BalanceSheet:
        # Implement the logic to apply the mutation to the balance sheet based on rule_input
        # This is a placeholder implementation

        # See when (and if) in the increment an audit should be done
        current_date = increment.to_date
        while current_date > increment.from_date:
            if current_date.month == self.audit_month and is_end_of_month(current_date):
                audit_date = current_date
                break
            current_date = add_months(current_date, -1, make_end_of_month=True)
        else:
            # No audit in this increment
            return bs

        closing_date = add_months(current_date, -((self.audit_month - self.closing_month) % 12), make_end_of_month=True)
        item = BalanceSheetItemRegistry.get("pnl account").add_condition(
            (pl.col("OriginationDate") <= closing_date) | pl.col("OriginationDate").is_null()
        )
        counter_item = BalanceSheetItemRegistry.get("Retained earnings")
        reason = MutationReason(module="Audit", rule=f"Audit as of {audit_date}", date=audit_date)
        bs.mutate_metric(
            item, BalanceSheetMetrics.get("quantity"), 0, reason, relative=False, counter_item=counter_item
        )

        return bs


class TaxTemplate(ScenarioTemplate):
    def load_excel_sheet(self, file_path: str, sheet_name: str) -> Scenario:
        df_raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None).set_index(0)

        # TODO: More comprehensive tax rules
        tax_rate = df_raw.iloc[1, 0]

        rule = TaxRule(tax_rate=tax_rate)

        return Scenario(rules={"Tax": rule})


class TaxRule(Rule):
    def __init__(self, tax_rate: float):
        self.tax_rate = tax_rate

    def apply(self, bs: BalanceSheet, increment: TimeIncrement, market_rates) -> BalanceSheet:
        income, expense = bs.pnls.select(
            income=pl.when(pl.col("Amount") < 0).then(pl.col("Amount")).otherwise(0).sum() * -pl.lit(self.tax_rate),
            expense=pl.when(pl.col("Amount") > 0).then(pl.col("Amount")).otherwise(0).sum() * -pl.lit(self.tax_rate),
        ).row(0)

        bs.add_single_pnl(expense, MutationReason(module="Tax", rule="Tax expense"), offset_liquidity=True)
        bs.add_single_pnl(income, MutationReason(module="Tax", rule="Tax benefit"), offset_liquidity=True)

        return bs


class MultiHeaderRule(Rule):
    def __init__(
        self,
        content: pd.DataFrame,
        col_headers: pd.DataFrame,
        row_headers: pd.DataFrame,
        general_tags: dict[str, str],
        rule_type: type["AmountRuleBase"],
    ):
        self.content = content
        self.col_headers = col_headers
        self.row_headers = row_headers
        self.general_tags = general_tags

        self.rule_type = rule_type

    def apply(self, bs: BalanceSheet, increment: TimeIncrement, market_rates) -> BalanceSheet:
        for idx, row in self.content.iterrows():
            for col in range(self.content.shape[1]):
                # Combine the content row, header, and tags into one dictionary
                amount = row.iloc[col]
                col_headers = self.col_headers.iloc[col].to_dict()
                row_headers = self.row_headers.iloc[idx].to_dict()
                rule_input = {**self.general_tags, **col_headers, **row_headers}
                rule = self.rule_type(rule_input, amount)
                bs = rule.apply(bs, increment, market_rates)
        return bs


class AmountRuleBase(Rule):
    @abstractmethod
    def __init__(self, rule_input: dict[str, Any], amount: float):
        pass


class BalanceSheetMutationRule(AmountRuleBase):
    def __init__(self, rule_input: dict[str, Any], amount: float):
        self.amount = amount
        self.relative = True
        self.multiplicative = False
        self.offset_liquidity = False
        self.offset_pnl = False
        self.reason = MutationReason(rule="BalanceSheetMutationRule")
        self.date: datetime.date | None = None

        if is_in_identifiers("item", list(rule_input.keys())):
            value = rule_input[get_identifier("item", list(rule_input.keys()))]
            if value in ["", np.nan, None]:
                self.item = BalanceSheetItem()
            else:
                self.item = BalanceSheetItemRegistry.get(value)
        else:
            self.item = BalanceSheetItem()
        if is_in_identifiers("counter item", list(rule_input.keys())):
            value = rule_input[get_identifier("counter item", list(rule_input.keys()))]
            if value in ["", np.nan, None]:
                self.counter_item = None
            else:
                self.counter_item = BalanceSheetItemRegistry.get(value)
        else:
            self.counter_item = None

        for key, value in rule_input.items():
            match clean_identifier(key):
                case _ if value in ["", np.nan, None]:
                    pass
                case "item" | "counteritem":
                    pass
                case "metric":
                    self.metric = BalanceSheetMetrics.get(value)
                case _ if key.startswith("counter"):
                    label = clean_identifier(key[len("counter") :])
                    if is_in_identifiers(label, BALANCE_SHEET_LABELS):
                        if self.counter_item is None:
                            self.counter_item = BalanceSheetItem(**{label: value})
                        else:
                            self.counter_item = self.counter_item.add_identifier(label, value)
                    else:
                        raise KeyError(f"{key} not recognized as valid balance sheet label")
                case _ if is_in_identifiers(key, BALANCE_SHEET_LABELS):
                    self.item = self.item.add_identifier(key, value)
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

    def apply(self, bs: BalanceSheet, increment: TimeIncrement, market_rates) -> BalanceSheet:
        # Implement the logic to apply the mutation to the balance sheet based on rule_input
        # This is a placeholder implementation

        if self.date is None or increment.contains(self.date):
            bs.mutate_metric(
                self.item,
                self.metric,
                self.amount,
                self.reason,
                self.relative,
                self.multiplicative,
                self.offset_liquidity,
                self.offset_pnl,
                self.counter_item,
            )

        bs.validate()

        return bs


def read_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ["true", "yes", "1"]:
            return True
        elif value in ["false", "no", "0"]:
            return False
    raise ValueError(f"Cannot convert {value} to bool")


def read_date(value: str | datetime.date | datetime.datetime) -> datetime.date:
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    if isinstance(value, str):
        return datetime.datetime.strptime(value, "%Y-%m-%d").date()
    raise ValueError(f"Cannot convert {value} to date")


class TemplateRegistry(BaseRegistry[ScenarioTemplate]):
    @classmethod
    def load_folder(cls, folder_path: str) -> Scenario:
        # Iterate over files in folder and load all Excel files
        scenario_list = []
        for file_name in os.listdir(folder_path):
            # Ignore temporary files
            if file_name.startswith("~$"):
                continue

            scenario = cls.load_file(os.path.join(folder_path, file_name))
            scenario_list.append(scenario)
        return Scenario.combine_list(scenario_list)

    @classmethod
    def load_file(cls, file_path: str) -> Scenario:
        name, extension = os.path.splitext(file_path)

        match extension:
            case ".xlsx" | ".xls":
                return cls.load_excel(file_path)
            case _:
                raise ValueError(f"Unsupported file type: {extension}")

    @classmethod
    def load_excel(cls, file_path: str) -> Scenario:
        xls = pd.ExcelFile(file_path)
        scenario_list = []
        for sheet_name in xls.sheet_names:
            scenario = cls.load_excel_sheet(file_path, sheet_name)
            scenario_list.append(scenario)
        return Scenario.combine_list(scenario_list)

    @classmethod
    def load_excel_sheet(cls, file_path: str, sheet_name: str) -> Scenario:
        template = cls.get_excel_sheet_template(file_path, sheet_name)
        return template.load_excel_sheet(file_path, sheet_name)

    @classmethod
    def get_excel_sheet_template(cls, file_path: str, sheet_name: str) -> ScenarioTemplate:
        # Read the first cell to determine the template type
        df_raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None, nrows=1, usecols=(0, 1))

        if clean_identifier(str(df_raw.iloc[0, 0])) != "template":
            raise ValueError(f"First cell must be 'Template', found {df_raw.iloc[0, 0]}")

        template_name = str(df_raw.iloc[0, 1])
        if is_in_identifiers(template_name, cls.items.keys()):
            return cls.get(template_name)
        else:
            raise ValueError(
                f"Template '{template_name}' not recognized. Available templates: {list(cls.items.keys())}"
            )


TemplateRegistry.register("balancesheetmutations", MultiHeaderTemplate(BalanceSheetMutationRule))
TemplateRegistry.register("interestrates", CurveTemplate())
TemplateRegistry.register("tax", TaxTemplate())
TemplateRegistry.register("audit", KeyValueTemplate(AuditRule))
