import datetime
from abc import ABC, abstractmethod

import pandas as pd

from bank_projections.config import BALANCE_SHEET_LABELS
from bank_projections.financials.balance_sheet import BalanceSheet, BalanceSheetItem, MutationReason
from bank_projections.financials.metrics import BalanceSheetMetrics
from bank_projections.projections.base_registry import clean_identifier, is_in_identifiers
from bank_projections.projections.rule import Rule
from bank_projections.projections.time import TimeIncrement


class ScenarioTemplate(ABC):
    @abstractmethod
    def process_excel(self, file_path: str, sheet_name: str) -> Rule:
        pass


class BalanceSheetMutations(ScenarioTemplate):
    def process_excel(self, file_path: str, sheet_name: str) -> Rule:
        df_raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

        # The first row must indicate the template name (later we can have multiple templates)
        if clean_identifier(df_raw.iloc[0, 0]) != "template":
            raise ValueError(f"First cell must be 'Template', found {df_raw.iloc[0, 0]}")
        if clean_identifier(df_raw.iloc[0, 1]) != "balancesheetmutations":
            raise ValueError(f"First cell must be 'BalanceSheetMutations', found {df_raw.iloc[0, 0]}")

        # Find cell with '*' in it
        star_row, star_col = df_raw[df_raw.map(lambda x: isinstance(x, str) and "*" in x)].stack().index[0]

        col_header_start_row = df_raw[df_raw[2].notnull()].index[0]
        col_headers = df_raw.iloc[col_header_start_row : (star_row + 1), star_col:].set_index(0).T
        col_headers.columns = [str(col).split("*")[-1] for idx, col in enumerate(col_headers.columns)]

        row_headers = df_raw.iloc[(star_row + 1) :, : (star_col + 1)]
        row_headers.columns = df_raw.iloc[star_row, : (star_col + 1)].values
        row_headers = row_headers.rename(columns={row_headers.columns[-1]: row_headers.columns[-1].split("*")[0]})

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
        for idx in range(1, star_row):
            key = str(df_raw.iloc[idx, 0]).strip()
            value = str(df_raw.iloc[idx, 1]).strip()
            if key and value:
                general_tags[key] = value

        return BalanceSheetMutationRuleSet(content, col_headers, row_headers, general_tags)


class BalanceSheetMutationRuleSet(Rule):
    def __init__(
        self, content: pd.DataFrame, col_headers: pd.DataFrame, row_headers: pd.DataFrame, general_tags: dict[str, str]
    ):
        self.content = content
        self.col_headers = col_headers
        self.row_headers = row_headers
        self.general_tags = general_tags

    def apply(self, bs: BalanceSheet, increment: TimeIncrement):
        for idx, row in self.content.iterrows():
            for col in range(self.content.shape[1]):
                # Combine the content row, header, and tags into one dictionary
                amount = row.iloc[col]
                col_headers = self.col_headers.iloc[col].to_dict()
                row_headers = self.row_headers.iloc[idx].to_dict()
                rule_input = {**self.general_tags, **col_headers, **row_headers}
                rule = BalanceSheetMutationRule(rule_input, amount)
                bs = rule.apply(bs, increment)
        return bs


class BalanceSheetMutationRule(Rule):
    def __init__(self, rule_input: dict, amount: float):
        self.amount = amount

        self.item = BalanceSheetItem()
        self.relative = True
        self.multiplicative = False
        self.offset_liquidity = False
        self.offset_pnl = False
        self.reason = MutationReason(rule="BalanceSheetMutationRule")
        self.date: datetime.date | None = None

        for key, value in rule_input.items():
            match clean_identifier(key):
                case "metric":
                    self.metric = BalanceSheetMetrics.get(value)
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

    def apply(self, bs: BalanceSheet, increment: TimeIncrement) -> BalanceSheet:
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
