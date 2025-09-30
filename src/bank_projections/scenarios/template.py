from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from bank_projections.financials.balance_sheet import BalanceSheet
from bank_projections.projections.rule import Rule
from bank_projections.projections.time import TimeIncrement
from bank_projections.scenarios.scenario import Scenario
from bank_projections.utils.parsing import strip_identifier


class AmountRuleBase(Rule):
    @abstractmethod
    def __init__(self, rule_input: dict[str, Any], amount: float):
        pass


class ScenarioTemplate(ABC):
    @abstractmethod
    def load_excel_sheet(self, file_path: str, sheet_name: str) -> Scenario:
        pass


class MultiHeaderTemplate(ScenarioTemplate):
    def __init__(self, rule_type: type["AmountRuleBase"]):
        self.rule_type = rule_type

    def load_excel_sheet(self, file_path: str, sheet_name: str) -> Scenario:
        df_raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

        # The first row must indicate the template name (later we can have multiple templates)
        if strip_identifier(str(df_raw.iloc[0, 0])) != "template":
            raise ValueError(f"First cell must be 'Template', found {df_raw.iloc[0, 0]}")

        # Find cell with '*' in it
        star_row, star_col = df_raw[df_raw.map(lambda x: isinstance(x, str) and "*" in x)].stack().index[0]

        # Find the first row with non-empty cells from the third column
        col_header_start_row = df_raw.iloc[:, 2:].apply(lambda row: row.notna().any(), axis=1).idxmax()
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


class OneHeaderTemplate(ScenarioTemplate):
    def __init__(self, rule_type: type["AmountRuleBase"]):
        self.rule_type = rule_type

    def load_excel_sheet(self, file_path: str, sheet_name: str) -> Scenario:
        df_raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

        # The first row must indicate the template name (later we can have multiple templates)
        if strip_identifier(str(df_raw.iloc[0, 0])) != "template":
            raise ValueError(f"First cell must be 'Template', found {df_raw.iloc[0, 0]}")

        # Find the first row with non-empty cells from the third column
        header_start_row = df_raw.iloc[:, 2:].apply(lambda row: row.notna().any(), axis=1).idxmax()

        # Read the table
        content = pd.read_excel(
            file_path,
            sheet_name=sheet_name,
            header=header_start_row,
        )

        # Read the tags above the start row (key in A and value in B)
        general_tags = {}
        for idx in range(1, header_start_row):
            key = str(df_raw.iloc[idx, 0]).strip()
            value = str(df_raw.iloc[idx, 1]).strip()
            if key and value:
                general_tags[key] = value

        return Scenario(rules={sheet_name: OneHeaderRule(content, general_tags, self.rule_type)})


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


class OneHeaderRule(Rule):
    def __init__(
        self,
        content: pd.DataFrame,
        general_tags: dict[str, str],
        rule_type: type["Rule"],
    ):
        self.content = content
        self.general_tags = general_tags

        self.rule_type = rule_type

    def apply(self, bs: BalanceSheet, increment: TimeIncrement, market_rates) -> BalanceSheet:
        for _idx, row in self.content.iterrows():
            rule = self.rule_type({**self.general_tags, **row.to_dict()})
            bs = rule.apply(bs, increment, market_rates)
        return bs
