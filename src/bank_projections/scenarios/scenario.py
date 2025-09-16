import os

import pandas as pd
from loguru import logger

from bank_projections.financials.balance_sheet import BalanceSheet
from bank_projections.projections.rule import Rule
from bank_projections.projections.time import TimeIncrement
from bank_projections.scenarios.templates import TemplateRegistry


class Scenario(Rule):
    def __init__(self, rules: dict[str, Rule]):
        self.rules = rules

    def apply(self, bs: BalanceSheet, increment: TimeIncrement) -> BalanceSheet:
        for name, rule in self.rules.items():
            logger.info(f"Applying {name}")
            bs = rule.apply(bs, increment)
        return bs

    @classmethod
    def from_folder(cls, folder_path: str) -> "Scenario":
        # Iterate over files in folder and load all Excel files
        rules = {}
        for file_name in os.listdir(folder_path):
            rules[file_name] = cls.from_file(os.path.join(folder_path, file_name))
        return Scenario(rules)

    @classmethod
    def from_file(cls, file_path: str) -> "Scenario":
        name, extension = os.path.splitext(file_path)

        match extension:
            case ".xlsx" | ".xls":
                return cls.from_excel(file_path)
            case _:
                raise ValueError(f"Unsupported file type: {extension}")

    @classmethod
    def from_excel(cls, file_path: str) -> "Scenario":
        xls = pd.ExcelFile(file_path)
        rules = {}
        for sheet_name in xls.sheet_names:
            rules[sheet_name] = TemplateRegistry.load_excel_sheet(file_path, sheet_name)
        return cls(rules)
