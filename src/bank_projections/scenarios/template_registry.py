import os

import pandas as pd

from bank_projections.scenarios.audit import AuditRule
from bank_projections.scenarios.cost_income import CostIncomeRule
from bank_projections.scenarios.curve import CurveTemplate
from bank_projections.scenarios.mutation import BalanceSheetMutationRule
from bank_projections.scenarios.production import ProductionRule
from bank_projections.scenarios.scenario import Scenario
from bank_projections.scenarios.tax import TaxTemplate
from bank_projections.scenarios.template import (
    KeyValueTemplate,
    MultiHeaderTemplate,
    OneHeaderTemplate,
    ScenarioTemplate,
)
from bank_projections.utils.base_registry import BaseRegistry
from bank_projections.utils.parsing import is_in_identifiers, strip_identifier


class TemplateRegistry(BaseRegistry[ScenarioTemplate]):

    @classmethod
    def load_paths(cls, paths: list[str]) -> Scenario:
        scenario_list = []
        for path in paths:
            scenario = cls.load_path(path)
            scenario_list.append(scenario)
        return Scenario.combine_list(scenario_list)

    @classmethod
    def load_path(cls, path: str) -> Scenario:

        if not os.path.exists(path):
            raise FileNotFoundError(f"Path does not exist: {path}")

        if os.path.isdir(path):
            return cls.load_folder(path)

        name, extension = os.path.splitext(path)

        match extension:
            case ".xlsx" | ".xls":
                return cls.load_excel(path)
            case _:
                raise ValueError(f"Unsupported file type: {extension}")

    @classmethod
    def load_folder(cls, folder_path: str) -> Scenario:
        # Iterate over files in folder and load all Excel files
        scenario_list = []
        for file_name in os.listdir(folder_path):
            # Ignore temporary files
            if file_name.startswith("~$"):
                continue

            scenario = cls.load_path(os.path.join(folder_path, file_name))
            scenario_list.append(scenario)
        return Scenario.combine_list(scenario_list)

    @classmethod
    def load_excel(cls, file_path: str) -> Scenario:
        xls = pd.ExcelFile(file_path)
        scenario_list = []
        for sheet_name in xls.sheet_names:
            scenario = cls.load_excel_sheet(file_path, sheet_name)
            scenario_list.append(scenario)
        return Scenario.combine_list(scenario_list)

    @classmethod
    def load_excel_sheet(cls, file_path: str, sheet_name: str | int) -> Scenario:
        sheet_name_str = str(sheet_name) if isinstance(sheet_name, int) else sheet_name
        template = cls.get_excel_sheet_template(file_path, sheet_name_str)
        return template.load_excel_sheet(file_path, sheet_name_str)

    @classmethod
    def get_excel_sheet_template(cls, file_path: str, sheet_name: str) -> ScenarioTemplate:
        # Read the first cell to determine the template type
        df_raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None, nrows=1, usecols=(0, 1))

        if strip_identifier(str(df_raw.iloc[0, 0])) != "template":
            raise ValueError(f"First cell must be 'Template', found {df_raw.iloc[0, 0]}")

        template_name = str(df_raw.iloc[0, 1])
        if is_in_identifiers(template_name, cls.stripped_names()):
            return cls.get(template_name)
        else:
            raise ValueError(f"Template '{template_name}' not recognized. Available templates: {cls.stripped_names()}")


TemplateRegistry.register("balancesheetmutations", MultiHeaderTemplate(BalanceSheetMutationRule))
TemplateRegistry.register("interestrates", CurveTemplate())
TemplateRegistry.register("tax", TaxTemplate())
TemplateRegistry.register("audit", KeyValueTemplate(AuditRule))
TemplateRegistry.register("production", OneHeaderTemplate(ProductionRule))
TemplateRegistry.register("costincome", OneHeaderTemplate(CostIncomeRule))
