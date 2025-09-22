
import os

import pandas as pd

from bank_projections.projections.base_registry import BaseRegistry
from bank_projections.scenarios.audit import AuditRule
from bank_projections.scenarios.curve import CurveTemplate
from bank_projections.scenarios.mutation import BalanceSheetMutationRule
from bank_projections.scenarios.scenario import Scenario
from bank_projections.scenarios.tax import TaxTemplate
from bank_projections.scenarios.template import (
    KeyValueTemplate,
    MultiHeaderTemplate,
    ScenarioTemplate,
)
from bank_projections.utils.parsing import clean_identifier, is_in_identifiers


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
