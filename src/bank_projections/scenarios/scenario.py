from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from bank_projections.financials.market_data import Curves
from bank_projections.scenarios.excel_sheet_format import ExcelInput
from bank_projections.scenarios.scenario_input_type import (
    AuditInput,
    BalanceSheetMutationInput,
    BalanceSheetMutationInputItem,
    CostIncomeInput,
    CostIncomeInputItem,
    CurveInput,
    ProductionInput,
    ProductionInputItem,
    TaxInput,
)
from bank_projections.utils.time import TimeHorizonConfig, TimeIncrement


@dataclass
class ScenarioSnapShot:
    curves: Curves
    tax: TaxInput
    audit: AuditInput
    production: list[ProductionInputItem]
    mutations: list[BalanceSheetMutationInputItem]
    cost_income: list[CostIncomeInputItem]


class Scenario:
    def __init__(self, excel_inputs: list[ExcelInput]) -> None:
        self.scenario_input = defaultdict(list)
        for excel_input in excel_inputs:
            self.scenario_input[excel_input.template_name].append(excel_input)

    def add_input(self, key: str, value: Any) -> None:
        self.scenario_input[key].append(value)

    def get_input(self, key: str) -> Any:
        return self.scenario_input[key]

    def snapshot_at(self, increment: TimeIncrement) -> ScenarioSnapShot:
        return ScenarioSnapShot(
            curves=CurveInput(self.scenario_input["interestrates"]).filter_on_date_snapshot(increment),
            tax=TaxInput(self.scenario_input["tax"]).filter_on_date_snapshot(increment),
            audit=AuditInput(self.scenario_input["audit"]).filter_on_date_snapshot(increment),
            production=ProductionInput(self.scenario_input["production"]).filter_on_date_snapshot(increment),
            mutations=BalanceSheetMutationInput(self.scenario_input["balancesheetmutations"]).filter_on_date_snapshot(
                increment
            ),
            cost_income=CostIncomeInput(self.scenario_input["costincome"]).filter_on_date_snapshot(increment),
        )


class ScenarioConfig(BaseModel):
    input_paths: list[str]
    time_horizon: TimeHorizonConfig
