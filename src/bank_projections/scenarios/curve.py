import pandas as pd

from bank_projections.financials.market_data import CurveData, MarketData
from bank_projections.scenarios.scenario import Scenario
from bank_projections.scenarios.template import ScenarioTemplate


class CurveTemplate(ScenarioTemplate):
    def load_excel_sheet(self, file_path: str, sheet_name: str) -> Scenario:
        data = pd.read_excel(file_path, sheet_name=sheet_name, header=1)
        curve_data = CurveData(data)
        return Scenario(market_data=MarketData(curve_data))
