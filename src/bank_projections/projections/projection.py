import datetime
from dataclasses import dataclass

import polars as pl
import xlsxwriter
from loguru import logger

from bank_projections.financials.balance_sheet import BalanceSheet
from bank_projections.projections.time import TimeHorizon
from bank_projections.scenarios.scenario import Scenario


@dataclass
class ProjectionResult:
    balance_sheets: list[pl.DataFrame]
    pnls: list[pl.DataFrame]
    cashflows: list[pl.DataFrame]
    horizon: TimeHorizon

    def to_dict(self):
        return {
            "BalanceSheets": pl.concat(
                [
                    self.balance_sheets[i].with_columns(ProjectionDate=increment.to_date)
                    for i, increment in enumerate(self.horizon)
                ],
                how="diagonal",
            ),
            "P&Ls": pl.concat(
                [
                    self.pnls[i].with_columns(ProjectionDate=increment.to_date)
                    for i, increment in enumerate(self.horizon)
                ],
                how="diagonal",
            ),
            "Cashflows": pl.concat(
                [
                    self.cashflows[i].with_columns(ProjectionDate=increment.to_date)
                    for i, increment in enumerate(self.horizon)
                ],
                how="diagonal",
            ),
        }

    def to_excel(self, file_path: str):
        date_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = file_path.replace(".xlsx", f"_{date_tag}.xlsx")

        with xlsxwriter.Workbook(file_path) as workbook:
            for name, df in self.to_dict().items():
                logger.info("Writing {name} to {file_path}", name=name, file_path=file_path)
                df.write_excel(workbook=workbook, worksheet=name)


class Projection:
    def __init__(self, scenario: Scenario, horizon: TimeHorizon):
        self.scenario = scenario
        self.horizon = horizon

    def run(self, bs: BalanceSheet) -> ProjectionResult:
        """Run the projection over the defined time horizon."""
        balance_sheets = []
        pnls_list = []
        cashflows_list = []

        total_increments = len(self.horizon)

        for i, increment in enumerate(self.horizon, 1):
            logger.info(f"Time increment {i}/{total_increments} - From {increment.from_date} to {increment.to_date}")
            bs = bs.initialize_new_date(increment.to_date)
            market_rates = self.scenario.market_data.get_market_rates(increment.to_date)
            bs = self.scenario.apply(bs, increment, market_rates)

            agg_bs, pnls, cashflows = bs.aggregate()
            balance_sheets.append(agg_bs)
            pnls_list.append(pnls)
            cashflows_list.append(cashflows)

            bs.validate()

        return ProjectionResult(balance_sheets, pnls_list, cashflows_list, self.horizon)
