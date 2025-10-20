import datetime
import os
import time
from dataclasses import dataclass
from typing import Any

import polars as pl
import xlsxwriter
from loguru import logger

from bank_projections.financials.balance_sheet import BalanceSheet
from bank_projections.metrics.metrics import calculate_metrics
from bank_projections.scenarios.scenario import Scenario
from bank_projections.utils.logging import log_iterator
from bank_projections.utils.time import TimeHorizon


@dataclass
class ProjectionResult:
    balance_sheets: list[pl.DataFrame]
    pnls: list[pl.DataFrame]
    cashflows: list[pl.DataFrame]
    ocis: list[pl.DataFrame]
    metric_list: list[pl.DataFrame]
    run_info: dict[str, Any]

    def to_dict(self) -> dict[str, pl.DataFrame]:
        return {
            "BalanceSheets": pl.concat(self.balance_sheets, how="diagonal"),
            "P&Ls": pl.concat(self.pnls, how="diagonal"),
            "Cashflows": pl.concat(self.cashflows, how="diagonal"),
            "OCIs": pl.concat(self.ocis, how="diagonal"),
            "Metrics": pl.concat(self.metric_list, how="diagonal"),
            "RunInfo": pl.DataFrame(self.run_info),
        }

    def to_excel(self, file_path: str, open_after: bool = False) -> None:
        date_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = file_path.replace(".xlsx", f"_{date_tag}.xlsx")

        with xlsxwriter.Workbook(file_path) as workbook:
            for name, df in self.to_dict().items():
                logger.info("Writing {name} to {file_path}", name=name, file_path=file_path)
                df.write_excel(workbook=workbook, worksheet=name)

        if open_after:
            logger.info("Opening {file_path}", file_path=file_path)
            os.startfile(file_path)


class Projection:
    def __init__(self, scenarios: dict[str, Scenario], horizon: TimeHorizon):
        self.scenarios = scenarios
        self.horizon = horizon

    def run(self, start_bs: BalanceSheet) -> ProjectionResult:
        """Run the projection over the defined time horizon."""

        start_time = time.time()

        balance_sheets = []
        pnls_list = []
        cashflows_list = []
        metric_list = []
        oci_list = []

        total_increments = len(self.horizon)

        start_bs_size = len(start_bs)

        for scenario_name, scenario in log_iterator(self.scenarios.items(), prefix="Scenario "):
            bs = start_bs.copy()

            for _i, increment in log_iterator(
                enumerate(self.horizon, 1), prefix="Time step ", suffix=f"/{total_increments}", timed=True
            ):
                bs = bs.initialize_new_date(increment.to_date)
                market_rates = scenario.market_data.get_market_rates(increment.to_date)
                bs = scenario.apply(bs, increment, market_rates)

                metrics = calculate_metrics(bs)

                agg_bs, pnls, cashflows, ocis = bs.aggregate()
                for df in [agg_bs, pnls, cashflows, ocis]:
                    df.insert_column(0, pl.lit(scenario_name).alias("Scenario"))
                    df.insert_column(1, pl.lit(increment.to_date).alias("ProjectionDate"))
                balance_sheets.append(agg_bs)
                pnls_list.append(pnls)
                cashflows_list.append(cashflows)
                metric_list.append(metrics)
                oci_list.append(ocis)

                bs.validate()

        run_info: dict[str, Any] = {
            "StartDate": self.horizon.start_date,
            "EndDate": self.horizon.end_date,
            "NumberOfIncrements": total_increments,
            "Starttime": datetime.datetime.fromtimestamp(start_time),
            "Endtime": datetime.datetime.now(),
            "TotalRunTimeSeconds": time.time() - start_time,
            "StartBalanceSheetSize": start_bs_size,
            "Scenarios": len(self.scenarios),
        }

        return ProjectionResult(balance_sheets, pnls_list, cashflows_list, oci_list, metric_list, run_info)
