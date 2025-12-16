import datetime
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import polars as pl
import xlsxwriter
from loguru import logger

from bank_projections.financials.balance_sheet import BalanceSheet
from bank_projections.metrics.metrics import calculate_metrics
from bank_projections.metrics.profitability import calculate_profitability
from bank_projections.output_config import AggregationConfig
from bank_projections.projections.projectionrule import ProjectionRule
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
    profitability_list: list[pl.DataFrame]

    run_info: dict[str, Any]

    def to_dict(self) -> dict[str, pl.DataFrame]:
        return {
            "BalanceSheets": pl.concat(self.balance_sheets, how="diagonal"),
            "P&Ls": pl.concat(self.pnls, how="diagonal"),
            "Cashflows": pl.concat(self.cashflows, how="diagonal"),
            "OCIs": pl.concat(self.ocis, how="diagonal"),
            "Metrics": pl.concat(self.metric_list, how="diagonal"),
            "Profitability": pl.concat(self.profitability_list, how="diagonal"),
            "RunInfo": pl.DataFrame(self.run_info),
        }

    def to_excel(self, file_path: str, open_after: bool = False) -> None:
        file_path = datetime.datetime.now().strftime(file_path)

        with xlsxwriter.Workbook(file_path) as workbook:
            for name, df in self.to_dict().items():
                logger.info("Writing {name} to {file_path}", name=name, file_path=file_path)
                df.write_excel(workbook=workbook, worksheet=name)

        if open_after:
            logger.info("Opening {file_path}", file_path=file_path)
            os.startfile(file_path)


class Projection:
    def __init__(self, scenarios: dict[str, Scenario], horizon: TimeHorizon, rules: dict[str, ProjectionRule]):
        self.scenarios = scenarios
        self.horizon = horizon
        self.rules = rules

    def run(
        self,
        start_bs: BalanceSheet,
        aggregation_config: AggregationConfig = AggregationConfig(),
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> ProjectionResult:
        """Run the projection over the defined time horizon.

        Parameters
        ----------
        start_bs : BalanceSheet
            Starting balance sheet
        progress_callback : Callable[[int, int], None] | None
            Optional callback function for progress updates.
            Called with (current_step, total_steps) after each time increment.

        Returns
        -------
        ProjectionResult
            Results of the projection including balance sheets, P&Ls, cashflows, and metrics
        """

        start_time = time.time()

        balance_sheets = []
        pnls_list = []
        cashflows_list = []
        metric_dicts = []
        metric_list = []
        profitability_list = []
        oci_list = []

        total_increments = len(self.horizon)
        total_steps = len(self.scenarios) * total_increments * len(self.rules)
        current_step = 0

        start_bs_size = len(start_bs)

        for scenario_name, scenario in log_iterator(
            self.scenarios.items(), prefix="Scenario ", item_name=lambda x: x[0], timed=True
        ):
            bs = start_bs.copy()

            for increment in log_iterator(self.horizon, prefix="Time step ", timed=True):
                bs = bs.initialize_new_date(increment.to_date)

                scenario_snapshot = scenario.snapshot_at(increment)

                for rule_name, rule in self.rules.items():
                    bs = rule.apply(bs, increment, scenario_snapshot)

                    # Update progress
                    current_step += 1
                    if progress_callback is not None:
                        progress_callback(current_step, total_steps)

                metrics_dict = calculate_metrics(bs)
                metrics_df = pl.DataFrame(metrics_dict)

                agg_bs, pnls, cashflows, ocis = bs.aggregate(aggregation_config)
                balance_sheets.append(agg_bs)
                pnls_list.append(pnls)
                cashflows_list.append(cashflows)
                metric_dicts.append(metrics_dict)
                metric_list.append(metrics_df)
                oci_list.append(ocis)

                for df in [agg_bs, pnls, cashflows, ocis, metrics_df]:
                    df.insert_column(0, pl.lit(scenario_name).alias("Scenario"))
                    df.insert_column(1, pl.lit(increment.to_date).alias("ProjectionDate"))

                profitability_dicts = calculate_profitability(metric_dicts, pnls_list, self.horizon)
                profitability = pl.DataFrame(profitability_dicts)
                profitability_list.append(profitability)

                for df in [profitability]:
                    if len(df) > 0:
                        df.insert_column(0, pl.lit(scenario_name).alias("Scenario"))
                        df.insert_column(1, pl.lit(increment.to_date).alias("ProjectionDate"))

        run_info: dict[str, Any] = {
            "StartDate": self.horizon.start_date,
            "EndDate": self.horizon.end_date,
            "NumberOfIncrements": total_increments,
            "StartTime": datetime.datetime.fromtimestamp(start_time),
            "Endtime": datetime.datetime.now(),
            "TotalRunTimeSeconds": time.time() - start_time,
            "StartBalanceSheetSize": start_bs_size,
            "Scenarios": len(self.scenarios),
        }

        return ProjectionResult(
            balance_sheets, pnls_list, cashflows_list, oci_list, metric_list, profitability_list, run_info
        )
