from dataclasses import dataclass

import polars as pl
from loguru import logger

from bank_projections.financials.balance_sheet import BalanceSheet
from bank_projections.projections.rule import Rule
from bank_projections.projections.time import TimeHorizon


@dataclass
class ProjectionResult:
    balance_sheets: list[pl.DataFrame]
    pnls: list[pl.DataFrame]
    cashflows: list[pl.DataFrame]


class Projection:
    def __init__(self, rule: Rule, horizon: TimeHorizon):
        self.rule = rule
        self.horizon = horizon

    def run(self, bs: BalanceSheet) -> ProjectionResult:
        """Run the projection over the defined time horizon."""
        balance_sheets = []
        pnls_list = []
        cashflows_list = []

        total_increments = len(self.horizon)

        for i, increment in enumerate(self.horizon, 1):
            logger.info(f"Time increment {i}/{total_increments} - From {increment.from_date} to {increment.to_date}")
            bs.clear_mutations()
            bs = self.rule.apply(bs, increment)

            agg_bs, pnls, cashflows = bs.aggregate()
            balance_sheets.append(agg_bs)
            pnls_list.append(pnls)
            cashflows_list.append(cashflows)

            bs.validate()

        return ProjectionResult(balance_sheets, pnls_list, cashflows_list)
