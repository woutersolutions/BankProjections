from loguru import logger

from bank_projections.financials.balance_sheet import BalanceSheet
from bank_projections.projections.rule import Rule
from bank_projections.projections.time import TimeHorizon

import polars as pl


class Projection:
    def __init__(self, rules: list[Rule], horizon: TimeHorizon):
        self.rules = rules
        self.horizon = horizon

    def run(self, bs: BalanceSheet) -> tuple[list[pl.Dataframe], list[pl.Dataframe], list[pl.Dataframe]]:
        """Run the projection over the defined time horizon."""
        balance_sheets = []
        pnls_list = []
        cashflows_list = []

        total_increments = len(self.horizon)

        for i, increment in enumerate(self.horizon, 1):
            logger.info(f"Time increment {i}/{total_increments} - From {increment.from_date} to {increment.to_date}")
            bs.clear_mutations()

            for rule in self.rules:
                bs = rule.apply(bs, increment)

                agg_bs, pnls, cashflows = bs.aggregate(["BalanceSheetSide", "ItemType"])
                balance_sheets.append(agg_bs)
                pnls_list.append(pnls)
                cashflows_list.append(cashflows)

        return balance_sheets, pnls_list, cashflows_list
