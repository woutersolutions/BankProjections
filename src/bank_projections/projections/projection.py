from loguru import logger

from bank_projections.financials.balance_sheet import BalanceSheet
from bank_projections.projections.rule import Rule
from bank_projections.projections.time import TimeHorizon


class Projection:
    def __init__(self, bs_start: BalanceSheet, rules: list[Rule], horizon: TimeHorizon):
        self.bs_start = bs_start
        self.rules = rules
        self.horizon = horizon

    def run(self) -> list[BalanceSheet]:
        """Run the projection over the defined time horizon."""
        balance_sheets = [self.bs_start]

        total_increments = len(self.horizon)

        for i, increment in enumerate(self.horizon, 1):
            logger.info(f"Time increment {i}/{total_increments} - From {increment.from_date} to {increment.to_date}")
            projected_bs = balance_sheets[-1].copy()

            for rule in self.rules:
                projected_bs = rule.apply(projected_bs, increment)

            balance_sheets.append(projected_bs)

        return balance_sheets
