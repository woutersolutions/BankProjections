from loguru import logger

from bank_projections.financials.balance_sheet import BalanceSheet
from bank_projections.projections.rule import Rule
from bank_projections.projections.time import TimeHorizon


class Projection:
    def __init__(self, rules: list[Rule], horizon: TimeHorizon):
        self.rules = rules
        self.horizon = horizon

    def run(self, bs: BalanceSheet) -> list[BalanceSheet]:
        """Run the projection over the defined time horizon."""
        balance_sheets = []

        total_increments = len(self.horizon)

        for i, increment in enumerate(self.horizon, 1):
            logger.info(f"Time increment {i}/{total_increments} - From {increment.from_date} to {increment.to_date}")

            for rule in self.rules:
                bs = rule.apply(bs, increment)

            balance_sheets.append(bs.aggregate(["BalanceSheetSide", "ItemType"]))

        return balance_sheets
