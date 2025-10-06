import datetime
from typing import Any

import pandas as pd

from bank_projections.config import Config
from bank_projections.financials.balance_sheet import BalanceSheet, MutationReason
from bank_projections.projections.market_data import MarketRates
from bank_projections.projections.time import TimeIncrement
from bank_projections.scenarios.template import AmountRuleBase
from bank_projections.utils.parsing import get_identifier, is_in_identifiers, read_date, strip_identifier


class CostIncomeRule(AmountRuleBase):
    def __init__(self, rule_input: dict[str, Any]):
        self.amount: float | None = None
        self.reason = MutationReason(module="Cost/Income")
        self.date: datetime.date | None = None

        for key, value in rule_input.items():
            match strip_identifier(key):
                case _ if pd.isna(value) or value == "":
                    pass
                case _ if is_in_identifiers(key, Config.CASHFLOW_AGGREGATION_LABELS + Config.PNL_AGGREGATION_LABELS):
                    label = get_identifier(key, Config.CASHFLOW_AGGREGATION_LABELS + Config.PNL_AGGREGATION_LABELS)
                    self.reason = self.reason.add_identifier(label, value)
                case "rule":
                    self.reason = self.reason.add_identifier("rule", value)
                case "date":
                    self.date = read_date(value)
                case "amount":
                    self.amount = float(value)
                case _:
                    raise KeyError(f"{key} not recognized in CostIncomeRule")

        if self.date is None:
            raise ValueError("Date must be specified in CostIncomeRule")
        if self.amount is None:
            raise ValueError("Amount must be specified in CostIncomeRule")

    def apply(self, bs: BalanceSheet, increment: TimeIncrement, market_rates: MarketRates) -> BalanceSheet:
        if not increment.contains(self.date):
            return bs

        bs.add_single_liquidity(self.amount, self.reason, True)


        return bs
