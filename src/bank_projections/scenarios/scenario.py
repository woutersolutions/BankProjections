from bank_projections.financials.balance_sheet import BalanceSheet
from bank_projections.financials.market_data import MarketData, MarketRates
from bank_projections.projections.rule import Rule
from bank_projections.utils.combine import Combinable
from bank_projections.utils.logging import log_iterator
from bank_projections.utils.time import TimeIncrement


class Scenario(Rule, Combinable):
    def __init__(self, rules: dict[str, Rule] | None = None, market_data: MarketData | None = None):
        self.rules = rules or {}
        self.market_data = market_data or MarketData()

    def apply(self, bs: BalanceSheet, increment: TimeIncrement, market_rates: MarketRates) -> BalanceSheet:
        for _name, rule in log_iterator(self.rules.items()):
            bs = rule.apply(bs, increment, market_rates)
        return bs

    def combine(self, other: "Scenario") -> "Scenario":
        combined_rules = {**self.rules, **other.rules}
        combined_market_data = self.market_data.combine(other.market_data)
        return Scenario(combined_rules, combined_market_data)
