from abc import ABC, abstractmethod

from bank_projections.financials.balance_sheet import BalanceSheet
from bank_projections.financials.market_data import MarketRates
from bank_projections.utils.time import TimeIncrement


class Rule(ABC):
    @abstractmethod
    def apply(self, bs: BalanceSheet, increment: TimeIncrement, market_rates: MarketRates) -> BalanceSheet:
        raise NotImplementedError("Subclasses must implement this method")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
