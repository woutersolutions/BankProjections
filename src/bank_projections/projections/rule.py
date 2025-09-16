from abc import ABC, abstractmethod

from bank_projections.financials.balance_sheet import BalanceSheet
from bank_projections.projections.time import TimeIncrement


class Rule(ABC):
    @abstractmethod
    def apply(self, bs: BalanceSheet, increment: TimeIncrement) -> BalanceSheet:
        raise NotImplementedError("Subclasses must implement this method")
