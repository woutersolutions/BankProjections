import datetime
from abc import ABC, abstractmethod

from bank_projections.financials.balance_sheet import BalanceSheet


class TimeIncrement:
    def __init__(self, from_date: datetime.date, to_date: datetime.date):
        self.from_date = from_date
        self.to_date = to_date

    @property
    def days(self):
        return (self.to_date - self.from_date).days


class Rule(ABC):
    @abstractmethod
    def apply(self, bs: BalanceSheet, increment: TimeIncrement) -> BalanceSheet:
        raise NotImplementedError("Subclasses must implement this method")
