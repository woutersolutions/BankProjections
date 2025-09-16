from abc import ABC, abstractmethod

from bank_projections.financials.balance_sheet import BalanceSheet
from bank_projections.projections.time import TimeIncrement


class Rule(ABC):
    @abstractmethod
    def apply(self, bs: BalanceSheet, increment: TimeIncrement) -> BalanceSheet:
        raise NotImplementedError("Subclasses must implement this method")


class RuleSet(Rule):
    def __init__(self, *rules: Rule | list[Rule]):
        self.rules = []
        for rule in rules:
            if isinstance(rule, list):
                self.rules.extend(rule)
            else:
                self.rules.append(rule)

    def apply(self, bs: BalanceSheet, increment: TimeIncrement) -> BalanceSheet:
        for rule in self.rules:
            bs = rule.apply(bs, increment)
        return bs

    def add(self, rule: Rule, position: int = None):
        if position is None:
            self.rules.append(rule)
        else:
            self.rules.insert(position, rule)
