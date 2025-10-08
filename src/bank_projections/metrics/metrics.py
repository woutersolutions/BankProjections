from abc import ABC, abstractmethod

import polars as pl

from bank_projections.financials.balance_sheet import BalanceSheet
from bank_projections.financials.balance_sheet_item import BalanceSheetItem, BalanceSheetItemRegistry
from bank_projections.financials.metrics import BalanceSheetMetrics
from bank_projections.projections.base_registry import BaseRegistry


def calculate_metrics(bs: BalanceSheet) -> pl.DataFrame:
    metrics = {}
    for _name, metric in MetricRegistry.items.items():
        metrics[metric.name] = metric.calculate(bs, metrics)

    return pl.DataFrame(metrics)


class Metric(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def calculate(self, bs: BalanceSheet, previous_metrics: dict[str, float]) -> float:
        pass

    def __neg__(self):
        return Multiplied(f"-{self.name}", self, -1)

class Ratio(Metric):
    def __init__(self, name: str, numerator: Metric, denominator: Metric):
        super().__init__(name)
        self.numerator = numerator
        self.denominator = denominator

    def calculate(self, bs: BalanceSheet, previous_metrics: dict[str, float]) -> float:

        numerator = self.numerator.calculate(bs, previous_metrics)
        denominator = self.denominator.calculate(bs, previous_metrics)
        return numerator / denominator if denominator != 0 else None

class Sum(Metric):
    def __init__(self, name: str, metrics: list[Metric]):
        super().__init__(name)
        self.metrics = metrics

    def calculate(self, bs: BalanceSheet, previous_metrics: dict[str, float]) -> float:
        return sum(metric.calculate(bs, previous_metrics) for metric in self.metrics)

class Clipped(Metric):
    def __init__(self, name: str, metric: Metric, min_value: float = None, max_value: float = None):
        super().__init__(name)
        self.metric = metric
        self.min_value = min_value
        self.max_value = max_value

    def calculate(self, bs: BalanceSheet, previous_metrics: dict[str, float]) -> float:
        value = self.metric.calculate(bs, previous_metrics)
        if self.min_value is not None:
            value = max(value, self.min_value)
        if self.max_value is not None:
            value = min(value, self.max_value)
        return value

class Multiplied(Metric):
    def __init__(self, name: str, metric: Metric, factor: float):
        super().__init__(name)
        self.metric = metric
        self.factor = factor

    def calculate(self, bs: BalanceSheet, previous_metrics: dict[str, float]) -> float:
        return self.metric.calculate(bs, previous_metrics) * self.factor

class BalanceSheetAggregation(Metric):
    def __init__(self, name: str, metric: str, item: BalanceSheetItem = BalanceSheetItem()):
        super().__init__(name)
        self.metric = BalanceSheetMetrics.get(metric)
        self.item = item

    def calculate(self, bs: BalanceSheet, previous_metrics: dict[str, float]) -> float:
        return bs.get_amount(self.item, self.metric)


class MetricRegistry(BaseRegistry[Metric]):
    pass


MetricRegistry.register("trea", BalanceSheetAggregation("TREA", "trea"))
MetricRegistry.register(
    "size",
    BalanceSheetAggregation(
        "Size",
        "Book value",
        BalanceSheetItemRegistry.get("Assets")
        | (BalanceSheetItemRegistry.get("Derivatives") & BalanceSheetItemRegistry.get("Positive")),
    ),
)
MetricRegistry.register(
    "cet1",
    -BalanceSheetAggregation(
        "CET1 capital",
        "Book value",
        BalanceSheetItem(ItemType="CET1 capital"),
    ),
)