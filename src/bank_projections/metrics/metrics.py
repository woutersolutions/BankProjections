from abc import ABC, abstractmethod

import polars as pl

from bank_projections.financials.balance_sheet import BalanceSheet
from bank_projections.financials.balance_sheet_item import BalanceSheetItem
from bank_projections.financials.metrics import BalanceSheetMetrics
from bank_projections.projections.base_registry import BaseRegistry


def calculate_metrics(bs: BalanceSheet) -> pl.DataFrame:
    metrics = {}
    for _name, metric in MetricRegistry.items.items():
        metrics[metric.name] = metric.calculate(bs)

    return pl.DataFrame(metrics)


class Metric(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def calculate(self, bs: BalanceSheet) -> float:
        pass


class BalanceSheetAggregation(Metric):
    def __init__(self, name: str, metric: str, item: BalanceSheetItem = BalanceSheetItem()):
        super().__init__(name)
        self.metric = BalanceSheetMetrics.get(metric)
        self.item = item

    def calculate(self, bs: BalanceSheet) -> float:
        return bs.get_amount(self.item, self.metric)


class MetricRegistry(BaseRegistry[Metric]):
    pass


MetricRegistry.register("trea", BalanceSheetAggregation("TREA", "trea"))
MetricRegistry.register(
    "total", BalanceSheetAggregation("Total assets", "book_value", BalanceSheetItem(BalanceSheetSide="Assets"))
)
