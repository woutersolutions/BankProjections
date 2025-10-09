from abc import ABC, abstractmethod

import polars as pl

from bank_projections.financials.balance_sheet import BalanceSheet
from bank_projections.financials.balance_sheet_item import BalanceSheetItem, BalanceSheetItemRegistry
from bank_projections.financials.metrics import BalanceSheetMetrics
from bank_projections.projections.base_registry import BaseRegistry


def calculate_metrics(bs: BalanceSheet) -> pl.DataFrame:
    metrics = {}
    for name, metric in MetricRegistry.items.items():
        metrics[name] = metric.calculate(bs, metrics)

    return pl.DataFrame(metrics)


class Metric(ABC):
    @abstractmethod
    def calculate(self, bs: BalanceSheet, previous_metrics: dict[str, float]) -> float:
        pass

    def __neg__(self):
        return Multiplied(self, -1)

    def clip(self, lower: float = None, upper: float = None) -> "Metric":
        return Clipped(self, lower, upper)

    def __add__(self, other: "Metric") -> "Metric":
        return Sum([self, other])

    def __sub__(self, other):
        return Sum([self, -other])


class Ratio(Metric):
    def __init__(self, numerator: Metric, denominator: Metric):
        self.numerator = numerator
        self.denominator = denominator

    def calculate(self, bs: BalanceSheet, previous_metrics: dict[str, float]) -> float:
        numerator = self.numerator.calculate(bs, previous_metrics)
        denominator = self.denominator.calculate(bs, previous_metrics)
        return numerator / denominator if denominator != 0 else None


class Sum(Metric):
    def __init__(self, metrics: list[Metric]):
        self.metrics = metrics

    def calculate(self, bs: BalanceSheet, previous_metrics: dict[str, float]) -> float:
        return sum(metric.calculate(bs, previous_metrics) for metric in self.metrics)


class Clipped(Metric):
    def __init__(self, metric: Metric, min_value: float = None, max_value: float = None):
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
    def __init__(self, metric: Metric, factor: float):
        self.metric = metric
        self.factor = factor

    def calculate(self, bs: BalanceSheet, previous_metrics: dict[str, float]) -> float:
        return self.metric.calculate(bs, previous_metrics) * self.factor


class BalanceSheetAggregation(Metric):
    def __init__(self, metric: str, item: BalanceSheetItem = BalanceSheetItem()):
        self.metric = BalanceSheetMetrics.get(metric)
        self.item = item

    def calculate(self, bs: BalanceSheet, previous_metrics: dict[str, float]) -> float:
        return bs.get_amount(self.item, self.metric)


class MRELEligibleLiabilities(Metric):
    def calculate(self, bs: BalanceSheet, previous_metrics: dict[str, float]) -> float:
        item = BalanceSheetItem(
            ItemType="Borrowings", expr=pl.col("MaturityDate") >= pl.lit(bs.date).dt.offset_by("1y")
        )
        return -bs.get_amount(item, BalanceSheetMetrics.get("Book value"))


class MetricRegistry(BaseRegistry[Metric]):
    pass


MetricRegistry.register(
    "Size",
    BalanceSheetAggregation(
        "Book value",
        BalanceSheetItemRegistry.get("Assets"),
    ),
)
MetricRegistry.register(
    "Leverage exposure", BalanceSheetAggregation("Exposure", BalanceSheetItemRegistry.get("Assets"))
)
MetricRegistry.register("TREA", BalanceSheetAggregation("trea", BalanceSheetItemRegistry.get("Assets")))
MetricRegistry.register(
    "CET1 Capital",
    -BalanceSheetAggregation(
        "Book value",
        BalanceSheetItem(ItemType="CET1 capital"),
    )
    - BalanceSheetAggregation("Book value", BalanceSheetItemRegistry.get("pnl account")).clip(upper=0),
)
MetricRegistry.register(
    "Tier 1 Capital",
    MetricRegistry.get("CET1 Capital")
    + BalanceSheetAggregation("Book value", BalanceSheetItem(ItemType="Additional Tier 1 capital")),
)
MetricRegistry.register(
    "Tier 2 Capital",
    BalanceSheetAggregation("Book value", BalanceSheetItem(ItemType="Tier 2 capital")),
)
MetricRegistry.register(
    "Total Capital",
    MetricRegistry.get("Tier 1 Capital") + MetricRegistry.get("Tier 2 Capital"),
)

MetricRegistry.register("CET1 Ratio", Ratio(MetricRegistry.get("CET1 Capital"), MetricRegistry.get("TREA")))
MetricRegistry.register("Tier 1 Ratio", Ratio(MetricRegistry.get("Tier 1 Capital"), MetricRegistry.get("TREA")))
MetricRegistry.register("Total Capital Ratio", Ratio(MetricRegistry.get("Total Capital"), MetricRegistry.get("TREA")))
MetricRegistry.register(
    "Leverage Ratio", Ratio(MetricRegistry.get("Tier 1 Capital"), MetricRegistry.get("Leverage exposure"))
)
MetricRegistry.register(
    "MREL Eligible Liabilities",
    MRELEligibleLiabilities(),
)
MetricRegistry.register(
    "MREL Ratio risk-based",
    Ratio(
        MetricRegistry.get("Total Capital") + MetricRegistry.get("MREL Eligible Liabilities"),
        MetricRegistry.get("TREA"),
    ),
)
MetricRegistry.register(
    "MREL Ratio leverage-based",
    Ratio(
        MetricRegistry.get("Total Capital") + MetricRegistry.get("MREL Eligible Liabilities"),
        MetricRegistry.get("Leverage exposure"),
    ),
)
