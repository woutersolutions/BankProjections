import datetime
from abc import ABC, abstractmethod

import polars as pl

from bank_projections.financials.balance_sheet import BalanceSheet
from bank_projections.financials.balance_sheet_item import BalanceSheetItem, BalanceSheetItemRegistry
from bank_projections.financials.balance_sheet_metric_registry import BalanceSheetMetrics
from bank_projections.financials.stage import IFRS9StageRegistry
from bank_projections.projections.frequency import FrequencyRegistry, coupon_payment
from bank_projections.projections.redemption import RedemptionRegistry
from bank_projections.utils.base_registry import BaseRegistry


def calculate_metrics(bs: BalanceSheet) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, metric in MetricRegistry.items.items():
        metrics[name] = metric.calculate(bs, metrics)

    return metrics


class Metric(ABC):
    @abstractmethod
    def calculate(self, bs: BalanceSheet, previous_metrics: dict[str, float]) -> float:
        pass

    def __neg__(self) -> "Metric":
        return Multiplied(self, -1)

    def clip(self, lower: float | None = None, upper: float | None = None) -> "Metric":
        return Clipped(self, lower, upper)

    def __add__(self, other: "Metric") -> "Metric":
        return Sum([self, other])

    def __sub__(self, other: "Metric") -> "Metric":
        return Sum([self, -other])


class Ratio(Metric):
    def __init__(self, numerator: Metric, denominator: Metric):
        self.numerator = numerator
        self.denominator = denominator

    def calculate(self, bs: BalanceSheet, previous_metrics: dict[str, float]) -> float:
        numerator = self.numerator.calculate(bs, previous_metrics)
        denominator = self.denominator.calculate(bs, previous_metrics)
        return numerator / denominator if denominator != 0 else 0.0


class Sum(Metric):
    def __init__(self, metrics: list[Metric]):
        self.metrics = metrics

    def calculate(self, bs: BalanceSheet, previous_metrics: dict[str, float]) -> float:
        return sum(metric.calculate(bs, previous_metrics) for metric in self.metrics)


class Clipped(Metric):
    def __init__(self, metric: Metric, min_value: float | None = None, max_value: float | None = None):
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


class ContractualInflowPrincipal(Metric):
    def __init__(self, item: BalanceSheetItem):
        self.item = item

    def calculate(self, bs: BalanceSheet, previous_metrics: dict[str, float]) -> float:
        to_date = bs.date + datetime.timedelta(days=30)

        matured = pl.col("MaturityDate") <= pl.lit(to_date)
        repayment_factors = (
            pl.when(~RedemptionRegistry.has_principal_exchange())
            .then(0.0)
            .when(matured)
            .then(1.0)
            .otherwise(
                RedemptionRegistry.redemption_factor(
                    pl.col("MaturityDate"), pl.col("InterestRate"), pl.col("NextCouponDate"), to_date
                )
            )
        )

        inflow = (
            bs._data.filter(self.item.filter_expression).select((repayment_factors * pl.col("Quantity")).sum()).item()
        )

        return float(inflow)


class ContractualInflowCoupon(Metric):
    def __init__(self, item: BalanceSheetItem):
        self.item = item

    def calculate(self, bs: BalanceSheet, previous_metrics: dict[str, float]) -> float:
        to_date = bs.date + datetime.timedelta(days=30)

        number_of_payments = FrequencyRegistry.number_due(
            pl.col("NextCouponDate"), pl.min_horizontal(pl.col("MaturityDate"), pl.lit(to_date))
        )
        coupon_payments = coupon_payment(pl.col("Quantity"), pl.col("InterestRate")) * number_of_payments

        inflow = bs._data.filter(self.item.filter_expression).select(coupon_payments.sum()).item()

        return float(inflow)


class UnencumberedHQLACapped(Metric):
    def calculate(self, bs: BalanceSheet, previous_metrics: dict[str, float]) -> float:
        unencumbered_hqla = previous_metrics["UnencumberedHQLA"]

        unencumbered_level1 = BalanceSheetAggregation(
            "UnencumberedHQLA", BalanceSheetItemRegistry.get("Assets").add_identifier("HQLAClass", "level1")
        ).calculate(bs, previous_metrics)

        unencumbered_level2a = BalanceSheetAggregation(
            "UnencumberedHQLA", BalanceSheetItemRegistry.get("Assets").add_identifier("HQLAClass", "level2a")
        ).calculate(bs, previous_metrics)

        unencumbered_level2b = BalanceSheetAggregation(
            "UnencumberedHQLA",
            BalanceSheetItemRegistry.get("Assets").add_identifier("HQLAClass", "level2bcorporate")
            | BalanceSheetItemRegistry.get("Assets").add_identifier("HQLAClass", "level2bequity"),
        ).calculate(bs, previous_metrics)

        # Apply Basel III caps: Level 2a capped at 40% of total HQLA after caps
        # Level 2 (2a + 2b) capped at 15% of total HQLA after caps
        level2a_capped = min(unencumbered_level2a, 0.4 * unencumbered_hqla)
        level2b_capped = unencumbered_level2b
        level2_total = min(
            level2a_capped + level2b_capped, 0.15 * (unencumbered_level1 + level2a_capped + level2b_capped)
        )

        return unencumbered_level1 + level2_total


class NetOutflow(Metric):
    def calculate(self, bs: BalanceSheet, previous_metrics: dict[str, float]) -> float:
        return previous_metrics["Outflow"] - min(previous_metrics["Inflow"], 0.75 * previous_metrics["Outflow"])


class MetricRegistry(BaseRegistry[Metric]):
    pass


MetricRegistry.register(
    "Total Assets",
    BalanceSheetAggregation(
        "Book value",
        BalanceSheetItemRegistry.get("Assets"),
    ),
)
MetricRegistry.register(
    "Total Liabilities",
    -BalanceSheetAggregation(
        "Book value",
        BalanceSheetItemRegistry.get("liabilities"),
    ),
)
MetricRegistry.register(
    "Total Equity",
    -BalanceSheetAggregation(
        "Book value",
        BalanceSheetItemRegistry.get("equity"),
    ),
)
MetricRegistry.register(
    "Basel exposure", BalanceSheetAggregation("BaselExposure", BalanceSheetItemRegistry.get("Assets"))
)
MetricRegistry.register(
    "Leverage exposure", BalanceSheetAggregation("LeverageExposure", BalanceSheetItemRegistry.get("Assets"))
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
MetricRegistry.register(
    "Encumbered Assets", BalanceSheetAggregation("Encumbered", BalanceSheetItemRegistry.get("Assets"))
)
MetricRegistry.register(
    "Unencumbered Assets", MetricRegistry.get("Total Assets") - MetricRegistry.get("Encumbered Assets")
)
MetricRegistry.register(
    "Encumbrance Ratio",
    Ratio(
        MetricRegistry.get("Encumbered Assets"),
        MetricRegistry.get("Total Assets"),
    ),
)

MetricRegistry.register("HQLA", BalanceSheetAggregation("HQLA", BalanceSheetItemRegistry.get("Assets")))
MetricRegistry.register(
    "EncumberedHQLA", BalanceSheetAggregation("EncumberedHQLA", BalanceSheetItemRegistry.get("Assets"))
)
MetricRegistry.register(
    "EncumberedHQLALevel1",
    BalanceSheetAggregation(
        "EncumberedHQLA", BalanceSheetItemRegistry.get("Assets").add_identifier("HQLAClass", "level1")
    ),
)
MetricRegistry.register(
    "EncumberedHQLALevel2a",
    BalanceSheetAggregation(
        "EncumberedHQLA", BalanceSheetItemRegistry.get("Assets").add_identifier("HQLAClass", "level2a")
    ),
)
MetricRegistry.register(
    "EncumberedHQLALevel2b",
    BalanceSheetAggregation(
        "EncumberedHQLA",
        BalanceSheetItemRegistry.get("Assets").add_identifier("HQLAClass", "level2bcorporate")
        | BalanceSheetItemRegistry.get("Assets").add_identifier("HQLAClass", "level2bequity"),
    ),
)
MetricRegistry.register(
    "UnencumberedHQLA", BalanceSheetAggregation("UnencumberedHQLA", BalanceSheetItemRegistry.get("Assets"))
)
MetricRegistry.register(
    "Encumbered HQLA Ratio", Ratio(MetricRegistry.get("EncumberedHQLA"), MetricRegistry.get("HQLA"))
)
MetricRegistry.register(
    "Unencumbered HQLA Ratio", Ratio(MetricRegistry.get("UnencumberedHQLA"), MetricRegistry.get("Total Assets"))
)
MetricRegistry.register("HQLA Ratio", Ratio(MetricRegistry.get("HQLA"), MetricRegistry.get("Total Assets")))
MetricRegistry.register("UnencumberedHQLACapped", UnencumberedHQLACapped())
MetricRegistry.register(
    "Required Stable Funding", BalanceSheetAggregation("StableFunding", BalanceSheetItemRegistry.get("Assets"))
)
MetricRegistry.register(
    "Available Stable Funding", -BalanceSheetAggregation("StableFunding", BalanceSheetItemRegistry.get("Funding"))
)
MetricRegistry.register(
    "NSFR",
    Ratio(
        MetricRegistry.get("Available Stable Funding"),
        MetricRegistry.get("Required Stable Funding"),
    ),
)

MetricRegistry.register("Loans", BalanceSheetAggregation("OnBalanceExposure", BalanceSheetItem(ItemType="Loans")))
MetricRegistry.register(
    "Deposits", -BalanceSheetAggregation("Book value", BalanceSheetItem(ItemType="Savings deposits"))
)
MetricRegistry.register(
    "Loan-to-Deposit Ratio",
    Ratio(MetricRegistry.get("Loans"), MetricRegistry.get("Deposits")),
)
MetricRegistry.register(
    "NonPerformingLoans",
    BalanceSheetAggregation(
        "OnBalanceExposure", BalanceSheetItem(ItemType="Loans").add_condition(IFRS9StageRegistry.is_default_expr())
    ),
)
MetricRegistry.register("NPL Ratio", Ratio(MetricRegistry.get("NonPerformingLoans"), MetricRegistry.get("Loans")))

MetricRegistry.register(
    "Contractual principal inflow", ContractualInflowPrincipal(BalanceSheetItemRegistry.get("Assets"))
)
MetricRegistry.register(
    "Contractual principal outflow", -ContractualInflowPrincipal(BalanceSheetItemRegistry.get("Funding"))
)
MetricRegistry.register("Contractual coupon inflow", ContractualInflowCoupon(BalanceSheetItemRegistry.get("Assets")))
MetricRegistry.register("Contractual coupon outflow", -ContractualInflowCoupon(BalanceSheetItemRegistry.get("Funding")))
MetricRegistry.register(
    "Stressed outflow", -BalanceSheetAggregation("Stressed outflow", BalanceSheetItemRegistry.get("Funding"))
)
MetricRegistry.register(
    "Inflow", MetricRegistry.get("Contractual principal inflow") + MetricRegistry.get("Contractual coupon inflow")
)
MetricRegistry.register(
    "Outflow",
    MetricRegistry.get("Contractual principal outflow")
    + MetricRegistry.get("Contractual coupon outflow")
    + MetricRegistry.get("Stressed outflow"),
)
MetricRegistry.register("Net Outflow", NetOutflow())
MetricRegistry.register(
    "LCR",
    Ratio(
        MetricRegistry.get("UnencumberedHQLACapped"),
        MetricRegistry.get("Net Outflow"),
    ),
)
