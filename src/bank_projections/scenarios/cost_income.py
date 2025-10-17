import datetime
from typing import Any

import pandas as pd
import polars as pl

from bank_projections.config import Config
from bank_projections.financials.balance_sheet import BalanceSheet, MutationReason
from bank_projections.financials.balance_sheet_item import BalanceSheetItem
from bank_projections.financials.market_data import MarketRates
from bank_projections.scenarios.template import AmountRuleBase
from bank_projections.utils.parsing import get_identifier, is_in_identifiers, read_date, strip_identifier
from bank_projections.utils.time import TimeIncrement


class CostIncomeRule(AmountRuleBase):
    def __init__(self, rule_input: dict[str, Any]):
        self.amount: float | None = None
        self.reason = MutationReason(module="Cost/Income")
        self.cashflow_date: datetime.date | None = None
        self.pnl_start: datetime.date | None = None
        self.pnl_end: datetime.date | None = None
        self.rule = ""

        for key, value in rule_input.items():
            match strip_identifier(key):
                case _ if pd.isna(value) or value == "":
                    pass
                case _ if is_in_identifiers(key, Config.CASHFLOW_AGGREGATION_LABELS + Config.PNL_AGGREGATION_LABELS):
                    label = get_identifier(key, Config.CASHFLOW_AGGREGATION_LABELS + Config.PNL_AGGREGATION_LABELS)
                    self.reason = self.reason.add_identifier(label, value)
                case "rule":
                    self.reason = self.reason.add_identifier("rule", value)
                    self.rule = value
                case "date":
                    self.cashflow_date = read_date(value)
                case "pnlstart":
                    self.pnl_start = read_date(value)
                case "pnlend":
                    self.pnl_end = read_date(value)
                case "amount":
                    self.amount = float(value)
                case _:
                    raise KeyError(f"{key} not recognized in CostIncomeRule")

        if self.cashflow_date is None:
            raise ValueError("Date must be specified in CostIncomeRule")
        if self.amount is None:
            raise ValueError("Amount must be specified in CostIncomeRule")
        if (self.pnl_start is None) != (self.pnl_end is None):
            raise ValueError("Both pnlstart and pnlend must be specified together in CostIncomeRule")

        if self.pnl_start is None:
            self.bs_item = None
        else:
            # At this point we know both pnl_start and pnl_end are not None (validated above)
            assert self.pnl_end is not None
            if self.pnl_start > self.pnl_end:
                raise ValueError("pnlstart must be before or equal to pnlend in CostIncomeRule")

    def apply(self, bs: BalanceSheet, increment: TimeIncrement, market_rates: MarketRates) -> BalanceSheet:
        # Type narrowing: after __init__, we know these are not None
        assert self.cashflow_date is not None
        assert self.amount is not None

        if self.pnl_start is None:
            if increment.contains(self.cashflow_date):
                bs.add_single_liquidity(self.amount, self.reason, True)

        elif self.cashflow_date <= self.pnl_start:
            # Type narrowing: pnl_start and pnl_end are both not None
            assert self.pnl_end is not None
            if self.amount > 0:
                bs_item = BalanceSheetItem(ItemType="Prepaid revenue")
            else:
                bs_item = BalanceSheetItem(ItemType="Prepaid expenses")
            if increment.contains(self.cashflow_date):
                bs.add_item(
                    based_on_item=bs_item,
                    labels={},
                    origination_date=self.cashflow_date,
                    metrics={"Quantity": -self.amount},
                    cashflows={self.reason: -pl.col("Quantity")},
                )
            if increment.overlaps(self.pnl_start, self.pnl_end):
                days_in_period = (self.pnl_end - self.pnl_start).days + 1
                amount_to_recognize = (
                    self.amount * increment.days_overlap(self.pnl_start, self.pnl_end) / days_in_period
                )
                bs.mutate_metric(
                    bs_item, "Quantity", amount_to_recognize, offset_pnl=True, reason=self.reason, relative=True
                )

        elif self.pnl_end is not None and self.cashflow_date >= self.pnl_end:
            # Type narrowing: pnl_start and pnl_end are both not None
            assert self.pnl_start is not None
            if self.amount > 0:
                bs_item = BalanceSheetItem(ItemType="Unpaid revenue")
            else:
                bs_item = BalanceSheetItem(ItemType="Unpaid expenses")
            if increment.overlaps(self.pnl_start, self.pnl_end):
                days_in_period = (self.pnl_end - self.pnl_start).days + 1
                amount_to_recognize = (
                    self.amount * increment.days_overlap(self.pnl_start, self.pnl_end) / days_in_period
                )
                if increment.from_date < self.pnl_start:
                    bs.add_item(
                        based_on_item=bs_item,
                        labels={},
                        origination_date=self.cashflow_date,
                        metrics={"Quantity": amount_to_recognize},
                        pnls={self.reason: pl.col("Quantity")},
                    )
                else:
                    bs.mutate_metric(
                        bs_item, "Quantity", amount_to_recognize, offset_pnl=True, reason=self.reason, relative=True
                    )

            if increment.contains(self.cashflow_date):
                bs.mutate_metric(
                    bs_item, "Quantity", -self.amount, offset_liquidity=True, reason=self.reason, relative=True
                )

        else:
            raise NotImplementedError("CostIncomeRule with cashflow within P&L period not implemented")

        bs.validate()

        return bs
