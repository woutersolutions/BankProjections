import datetime
from typing import Any

import pandas as pd

from bank_projections.app_config import Config
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

        for key, value in rule_input.items():
            match strip_identifier(key):
                case _ if pd.isna(value) or value == "":
                    pass
                case _ if is_in_identifiers(key, Config.cashflow_labels() + Config.pnl_labels()):
                    label = get_identifier(key, Config.cashflow_labels() + Config.pnl_labels())
                    self.reason = self.reason.add_identifier(label, value)
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
                    metrics={"Nominal": abs(self.amount)},
                    origination_date=self.cashflow_date,
                    offset_liquidity=self.reason,
                )
            if increment.overlaps(self.pnl_start, self.pnl_end):
                days_in_period = (self.pnl_end - self.pnl_start).days + 1
                amount_to_recognize = (
                    abs(self.amount) * increment.days_overlap(self.pnl_start, self.pnl_end) / days_in_period
                )
                bs.mutate_metric(
                    bs_item, "Nominal", -amount_to_recognize, offset_pnl=True, reason=self.reason, relative=True
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
                    abs(self.amount) * increment.days_overlap(self.pnl_start, self.pnl_end) / days_in_period
                )
                if increment.from_date < self.pnl_start:
                    bs.add_item(
                        based_on_item=bs_item,
                        labels={},
                        metrics={"Nominal": amount_to_recognize},
                        origination_date=self.cashflow_date,
                        offset_pnl=self.reason,
                    )
                else:
                    bs.mutate_metric(
                        bs_item, "Nominal", amount_to_recognize, offset_pnl=True, reason=self.reason, relative=True
                    )

            if increment.contains(self.cashflow_date):
                bs.mutate_metric(
                    bs_item, "Nominal", -abs(self.amount), offset_liquidity=True, reason=self.reason, relative=True
                )

        else:
            # Cashflow date is within the P&L period (pnl_start < cashflow_date < pnl_end)
            # Type narrowing: pnl_start and pnl_end are both not None
            assert self.pnl_start is not None
            assert self.pnl_end is not None

            if self.amount > 0:
                bs_item = BalanceSheetItem(ItemType="Unpaid revenue")
            else:
                bs_item = BalanceSheetItem(ItemType="Unpaid expenses")

            days_in_period = (self.pnl_end - self.pnl_start).days + 1

            # Before cashflow: accrue unpaid revenue/expense
            if increment.overlaps(self.pnl_start, self.cashflow_date - datetime.timedelta(days=1)):
                amount_to_recognize = (
                    abs(self.amount)
                    * increment.days_overlap(self.pnl_start, self.cashflow_date - datetime.timedelta(days=1))
                    / days_in_period
                )
                if increment.from_date < self.pnl_start:
                    bs.add_item(
                        based_on_item=bs_item,
                        labels={},
                        metrics={"Nominal": amount_to_recognize},
                        origination_date=self.cashflow_date,
                        offset_pnl=self.reason,
                    )
                else:
                    bs.mutate_metric(
                        bs_item, "Nominal", amount_to_recognize, offset_pnl=True, reason=self.reason, relative=True
                    )

            # On cashflow date: settle the accrued amount with cash
            if increment.contains(self.cashflow_date):
                # First, recognize revenue/expense for the cashflow day itself
                daily_amount = abs(self.amount) / days_in_period
                bs.mutate_metric(bs_item, "Nominal", daily_amount, offset_pnl=True, reason=self.reason, relative=True)
                # Then settle the full accrued amount with cash
                bs.mutate_metric(
                    bs_item, "Nominal", -abs(self.amount), offset_liquidity=True, reason=self.reason, relative=True
                )

            # After cashflow but still within P&L period: recognize remaining revenue/expense
            if increment.overlaps(self.cashflow_date + datetime.timedelta(days=1), self.pnl_end):
                amount_to_recognize = (
                    abs(self.amount)
                    * increment.days_overlap(self.cashflow_date + datetime.timedelta(days=1), self.pnl_end)
                    / days_in_period
                )
                if self.amount > 0:
                    post_item = BalanceSheetItem(ItemType="Prepaid revenue")
                else:
                    post_item = BalanceSheetItem(ItemType="Prepaid expenses")

                if increment.contains(self.cashflow_date):
                    # We received/paid more than we've earned/spent, create prepaid item
                    bs.add_item(
                        based_on_item=post_item,
                        labels={},
                        metrics={"Nominal": amount_to_recognize},
                        origination_date=self.cashflow_date,
                        offset_liquidity=self.reason,
                    )
                else:
                    # Amortize the prepaid item
                    bs.mutate_metric(
                        post_item, "Nominal", -amount_to_recognize, offset_pnl=True, reason=self.reason, relative=True
                    )

        bs.validate()

        return bs
