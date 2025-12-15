import datetime

from bank_projections.financials.balance_sheet import BalanceSheet
from bank_projections.financials.balance_sheet_item import BalanceSheetItem
from bank_projections.projections.projectionrule import ProjectionRule
from bank_projections.scenarios.scenario import ScenarioSnapShot
from bank_projections.scenarios.scenario_input_type import CostIncomeInputItem
from bank_projections.utils.time import TimeIncrement


class CostIncomeRule(ProjectionRule):
    def apply(self, bs: BalanceSheet, increment: TimeIncrement, scenario: ScenarioSnapShot) -> BalanceSheet:
        for ci_item in scenario.cost_income:
            bs = self.apply_item(bs, increment, ci_item)

        return bs

    def apply_item(self, bs: BalanceSheet, increment: TimeIncrement, ci_item: CostIncomeInputItem) -> BalanceSheet:
        assert ci_item.cashflow_date is not None
        assert ci_item.amount is not None

        if ci_item.pnl_start is None:
            if increment.contains(ci_item.cashflow_date):
                bs.add_single_liquidity(ci_item.amount, ci_item.reason, True)

        elif ci_item.cashflow_date <= ci_item.pnl_start:
            # Type narrowing: pnl_start and pnl_end are both not None
            assert ci_item.pnl_end is not None
            if ci_item.amount > 0:
                bs_item = BalanceSheetItem(ItemType="Prepaid revenue")
            else:
                bs_item = BalanceSheetItem(ItemType="Prepaid expenses")
            if increment.contains(ci_item.cashflow_date):
                bs.add_item(
                    based_on_item=bs_item,
                    labels={},
                    metrics={"Nominal": abs(ci_item.amount)},
                    origination_date=ci_item.cashflow_date,
                    offset_liquidity=ci_item.reason,
                )
            if increment.overlaps(ci_item.pnl_start, ci_item.pnl_end):
                days_in_period = (ci_item.pnl_end - ci_item.pnl_start).days + 1
                amount_to_recognize = (
                    abs(ci_item.amount) * increment.days_overlap(ci_item.pnl_start, ci_item.pnl_end) / days_in_period
                )
                bs.mutate_metric(
                    bs_item, "Nominal", -amount_to_recognize, offset_pnl=True, reason=ci_item.reason, relative=True
                )

        elif ci_item.pnl_end is not None and ci_item.cashflow_date >= ci_item.pnl_end:
            # Type narrowing: pnl_start and pnl_end are both not None
            assert ci_item.pnl_start is not None
            if ci_item.amount > 0:
                bs_item = BalanceSheetItem(ItemType="Unpaid revenue")
            else:
                bs_item = BalanceSheetItem(ItemType="Unpaid expenses")
            if increment.overlaps(ci_item.pnl_start, ci_item.pnl_end):
                days_in_period = (ci_item.pnl_end - ci_item.pnl_start).days + 1
                amount_to_recognize = (
                    abs(ci_item.amount) * increment.days_overlap(ci_item.pnl_start, ci_item.pnl_end) / days_in_period
                )
                if increment.from_date < ci_item.pnl_start:
                    bs.add_item(
                        based_on_item=bs_item,
                        labels={},
                        metrics={"Nominal": amount_to_recognize},
                        origination_date=ci_item.cashflow_date,
                        offset_pnl=ci_item.reason,
                    )
                else:
                    bs.mutate_metric(
                        bs_item, "Nominal", amount_to_recognize, offset_pnl=True, reason=ci_item.reason, relative=True
                    )

            if increment.contains(ci_item.cashflow_date):
                bs.mutate_metric(
                    bs_item,
                    "Nominal",
                    -abs(ci_item.amount),
                    offset_liquidity=True,
                    reason=ci_item.reason,
                    relative=True,
                )

        else:
            # Cashflow date is within the P&L period (pnl_start < cashflow_date < pnl_end)
            # Type narrowing: pnl_start and pnl_end are both not None
            assert ci_item.pnl_start is not None
            assert ci_item.pnl_end is not None

            if ci_item.amount > 0:
                bs_item = BalanceSheetItem(ItemType="Unpaid revenue")
            else:
                bs_item = BalanceSheetItem(ItemType="Unpaid expenses")

            days_in_period = (ci_item.pnl_end - ci_item.pnl_start).days + 1

            # Before cashflow: accrue unpaid revenue/expense
            if increment.overlaps(ci_item.pnl_start, ci_item.cashflow_date - datetime.timedelta(days=1)):
                amount_to_recognize = (
                    abs(ci_item.amount)
                    * increment.days_overlap(ci_item.pnl_start, ci_item.cashflow_date - datetime.timedelta(days=1))
                    / days_in_period
                )
                if increment.from_date < ci_item.pnl_start:
                    bs.add_item(
                        based_on_item=bs_item,
                        labels={},
                        metrics={"Nominal": amount_to_recognize},
                        origination_date=ci_item.cashflow_date,
                        offset_pnl=ci_item.reason,
                    )
                else:
                    bs.mutate_metric(
                        bs_item, "Nominal", amount_to_recognize, offset_pnl=True, reason=ci_item.reason, relative=True
                    )

            # On cashflow date: settle the accrued amount with cash
            if increment.contains(ci_item.cashflow_date):
                # First, recognize revenue/expense for the cashflow day itself
                daily_amount = abs(ci_item.amount) / days_in_period
                bs.mutate_metric(
                    bs_item, "Nominal", daily_amount, offset_pnl=True, reason=ci_item.reason, relative=True
                )
                # Then settle the full accrued amount with cash
                bs.mutate_metric(
                    bs_item,
                    "Nominal",
                    -abs(ci_item.amount),
                    offset_liquidity=True,
                    reason=ci_item.reason,
                    relative=True,
                )

            # After cashflow but still within P&L period: recognize remaining revenue/expense
            if increment.overlaps(ci_item.cashflow_date + datetime.timedelta(days=1), ci_item.pnl_end):
                amount_to_recognize = (
                    abs(ci_item.amount)
                    * increment.days_overlap(ci_item.cashflow_date + datetime.timedelta(days=1), ci_item.pnl_end)
                    / days_in_period
                )
                if ci_item.amount > 0:
                    post_item = BalanceSheetItem(ItemType="Prepaid revenue")
                else:
                    post_item = BalanceSheetItem(ItemType="Prepaid expenses")

                if increment.contains(ci_item.cashflow_date):
                    # We received/paid more than we've earned/spent, create prepaid item
                    bs.add_item(
                        based_on_item=post_item,
                        labels={},
                        metrics={"Nominal": amount_to_recognize},
                        origination_date=ci_item.cashflow_date,
                        offset_liquidity=ci_item.reason,
                    )
                else:
                    # Amortize the prepaid item
                    bs.mutate_metric(
                        post_item,
                        "Nominal",
                        -amount_to_recognize,
                        offset_pnl=True,
                        reason=ci_item.reason,
                        relative=True,
                    )

        bs.validate()

        return bs
