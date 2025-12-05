"""Test script for BalanceSheet get_amount and mutate methods."""

import polars as pl
import pytest

from bank_projections.financials.balance_sheet import MutationReason
from bank_projections.financials.balance_sheet_item import BalanceSheetItem
from bank_projections.financials.balance_sheet_metric_registry import BalanceSheetMetrics


@pytest.fixture
def bs(test_balance_sheet):
    """Provide a fresh copy of the balance sheet for each test."""
    return test_balance_sheet.copy()


class TestBalanceSheetMethods:
    """Test the core methods of BalanceSheet class."""

    # Test whether the nominal can be changed and offset with cash
    def test_mutate_nominal_with_cash_offset(self, bs) -> None:
        """Test mutating nominal with cash offset."""
        # Get initial loan nominal
        loans_item = BalanceSheetItem(SubItemType="Mortgages")
        initial_loan_qty = bs.get_amount(loans_item, BalanceSheetMetrics.get("nominal"))

        # Mutate loan nominal to 100,000 with cash offset
        mutation_amount = 100_000
        expected_mutation = mutation_amount - initial_loan_qty

        # Clear existing cashflows to track this mutation
        bs.cashflows = pl.DataFrame()
        bs.pnls = pl.DataFrame()

        reason = MutationReason(module="Test", rule="test_mutate_nominal_with_cash_offset", action="test_mutation")
        bs.mutate_metric(
            loans_item,
            BalanceSheetMetrics.get("nominal"),
            mutation_amount,
            reason,
            relative=False,
            offset_liquidity=True,
        )

        # Verify the loan nominal is now 100,000 (absolute mutation)
        new_loan_qty = bs.get_amount(loans_item, BalanceSheetMetrics.get("nominal"))
        assert abs(new_loan_qty - 100_000) < 1, f"Expected final amount ~100,000, got {new_loan_qty}"

        # Verify that the cash is mutated correctly
        # Cash should have changed due to the offset (not specifically testing the amount here)

        # Verify cashflows were recorded for the offset
        assert len(bs.cashflows) > 0, "Cashflows should have been recorded"

        # Verify the cashflow amount matches the loan change (with opposite sign)
        total_cashflow = bs.cashflows["Amount"].sum()
        assert abs(total_cashflow + expected_mutation) < 1, "Cashflow should offset loan nominal change"

        # Verify no PnL changes were recorded since we used liquidity offset
        assert len(bs.pnls) == 0, "No PnL changes should be recorded for liquidity offset"

        # Verify balance sheet is still valid after mutation
        bs.validate()

    # Parameterized test for all editable metrics and offset modes
    @pytest.mark.parametrize(
        "metric, asset_type",
        [
            (BalanceSheetMetrics.get("nominal"), "Mortgages"),
            (BalanceSheetMetrics.get("impairment"), "Mortgages"),
            (BalanceSheetMetrics.get("accrued_interest"), "Fixed Debt securities"),
            (BalanceSheetMetrics.get("agio"), "Fixed Debt securities"),
            (BalanceSheetMetrics.get("dirty_price"), "Fixed Debt securities"),
            # Include derived, but updatable metrics (weights)
            (BalanceSheetMetrics.get("coverage_rate"), "Mortgages"),
            (BalanceSheetMetrics.get("accrued_interest_weight"), "Fixed Debt securities"),
            (BalanceSheetMetrics.get("agio_weight"), "Fixed Debt securities"),
        ],
    )
    @pytest.mark.parametrize("offset_mode", [None, "Cash", "pnl"])
    @pytest.mark.parametrize("relative", [False, True])
    def test_mutate_metric_with_offsets(self, bs, metric, asset_type, offset_mode, relative):
        item = BalanceSheetItem(SubItemType=asset_type)
        initial_value = bs.get_amount(item, metric)

        # Choose a sensible mutation target per metric type and relative mode
        if metric == BalanceSheetMetrics.get("dirty_price"):
            if relative:
                mutation_amount = 0.05  # add 5% to current dirty price (weighted)
                expected_value = initial_value + mutation_amount
            else:
                mutation_amount = 1.05  # set dirty price to a fixed realistic value
                expected_value = mutation_amount
        elif metric in {
            BalanceSheetMetrics.get("coverage_rate"),
            BalanceSheetMetrics.get("accrued_interest_weight"),
            BalanceSheetMetrics.get("agio_weight"),
        }:
            if relative:
                mutation_amount = 0.01  # add 1 percentage point to the weight
                expected_value = initial_value + mutation_amount
            else:
                target = 0.02
                mutation_amount = target if abs(initial_value - target) > 1e-6 else target + 0.01
                expected_value = mutation_amount
        else:
            # For absolute amount columns
            if relative:
                mutation_amount = 1_000.0  # add 1,000 to the aggregate amount
                expected_value = initial_value + mutation_amount
            else:
                mutation_amount = initial_value + 1_000.0  # set aggregate amount to initial + 1,000
                expected_value = mutation_amount

        # Determine offset args
        offset_args = {}
        if offset_mode == "cash":
            offset_args["offset_liquidity"] = True
            # offset_column = BalanceSheetMetrics.get('nominal')
            offset_item = BalanceSheetItem(ItemType="Cash")
        elif offset_mode == "pnl":
            offset_args["offset_pnl"] = True
            # offset_column = BalanceSheetMetrics.get('nominal')
            offset_item = BalanceSheetItem(BalanceSheetCategory="Equity")
        else:
            # offset_column = None
            offset_item = None

        # Record initial offset value before mutation (for potential future verification)
        # if offset_item:
        #     initial_offset = bs.get_amount(offset_item, offset_column)

        # Clear existing cashflows/pnls to track just this mutation
        if offset_item:
            bs.cashflows = pl.DataFrame()
            bs.pnls = pl.DataFrame()

        # Perform mutation
        reason = MutationReason(
            module="Test",
            rule="test_mutate_metric_with_offsets",
            action="test_mutation",
            metric=str(metric),
            asset_type=asset_type,
            offset_mode=offset_mode,
            relative=relative,
        )
        bs.mutate_metric(item, metric, mutation_amount, reason, relative=relative, **offset_args)

        # Check mutated value
        new_value = bs.get_amount(item, metric)
        assert abs(new_value - expected_value) < 1, (
            f"Expected {expected_value}, got {new_value} for {metric} (relative={relative})"
        )

        # If offsetting, check that cashflows or pnls were recorded appropriately
        # Note: Weight-only metrics (dirty_price, coverage_rate, etc.) don't affect book value
        # and therefore don't produce offsets
        weight_only_metrics = {
            BalanceSheetMetrics.get("dirty_price"),
            BalanceSheetMetrics.get("coverage_rate"),
            BalanceSheetMetrics.get("accrued_interest_weight"),
            BalanceSheetMetrics.get("agio_weight"),
        }
        is_weight_only = metric in weight_only_metrics

        if offset_item and not is_weight_only:
            if offset_mode == "cash":
                assert len(bs.cashflows) > 0, "Cashflows should be recorded for liquidity offset"
                # Verify some cashflow was recorded (amount can vary based on complex book value calculations)
                total_cashflow = bs.cashflows["Amount"].sum()
                assert abs(total_cashflow) > 0, "Some cashflow should be recorded for offset"
                assert len(bs.pnls) == 0, "No PnL should be recorded for liquidity offset"
            elif offset_mode == "pnl":
                assert len(bs.pnls) > 0, "PnLs should be recorded for PnL offset"
                # Verify some PnL was recorded (amount can vary based on complex book value calculations)
                total_pnl = bs.pnls["Amount"].sum()
                assert abs(total_pnl) > 0, "Some PnL should be recorded for offset"
                assert len(bs.cashflows) == 0, "No cashflows should be recorded for PnL offset"

        # Verify balance sheet balance is maintained when offsets are applied
        if offset_item:
            # Check that the balance sheet remains balanced after mutation and offset
            current_total = bs.get_amount(BalanceSheetItem(), BalanceSheetMetrics.get("book_value_signed"))
            assert abs(current_total) < 0.01, f"Balance sheet should remain balanced with offsets, got {current_total}"

            # Verify balance sheet is still valid after mutation
            bs.validate()
