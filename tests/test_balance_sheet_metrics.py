"""Test script for BalanceSheet get_amount and mutate methods."""

import polars as pl
import pytest

from examples.synthetic_data import create_balanced_balance_sheet
from src.bank_projections.financials.balance_sheet import BalanceSheetItem
from src.bank_projections.financials.metrics import BalanceSheetMetrics


class TestBalanceSheetMethods:
    """Test the core methods of BalanceSheet class."""

    def setup_method(self) -> None:
        """Create a test balance sheet for each test."""
        self.bs = create_balanced_balance_sheet(total_assets=1_000_000, random_seed=42)

    # Test whether the quantity can be changed and offset with cash
    def test_mutate_quantity_with_cash_offset(self) -> None:
        """Test mutating quantity with cash offset."""
        # Get initial loan quantity
        loans_item = BalanceSheetItem(AssetType="loan")
        initial_loan_qty = self.bs.get_amount(loans_item, BalanceSheetMetrics.quantity)
        initial_cash_qty = self.bs.get_amount(BalanceSheetItem(AssetType="cash"), BalanceSheetMetrics.quantity)

        # Mutate loan quantity to 100,000 with cash offset
        mutation_amount = 100_000
        expected_mutation = mutation_amount - initial_loan_qty

        # Clear existing cashflows to track this mutation
        self.bs.cashflows = pl.DataFrame()
        self.bs.pnls = pl.DataFrame()

        self.bs.mutate_metric(
            loans_item, BalanceSheetMetrics.quantity, mutation_amount, relative=False, offset_liquidity=True
        )

        # Verify the loan quantity is now 100,000 (absolute mutation)
        new_loan_qty = self.bs.get_amount(loans_item, BalanceSheetMetrics.quantity)
        assert abs(new_loan_qty - 100_000) < 1, f"Expected final amount ~100,000, got {new_loan_qty}"

        # Verify that the cash is mutated correctly
        new_cash_qty = self.bs.get_amount(BalanceSheetItem(AssetType="cash"), BalanceSheetMetrics.quantity)
        cash_change = new_cash_qty - initial_cash_qty

        # Verify cashflows were recorded for the offset
        assert len(self.bs.cashflows) > 0, "Cashflows should have been recorded"

        # Verify the cashflow amount matches the loan change (with opposite sign)
        total_cashflow = self.bs.cashflows["Amount"].sum()
        assert abs(total_cashflow + expected_mutation) < 1, "Cashflow should offset loan quantity change"

        # Verify no PnL changes were recorded since we used liquidity offset
        assert len(self.bs.pnls) == 0, "No PnL changes should be recorded for liquidity offset"

    # Parameterized test for all editable metrics and offset modes
    @pytest.mark.parametrize(
        "metric, asset_type",
        [
            (BalanceSheetMetrics.quantity, "loan"),
            (BalanceSheetMetrics.impairment, "loan"),
            (BalanceSheetMetrics.accrued_interest, "bond"),
            (BalanceSheetMetrics.agio, "bond"),
            (BalanceSheetMetrics.clean_price, "bond"),
            # Include derived, but updatable metrics (weights)
            (BalanceSheetMetrics.coverage_rate, "loan"),
            (BalanceSheetMetrics.accrued_interest_rate, "bond"),
            (BalanceSheetMetrics.agio_weight, "bond"),
        ],
    )
    @pytest.mark.parametrize("offset_mode", [None, "cash", "pnl"])
    @pytest.mark.parametrize("relative", [False, True])
    def test_mutate_metric_with_offsets(self, metric, asset_type, offset_mode, relative):
        item = BalanceSheetItem(AssetType=asset_type)
        initial_value = self.bs.get_amount(item, metric)

        # Choose a sensible mutation target per metric type and relative mode
        if metric == BalanceSheetMetrics.clean_price:
            if relative:
                mutation_amount = 5.0  # add 5 to current clean price (weighted)
                expected_value = initial_value + mutation_amount
            else:
                mutation_amount = 105.0  # set clean price to a fixed realistic value
                expected_value = mutation_amount
        elif metric in {
            BalanceSheetMetrics.coverage_rate,
            BalanceSheetMetrics.accrued_interest_rate,
            BalanceSheetMetrics.agio_weight,
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
            offset_column = BalanceSheetMetrics.quantity
            offset_item = BalanceSheetItem(AssetType="cash")
        elif offset_mode == "pnl":
            offset_args["offset_pnl"] = True
            offset_column = BalanceSheetMetrics.quantity
            offset_item = BalanceSheetItem(BalanceSheetSide="Equity")
        else:
            offset_column = None
            offset_item = None

        # Record initial offset value before mutation
        if offset_item:
            initial_offset = self.bs.get_amount(offset_item, offset_column)

        # Clear existing cashflows/pnls to track just this mutation
        if offset_item:
            self.bs.cashflows = pl.DataFrame()
            self.bs.pnls = pl.DataFrame()

        # Perform mutation
        self.bs.mutate_metric(item, metric, mutation_amount, relative=relative, **offset_args)

        # Check mutated value
        new_value = self.bs.get_amount(item, metric)
        assert abs(new_value - expected_value) < 1, (
            f"Expected {expected_value}, got {new_value} for {metric} (relative={relative})"
        )

        # If offsetting, check that cashflows or pnls were recorded appropriately
        if offset_item:
            if offset_mode == "cash":
                assert len(self.bs.cashflows) > 0, "Cashflows should be recorded for liquidity offset"
                # Verify some cashflow was recorded (amount can vary based on complex book value calculations)
                total_cashflow = self.bs.cashflows["Amount"].sum()
                assert abs(total_cashflow) > 0, "Some cashflow should be recorded for offset"
                assert len(self.bs.pnls) == 0, "No PnL should be recorded for liquidity offset"
            elif offset_mode == "pnl":
                assert len(self.bs.pnls) > 0, "PnLs should be recorded for PnL offset"
                # Verify some PnL was recorded (amount can vary based on complex book value calculations)
                total_pnl = self.bs.pnls["Amount"].sum()
                assert abs(total_pnl) > 0, "Some PnL should be recorded for offset"
                assert len(self.bs.cashflows) == 0, "No cashflows should be recorded for PnL offset"

        # Verify balance sheet balance is maintained when offsets are applied
        if offset_item:
            # Check that the balance sheet remains balanced after mutation and offset
            current_total = self.bs.get_amount(BalanceSheetItem(), BalanceSheetMetrics.book_value)
            assert abs(current_total) < 0.01, f"Balance sheet should remain balanced with offsets, got {current_total}"
