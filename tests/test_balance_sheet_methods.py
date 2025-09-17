"""Test script for BalanceSheet get_amount and mutate methods."""

import datetime

import polars as pl
import pytest

from bank_projections.financials.balance_sheet import BalanceSheetItem, MutationReason
from bank_projections.financials.metrics import BalanceSheetMetrics
from examples.synthetic_data import create_synthetic_balance_sheet


class TestBalanceSheetMethods:
    """Test the core methods of BalanceSheet class."""

    def setup_method(self) -> None:
        """Create a test balance sheet for each test."""
        self.bs = create_synthetic_balance_sheet(current_date=datetime.date(2024, 12, 31))
        # Validate initial balance sheet
        self.bs.validate()

    def test_get_amount_total_assets(self) -> None:
        """Test getting total asset amounts."""
        # Get total book value for all assets
        total_assets = self.bs.get_amount(
            BalanceSheetItem(BalanceSheetSide="Assets"), BalanceSheetMetrics.get("book_value")
        )

        assert total_assets > 0, "Total assets should be positive"

    def test_get_amount_by_asset_type(self) -> None:
        """Test getting amounts filtered by asset type."""
        # Get loan amounts
        loan_amount = self.bs.get_amount(BalanceSheetItem(ItemType="Mortgages"), BalanceSheetMetrics.get("book_value"))

        # Should have some loans in our synthetic data
        assert loan_amount != 0, "Should have some loan positions"

    def test_get_book_value(self) -> None:
        """Test difference between quantity and book value."""
        total_book_value = self.bs.get_amount(BalanceSheetItem(), BalanceSheetMetrics.get("book_value"))

        # Both should be close to zero for balanced sheet, but may differ due to valuation adjustments
        assert abs(total_book_value) < 0.01, f"Total book value should be ~0, got {total_book_value}"

    def test_mutate_quantity_absolute(self) -> None:
        """Test mutating quantity with absolute amounts."""
        # Get initial loan quantity
        loans_item = BalanceSheetItem(ItemType="Mortgages")
        initial_loan_qty = self.bs.get_amount(loans_item, BalanceSheetMetrics.get("quantity"))

        # Mutate loan quantity to 100,000 (absolute)
        reason = MutationReason(action="test_mutation", test_name="test_mutate_quantity_absolute")
        self.bs.mutate_metric(loans_item, BalanceSheetMetrics.get("quantity"), 100_000, reason, relative=False)

        # Verify the loan quantity is now 100,000 (absolute mutation)
        new_loan_qty = self.bs.get_amount(loans_item, BalanceSheetMetrics.get("quantity"))
        assert abs(new_loan_qty - 100_000) < 1, f"Expected final amount ~100,000, got {new_loan_qty}"

        # Verify the change is tracked correctly
        expected_change = 100_000 - initial_loan_qty
        actual_change = new_loan_qty - initial_loan_qty
        assert abs(actual_change - expected_change) < 1, f"Expected change {expected_change}, got {actual_change}"

        # Note: Balance sheet will be unbalanced after mutation without offset
        # This is expected behavior for this test

    def test_mutate_quantity_relative(self) -> None:
        """Test mutating quantity with relative amounts."""
        # Get initial loan quantity
        loans_item = BalanceSheetItem(ItemType="Mortgages")
        initial_loan_qty = self.bs.get_amount(loans_item, BalanceSheetMetrics.get("quantity"))

        # Mutate loan quantity by adding 50,000 relatively
        reason = MutationReason(action="test_mutation", test_name="test_mutate_quantity_relative")
        self.bs.mutate_metric(loans_item, BalanceSheetMetrics.get("quantity"), 50_000, reason, relative=True)

        # Verify the loan quantity increased
        new_loan_qty = self.bs.get_amount(loans_item, BalanceSheetMetrics.get("quantity"))
        assert new_loan_qty > initial_loan_qty, "Loan quantity should have increased"
        assert abs((new_loan_qty - initial_loan_qty) - 50_000) < 0.001, "Should add exactly 50,000"

        # Note: Balance sheet will be unbalanced after mutation without offset
        # This is expected behavior for this test

    def test_mutate_with_liquidity_offset(self) -> None:
        """Test mutating with liquidity offset."""
        # Get initial loan amount
        loans_item = BalanceSheetItem(ItemType="Mortgages")
        initial_loans = self.bs.get_amount(loans_item, BalanceSheetMetrics.get("book_value"))

        # Clear existing cashflows to track just this mutation
        initial_cashflows_len = len(self.bs.cashflows)

        # Increase loans with liquidity offset
        reason = MutationReason(action="test_mutation", test_name="test_mutate_with_liquidity_offset")
        self.bs.mutate_metric(
            loans_item, BalanceSheetMetrics.get("quantity"), 100_000, reason, relative=True, offset_liquidity=True
        )

        # Verify loans increased and cashflows were recorded
        new_loans = self.bs.get_amount(loans_item, BalanceSheetMetrics.get("book_value"))
        loan_increase = new_loans - initial_loans

        assert loan_increase > 0, "Loans should have increased"

        # Verify cashflows were recorded
        assert len(self.bs.cashflows) > initial_cashflows_len, "Cashflows should have been recorded"

        # Verify cashflow amount matches loan increase (with opposite sign)
        total_cashflow = self.bs.cashflows["Amount"].sum()
        assert abs(total_cashflow + loan_increase) < 1, "Cashflow should offset loan increase"

        # Note: Balance sheet will be unbalanced after mutation without offset
        # This is expected behavior for this test

    def test_mutate_preserves_balance(self) -> None:
        """Test that mutations preserve balance sheet balance."""
        # Get initial total book value
        initial_total = self.bs.get_amount(BalanceSheetItem(), BalanceSheetMetrics.get("book_value"))

        # Perform several mutations with offsets
        loans_item = BalanceSheetItem(ItemType="Mortgages")

        # Mutation with liquidity offset should preserve balance
        reason = MutationReason(action="test_mutation", test_name="test_mutate_preserves_balance")
        self.bs.mutate_metric(
            loans_item, BalanceSheetMetrics.get("quantity"), 50_000, reason, relative=True, offset_liquidity=True
        )

        # Check balance is still maintained
        new_total = self.bs.get_amount(BalanceSheetItem(), BalanceSheetMetrics.get("book_value"))
        assert abs(new_total - initial_total) < 0.01, (
            f"Balance should be preserved, initial: {initial_total}, new: {new_total}"
        )

        # Note: Balance sheet will be unbalanced after mutation without offset
        # This is expected behavior for this test

    def test_mutate_error_conditions(self) -> None:
        """Test error conditions for mutate method."""
        loans_item = BalanceSheetItem(ItemType="Mortgages")

        # Should raise error if both offset_liquidity and offset_pnl are True
        reason = MutationReason(action="test_mutation", test_name="test_mutate_error_conditions")
        with pytest.raises(ValueError):
            self.bs.mutate_metric(
                loans_item, BalanceSheetMetrics.get("quantity"), 100_000, reason, offset_liquidity=True, offset_pnl=True
            )

    def test_mutate_basic_functionality(self) -> None:
        """Test basic functionality of the new mutate method."""
        loans_item = BalanceSheetItem(ItemType="Mortgages")
        initial_quantity = self.bs.get_amount(loans_item, BalanceSheetMetrics.get("quantity"))

        # Test simple quantity mutation
        self.bs.mutate(loans_item, Quantity=pl.col("Quantity") + 10_000)

        # Verify the mutation
        new_quantity = self.bs.get_amount(loans_item, BalanceSheetMetrics.get("quantity"))
        loan_rows = len(self.bs._data.filter(loans_item.filter_expression))
        expected_total = initial_quantity + (loan_rows * 10_000)

        assert abs(new_quantity - expected_total) < 1, f"Expected {expected_total}, got {new_quantity}"

        # Note: Balance sheet will be unbalanced after mutation without offset
        # This is expected behavior for this test

    def test_mutate_multiple_columns(self) -> None:
        """Test mutating multiple columns simultaneously."""
        loans_item = BalanceSheetItem(ItemType="Mortgages")
        initial_quantity = self.bs.get_amount(loans_item, BalanceSheetMetrics.get("quantity"))
        initial_impairment = self.bs.get_amount(loans_item, BalanceSheetMetrics.get("impairment"))

        # Test mutation with multiple columns
        self.bs.mutate(loans_item, Quantity=pl.col("Quantity") + 5_000, Impairment=pl.col("Impairment") + 500)

        # Verify both mutations
        new_quantity = self.bs.get_amount(loans_item, BalanceSheetMetrics.get("quantity"))
        new_impairment = self.bs.get_amount(loans_item, BalanceSheetMetrics.get("impairment"))

        loan_rows = len(self.bs._data.filter(loans_item.filter_expression))
        expected_quantity = initial_quantity + (loan_rows * 5_000)
        expected_impairment = initial_impairment + (loan_rows * 500)

        assert abs(new_quantity - expected_quantity) < 1, f"Expected quantity {expected_quantity}, got {new_quantity}"
        assert abs(new_impairment - expected_impairment) < 1, (
            f"Expected impairment {expected_impairment}, got {new_impairment}"
        )

        # Note: Balance sheet will be unbalanced after mutation without offset
        # This is expected behavior for this test

    def test_mutate_with_custom_pnl_expression(self) -> None:
        """Test mutate with custom PnL expression."""
        loans_item = BalanceSheetItem(ItemType="Mortgages")

        # Clear existing cashflows/pnls
        self.bs.cashflows = pl.DataFrame()
        self.bs.pnls = pl.DataFrame()

        # Test with custom PnL expression (fixed amount per row)
        reason = MutationReason(action="test_mutation", test_name="test_mutate_with_custom_pnl_expression")
        self.bs.mutate(loans_item, pnls={reason: pl.lit(1000.0)}, Quantity=pl.col("Quantity") + 1_000)

        # Verify PnL was recorded
        assert len(self.bs.pnls) > 0, "PnL should have been recorded"
        total_pnl = self.bs.pnls["Amount"].sum()
        loan_rows = len(self.bs._data.filter(loans_item.filter_expression))
        expected_pnl = loan_rows * 1000.0  # pl.lit(1000.0) applied to each row
        assert abs(total_pnl - expected_pnl) < 1, f"Expected PnL {expected_pnl}, got {total_pnl}"

        # Note: Balance sheet will be unbalanced after mutation without offset
        # This is expected behavior for this test

    def test_mutate_with_custom_liquidity_expression(self) -> None:
        """Test mutate with custom liquidity expression."""
        loans_item = BalanceSheetItem(ItemType="Mortgages")

        # Clear existing cashflows/pnls
        self.bs.cashflows = pl.DataFrame()
        self.bs.pnls = pl.DataFrame()

        # Test with custom liquidity expression
        reason = MutationReason(action="test_mutation", test_name="test_mutate_with_custom_liquidity_expression")
        self.bs.mutate(loans_item, cashflows={reason: pl.lit(-500.0)}, Quantity=pl.col("Quantity") + 2_000)

        # Verify cashflow was recorded
        assert len(self.bs.cashflows) > 0, "Cashflow should have been recorded"
        total_cashflow = self.bs.cashflows["Amount"].sum()
        loan_rows = len(self.bs._data.filter(loans_item.filter_expression))
        expected_cashflow = loan_rows * (-500.0)
        assert abs(total_cashflow - expected_cashflow) < 1, (
            f"Expected cashflow {expected_cashflow}, got {total_cashflow}"
        )

        # Note: Balance sheet will be unbalanced after mutation without offset
        # This is expected behavior for this test

    def test_mutate_with_offset_pnl_flag(self) -> None:
        """Test mutate with offset_pnl flag (automatic book value offset)."""
        loans_item = BalanceSheetItem(ItemType="Mortgages")
        initial_balance = self.bs.get_amount(BalanceSheetItem(), BalanceSheetMetrics.get("book_value"))

        # Clear existing cashflows/pnls
        self.bs.cashflows = pl.DataFrame()
        self.bs.pnls = pl.DataFrame()

        # Test with automatic PnL offset
        reason = MutationReason(action="test_mutation", test_name="test_mutate_with_offset_pnl_flag")
        self.bs.mutate(loans_item, offset_pnl=reason, Quantity=pl.col("Quantity") + 3_000)

        # Verify PnL offset was recorded and balance is maintained
        assert len(self.bs.pnls) > 0, "PnL should have been recorded for offset"
        final_balance = self.bs.get_amount(BalanceSheetItem(), BalanceSheetMetrics.get("book_value"))
        assert abs(final_balance - initial_balance) < 0.01, f"Balance should be maintained, got {final_balance}"

        # Note: Balance sheet will be unbalanced after mutation without offset
        # This is expected behavior for this test

    def test_mutate_with_offset_liquidity_flag(self) -> None:
        """Test mutate with offset_liquidity flag (automatic book value offset)."""
        loans_item = BalanceSheetItem(ItemType="Mortgages")
        initial_balance = self.bs.get_amount(BalanceSheetItem(), BalanceSheetMetrics.get("book_value"))

        # Clear existing cashflows/pnls
        self.bs.cashflows = pl.DataFrame()
        self.bs.pnls = pl.DataFrame()

        # Test with automatic liquidity offset
        reason = MutationReason(action="test_mutation", test_name="test_mutate_with_offset_liquidity_flag")
        self.bs.mutate(loans_item, offset_liquidity=reason, Quantity=pl.col("Quantity") + 4_000)

        # Verify liquidity offset was recorded and balance is maintained
        assert len(self.bs.cashflows) > 0, "Cashflow should have been recorded for offset"
        final_balance = self.bs.get_amount(BalanceSheetItem(), BalanceSheetMetrics.get("book_value"))
        assert abs(final_balance - initial_balance) < 0.01, f"Balance should be maintained, got {final_balance}"

        # Note: Balance sheet will be unbalanced after mutation without offset
        # This is expected behavior for this test

    def test_mutate_error_invalid_column(self) -> None:
        """Test that mutate raises error for invalid column names."""
        loans_item = BalanceSheetItem(ItemType="Mortgages")

        with pytest.raises(ValueError, match="Invalid column"):
            self.bs.mutate(loans_item, InvalidColumn=pl.lit(100))

    def test_mutate_error_both_offset_flags(self) -> None:
        """Test that mutate raises error when both offset flags are True."""
        loans_item = BalanceSheetItem(ItemType="Mortgages")

        reason = MutationReason(action="test_mutation", test_name="test_mutate_error_both_offset_flags")
        with pytest.raises(ValueError):
            self.bs.mutate(loans_item, offset_pnl=reason, offset_liquidity=reason, Quantity=pl.col("Quantity") + 1000)

    def test_mutate_cleanup_temporary_columns(self) -> None:
        """Test that temporary columns are properly cleaned up after mutation."""
        loans_item = BalanceSheetItem(ItemType="Mortgages")
        initial_columns = set(self.bs._data.columns)

        # Perform mutation with PnL expression (creates temporary columns)
        reason = MutationReason(action="test_mutation", test_name="test_mutate_cleanup_temporary_columns")
        self.bs.mutate(loans_item, pnls={reason: pl.lit(100.0)}, Quantity=pl.col("Quantity") + 1000)

        # Check that no temporary columns remain
        final_columns = set(self.bs._data.columns)
        temp_columns = {"pnl", "liquidity", "BookValueBefore"}

        for temp_col in temp_columns:
            assert temp_col not in final_columns, f"Temporary column '{temp_col}' was not cleaned up"

        # Should have same columns as before (plus any permanent additions if any)
        assert initial_columns.issubset(final_columns), "Original columns should be preserved"

        # Note: Balance sheet will be unbalanced after mutation without offset
        # This is expected behavior for this test


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
