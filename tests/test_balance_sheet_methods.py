"""Test script for BalanceSheet get_amount and mutate methods."""

import pytest

from examples.synthetic_data import create_balanced_balance_sheet
from src.bank_projections.financials.balance_sheet import BalanceSheetItem
from src.bank_projections.financials.metrics import BalanceSheetMetrics


class TestBalanceSheetMethods:
    """Test the core methods of BalanceSheet class."""

    def setup_method(self) -> None:
        """Create a test balance sheet for each test."""
        self.bs = create_balanced_balance_sheet(total_assets=1_000_000, random_seed=42)

    def test_get_amount_total_assets(self) -> None:
        """Test getting total asset amounts."""
        # Get total book value for all assets
        total_assets = self.bs.get_amount(BalanceSheetItem(BalanceSheetSide="Asset"), BalanceSheetMetrics.book_value)

        assert total_assets > 0, "Total assets should be positive"
        assert abs(total_assets - 1_000_000) < 1000, f"Total assets should be ~1M, got {total_assets}"

    def test_get_amount_by_asset_type(self) -> None:
        """Test getting amounts filtered by asset type."""
        # Get loan amounts
        loan_amount = self.bs.get_amount(BalanceSheetItem(AssetType="loan"), BalanceSheetMetrics.book_value)

        # Should have some loans in our synthetic data
        assert loan_amount != 0, "Should have some loan positions"

    def test_get_amount_quantity_vs_book_value(self) -> None:
        """Test difference between quantity and book value."""
        total_quantity = self.bs.get_amount(BalanceSheetItem(), BalanceSheetMetrics.quantity)

        total_book_value = self.bs.get_amount(BalanceSheetItem(), BalanceSheetMetrics.book_value)

        # Both should be close to zero for balanced sheet, but may differ due to valuation adjustments
        assert abs(total_quantity) < 0.01, f"Total quantity should be ~0, got {total_quantity}"
        assert abs(total_book_value) < 0.01, f"Total book value should be ~0, got {total_book_value}"

    def test_mutate_quantity_absolute(self) -> None:
        """Test mutating quantity with absolute amounts."""
        # Get initial loan quantity
        loans_item = BalanceSheetItem(AssetType="loan")
        initial_loan_qty = self.bs.get_amount(loans_item, BalanceSheetMetrics.quantity)

        # Mutate loan quantity to 100,000 (absolute)
        self.bs.mutate_metric(loans_item, BalanceSheetMetrics.quantity, 100_000, relative=False)

        # Verify the loan quantity is now 100,000 (absolute mutation)
        new_loan_qty = self.bs.get_amount(loans_item, BalanceSheetMetrics.quantity)
        assert abs(new_loan_qty - 100_000) < 1, f"Expected final amount ~100,000, got {new_loan_qty}"

        # Verify the change is tracked correctly
        expected_change = 100_000 - initial_loan_qty
        actual_change = new_loan_qty - initial_loan_qty
        assert abs(actual_change - expected_change) < 1, f"Expected change {expected_change}, got {actual_change}"

    def test_mutate_quantity_relative(self) -> None:
        """Test mutating quantity with relative amounts."""
        # Get initial loan quantity
        loans_item = BalanceSheetItem(AssetType="loan")
        initial_loan_qty = self.bs.get_amount(loans_item, BalanceSheetMetrics.quantity)

        # Mutate loan quantity by adding 50,000 relatively
        self.bs.mutate_metric(loans_item, BalanceSheetMetrics.quantity, 50_000, relative=True)

        # Verify the loan quantity increased
        new_loan_qty = self.bs.get_amount(loans_item, BalanceSheetMetrics.quantity)
        assert new_loan_qty > initial_loan_qty, "Loan quantity should have increased"
        assert (new_loan_qty - initial_loan_qty) == 50_000, "Should add exactly 50,000"

    def test_mutate_with_liquidity_offset(self) -> None:
        """Test mutating with liquidity offset."""
        # Get initial cash amount
        cash_item = BalanceSheetItem(AssetType="cash")
        initial_cash = self.bs.get_amount(cash_item, BalanceSheetMetrics.book_value)

        # Get initial loan amount
        loans_item = BalanceSheetItem(AssetType="loan")
        initial_loans = self.bs.get_amount(loans_item, BalanceSheetMetrics.book_value)

        # Clear existing cashflows to track just this mutation
        initial_cashflows_len = len(self.bs.cashflows)

        # Increase loans with liquidity offset
        self.bs.mutate_metric(loans_item, BalanceSheetMetrics.quantity, 100_000, relative=True, offset_liquidity=True)

        # Verify loans increased and cashflows were recorded
        new_loans = self.bs.get_amount(loans_item, BalanceSheetMetrics.book_value)
        loan_increase = new_loans - initial_loans

        assert loan_increase > 0, "Loans should have increased"

        # Verify cashflows were recorded
        assert len(self.bs.cashflows) > initial_cashflows_len, "Cashflows should have been recorded"

        # Verify cashflow amount matches loan increase (with opposite sign)
        total_cashflow = self.bs.cashflows["Amount"].sum()
        assert abs(total_cashflow + loan_increase) < 1, "Cashflow should offset loan increase"

    def test_mutate_preserves_balance(self) -> None:
        """Test that mutations preserve balance sheet balance."""
        # Get initial total book value
        initial_total = self.bs.get_amount(BalanceSheetItem(), BalanceSheetMetrics.book_value)

        # Perform several mutations with offsets
        loans_item = BalanceSheetItem(AssetType="loan")

        # Mutation with liquidity offset should preserve balance
        self.bs.mutate_metric(loans_item, BalanceSheetMetrics.quantity, 50_000, relative=True, offset_liquidity=True)

        # Check balance is still maintained
        new_total = self.bs.get_amount(BalanceSheetItem(), BalanceSheetMetrics.book_value)
        assert abs(new_total - initial_total) < 0.01, (
            f"Balance should be preserved, initial: {initial_total}, new: {new_total}"
        )

    def test_mutate_error_conditions(self) -> None:
        """Test error conditions for mutate method."""
        loans_item = BalanceSheetItem(AssetType="loan")

        # Should raise error if both offset_liquidity and offset_pnl are True
        with pytest.raises(ValueError, match="Cannot offset with both cash and pnl"):
            self.bs.mutate_metric(
                loans_item, BalanceSheetMetrics.quantity, 100_000, offset_liquidity=True, offset_pnl=True
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
