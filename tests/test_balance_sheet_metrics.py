"""Test script for BalanceSheet get_amount and mutate methods."""

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
        mutation_result = self.bs.mutate(
            loans_item, BalanceSheetMetrics.quantity, mutation_amount, relative=False, offset_liquidity=True
        )
        expected_mutation = mutation_amount - initial_loan_qty

        # Verify the loan quantity is now 100,000 (absolute mutation)
        new_loan_qty = self.bs.get_amount(loans_item, BalanceSheetMetrics.quantity)
        assert abs(new_loan_qty - 100_000) < 1, f"Expected final amount ~100,000, got {new_loan_qty}"

        # Verify that the cash is mutated correctly
        new_cash_qty = self.bs.get_amount(BalanceSheetItem(AssetType="cash"), BalanceSheetMetrics.quantity)
        cash_change = new_cash_qty - initial_cash_qty

        # Check that the cash offset matches the loan quantity change
        assert abs(cash_change + expected_mutation) < 1, "Expected cash change to offset loan quantity change, "

        # Verify the mutation result table is returned
        assert len(mutation_result) > 0, "Mutation should return a results table"

        # Verify that the mutation table shows the changed quantity and cash
        assert abs(mutation_result["BookValue"].sum() - expected_mutation) < 1, (
            "Total book value change should be close to zero after mutation"
        )
        assert abs(mutation_result["Liquidity"].sum() + expected_mutation) < 1, (
            "Total cash change should match the loan quantity change"
        )
        assert abs(mutation_result["PnL"].sum()) < 1, "Total cash change should match the loan quantity change"
