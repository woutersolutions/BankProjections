"""Test script for balance sheet creation with minimal code."""

import pytest

from examples.synthetic_data import create_balanced_balance_sheet
from src.bank_projections.financials.balance_sheet import BalanceSheetItem
from src.bank_projections.financials.metrics import BalanceSheetMetrics


def test_create_simple_balance_sheet():
    """Test that a balance sheet can be created with minimal code."""
    # Create a balanced balance sheet with default parameters
    bs = create_balanced_balance_sheet(
        total_assets=1_000_000,
        random_seed=42,  # For reproducible tests
    )

    # Verify the balance sheet was created successfully
    assert len(bs) > 0

    # Verify it's balanced (total book value should be ~0)
    total_book_value = bs.get_amount(BalanceSheetItem(), BalanceSheetMetrics.book_value)
    assert abs(total_book_value) < 0.01, f"Balance sheet not balanced: {total_book_value}"


def test_balance_sheet_components():
    """Test that the balance sheet has proper asset, liability, and equity components."""
    bs = create_balanced_balance_sheet(
        total_assets=1_000_000, asset_liability_ratio=0.8, equity_ratio=0.2, random_seed=42
    )

    # Check assets
    assets = bs.get_amount(BalanceSheetItem(BalanceSheetSide="Asset"), BalanceSheetMetrics.book_value)
    assert assets > 0, "Assets should be positive"
    assert abs(assets - 1_000_000) < 1000, f"Assets should be close to 1M, got {assets}"

    # Check liabilities
    liabilities = bs.get_amount(BalanceSheetItem(BalanceSheetSide="Liability"), BalanceSheetMetrics.book_value)
    assert liabilities < 0, "Liabilities should be negative"
    assert abs(abs(liabilities) - 800_000) < 1000, f"Liabilities should be close to 800K, got {abs(liabilities)}"

    # Check equity
    equity = bs.get_amount(BalanceSheetItem(BalanceSheetSide="Equity"), BalanceSheetMetrics.book_value)
    assert equity < 0, "Equity should be negative"
    assert abs(abs(equity) - 200_000) < 1000, f"Equity should be close to 200K, got {abs(equity)}"


def test_different_balance_sheet_sizes():
    """Test creating balance sheets of different sizes."""
    for total_assets in [100_000, 1_000_000, 10_000_000]:
        bs = create_balanced_balance_sheet(total_assets=total_assets, random_seed=42)

        actual_assets = bs.get_amount(BalanceSheetItem(BalanceSheetSide="Asset"), BalanceSheetMetrics.book_value)
        assert abs(actual_assets - total_assets) < 1000, f"Assets mismatch for size {total_assets}"

        # Verify balance
        total_book_value = bs.get_amount(BalanceSheetItem(), BalanceSheetMetrics.book_value)
        assert abs(total_book_value) < 0.01, f"Balance sheet not balanced for size {total_assets}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
