"""Test script for balance sheet creation with minimal code."""

import datetime

import pytest

from bank_projections.financials.balance_sheet_item import BalanceSheetItem
from bank_projections.financials.metrics import BalanceSheetMetrics
from examples.synthetic_data import create_synthetic_balance_sheet


def test_create_simple_balance_sheet(minimal_scenario):
    """Test that a balance sheet can be created with minimal code."""
    # Create a balanced balance sheet with default parameters
    bs = create_synthetic_balance_sheet(current_date=datetime.date(2024, 12, 31), scenario=minimal_scenario)

    # Verify the balance sheet was created successfully
    assert len(bs) > 0

    # Verify it's balanced (total book value should be ~0)
    total_book_value = bs.get_amount(BalanceSheetItem(), BalanceSheetMetrics.get("book_value"))
    assert abs(total_book_value) < 0.01, f"Balance sheet not balanced: {total_book_value}"

    # Verify balance sheet is valid
    bs.validate()


def test_balance_sheet_components(minimal_scenario):
    """Test that the balance sheet has proper asset, liability, and equity components."""
    bs = create_synthetic_balance_sheet(current_date=datetime.date(2024, 12, 31), scenario=minimal_scenario)

    # Check assets
    assets = bs.get_amount(BalanceSheetItem(BalanceSheetSide="Assets"), BalanceSheetMetrics.get("book_value"))
    assert assets > 0, "Assets should be positive"

    # Check liabilities
    liabilities = bs.get_amount(BalanceSheetItem(BalanceSheetSide="Liabilities"), BalanceSheetMetrics.get("book_value"))
    assert liabilities < 0, "Liabilities should be negative"

    # Check equity
    equity = bs.get_amount(BalanceSheetItem(BalanceSheetSide="Equity"), BalanceSheetMetrics.get("book_value"))
    assert equity < 0, "Equity should be negative"

    # Verify balance sheet is valid
    bs.validate()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
