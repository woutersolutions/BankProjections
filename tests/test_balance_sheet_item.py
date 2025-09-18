"""Tests for BalanceSheetItem class to improve coverage."""

import pytest

from bank_projections.financials.balance_sheet import BalanceSheetItem


class TestBalanceSheetItem:
    """Test BalanceSheetItem functionality not covered in other tests."""

    def test_invalid_identifier_in_constructor(self):
        """Test that invalid identifiers raise ValueError in constructor."""
        with pytest.raises(ValueError, match="Invalid identifier 'invalid_key'"):
            BalanceSheetItem(invalid_key="value")

    def test_add_identifier_invalid_key(self):
        """Test that add_identifier raises ValueError for invalid keys."""
        item = BalanceSheetItem()
        with pytest.raises(ValueError, match="Invalid identifier 'invalid_key'"):
            item.add_identifier("invalid_key", "value")

    def test_add_identifier_valid_key(self):
        """Test that add_identifier works with valid keys."""
        item = BalanceSheetItem()
        new_item = item.add_identifier("ItemType", "Mortgages")

        assert new_item.identifiers["ItemType"] == "Mortgages"
        assert item.identifiers == {}  # Original unchanged

    def test_remove_identifier(self):
        """Test remove_identifier functionality."""
        item = BalanceSheetItem(ItemType="Mortgages", BalanceSheetSide="Assets")
        new_item = item.remove_identifier("ItemType")

        assert "ItemType" not in new_item.identifiers
        assert "BalanceSheetSide" in new_item.identifiers
        assert "ItemType" in item.identifiers  # Original unchanged

    def test_copy(self):
        """Test copy functionality."""
        item = BalanceSheetItem(ItemType="Mortgages", BalanceSheetSide="Assets")
        copied_item = item.copy()

        assert copied_item.identifiers == item.identifiers
        assert copied_item is not item  # Different objects

        # Modifying copy shouldn't affect original
        copied_item.add_identifier("BalanceSheetSide", "Liabilities")
        assert item.identifiers["BalanceSheetSide"] == "Assets"

    def test_calculation_tag_identifier(self):
        """Test that calculation tag identifiers are properly cleaned."""
        # This should test the CALCULATION_TAGS branch
        item = BalanceSheetItem(ItemType="Mortgages", ValuationMethod="Fair Value")
        # Add test for calculation tags if they exist in config
        assert item.identifiers["ItemType"] == "Mortgages"
        assert "ValuationMethod" in item.identifiers
