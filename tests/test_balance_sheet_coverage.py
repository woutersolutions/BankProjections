"""Tests to improve coverage of BalanceSheet class."""

import datetime

import polars as pl
import pytest

from bank_projections.financials.balance_sheet import BalanceSheet, BalanceSheetItem, MutationReason
from bank_projections.financials.metrics import BalanceSheetMetrics
from examples.synthetic_data import create_synthetic_balance_sheet


class TestBalanceSheetCoverage:
    """Tests to improve coverage of BalanceSheet methods."""

    def test_mutate_metric_empty_filter(self):
        """Test that mutate_metric raises ValueError when no positions match filter."""
        bs = create_synthetic_balance_sheet(datetime.date(2024, 12, 31))

        # Create an item that won't match any positions
        item = BalanceSheetItem(ItemType="NonExistentItemType")
        metric = BalanceSheetMetrics.get("quantity")
        reason = MutationReason(test="empty_filter")

        with pytest.raises(ValueError):
            bs.mutate_metric(item, metric, 1000.0, reason)

    def test_copy_method(self):
        """Test the copy method returns a proper copy."""
        bs = create_synthetic_balance_sheet(datetime.date(2024, 12, 31))
        bs_copy = bs.copy()

        # Should be different objects
        assert bs_copy is not bs
        assert bs_copy._data is not bs._data

        # But should have same content
        assert bs_copy._data.shape == bs._data.shape

        # Modifying copy shouldn't affect original
        item = BalanceSheetItem(ItemType="Mortgages")
        metric = BalanceSheetMetrics.get("quantity")
        reason = MutationReason(test="copy_test")

        original_quantity = bs.get_amount(item, metric)
        bs_copy.mutate_metric(item, metric, 1000.0, reason)

        # Original should be unchanged
        assert bs.get_amount(item, metric) == original_quantity

    def test_aggregate_method(self):
        """Test the aggregate method."""
        bs = create_synthetic_balance_sheet(datetime.date(2024, 12, 31))

        aggregated_data, pnls, cashflows = bs.aggregate()

        # Should return three DataFrames
        assert isinstance(aggregated_data, pl.DataFrame)
        assert isinstance(pnls, pl.DataFrame)
        assert isinstance(cashflows, pl.DataFrame)

        # Aggregated data should have fewer rows than original (aggregation)
        assert len(aggregated_data) <= len(bs._data)

    def test_get_differences_method(self):
        """Test the get_differences class method."""
        bs1 = create_synthetic_balance_sheet(datetime.date(2024, 12, 31))
        bs2 = bs1.copy()

        # Modify bs2
        item = BalanceSheetItem(ItemType="Mortgages")
        metric = BalanceSheetMetrics.get("quantity")
        reason = MutationReason(test="differences")
        bs2.mutate_metric(item, metric, 1000.0, reason)

        diff_df = BalanceSheet.get_differences(bs1, bs2)

        # Should have columns with Delta_ prefix
        delta_columns = [col for col in diff_df.columns if col.startswith("Delta_")]
        assert len(delta_columns) > 0

        # At least one row should have non-zero differences
        total_changes = sum([abs(diff_df[col].sum()) for col in delta_columns])
        assert total_changes > 0

    def test_debug_method(self):
        """Test the debug class method."""
        bs1 = create_synthetic_balance_sheet(datetime.date(2024, 12, 31))
        bs2 = bs1.copy()

        # Modify bs2 slightly
        item = BalanceSheetItem(ItemType="Mortgages")
        metric = BalanceSheetMetrics.get("quantity")
        reason = MutationReason(test="debug")
        bs2.mutate_metric(item, metric, 500.0, reason)

        debug_info = BalanceSheet.debug(bs1, bs2)

        # Should return a dictionary
        assert isinstance(debug_info, dict)

        # Should have some debug information
        assert len(debug_info) > 0
