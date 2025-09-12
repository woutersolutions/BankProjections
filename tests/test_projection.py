"""Unit tests for projection module."""

import datetime
from unittest.mock import Mock, patch

from bank_projections.financials.balance_sheet import BalanceSheet
from bank_projections.projections.projection import Projection
from bank_projections.projections.rule import Rule
from bank_projections.projections.time import TimeHorizon


class MockRule(Rule):
    """Mock implementation of Rule for testing."""

    def __init__(self, name: str = "mock"):
        self.name = name
        self.apply_calls = []

    def apply(self, bs: BalanceSheet, increment) -> BalanceSheet:
        """Mock apply method that tracks calls and returns modified copy."""
        self.apply_calls.append((bs, increment))
        # Create a simple modification to verify the rule was applied
        modified_bs = bs.copy()
        return modified_bs


class TestProjection:
    """Test Projection class functionality."""

    def test_init(self) -> None:
        """Test Projection initialization."""
        mock_bs = Mock(spec=BalanceSheet)
        mock_rules = [MockRule("rule1")]
        mock_horizon = Mock(spec=TimeHorizon)

        projection = Projection(mock_bs, mock_rules, mock_horizon)

        assert projection.bs_start is mock_bs
        assert projection.rules == mock_rules
        assert projection.horizon is mock_horizon

    def test_run_single_rule_single_increment(self) -> None:
        """Test run method with single rule and single time increment."""
        mock_bs = Mock(spec=BalanceSheet)
        mock_bs.copy.return_value = Mock(spec=BalanceSheet)

        rule = MockRule("test_rule")

        # Create a simple time horizon with one increment
        start_date = datetime.date(2024, 1, 1)
        end_date = datetime.date(2024, 1, 31)
        horizon = TimeHorizon([start_date, end_date])

        projection = Projection(mock_bs, [rule], horizon)

        with patch("bank_projections.projections.projection.logger"):
            result = projection.run()

        # TimeHorizon with 2 dates yields 2 increments: (start->start) and (start->end)
        # This means 3 balance sheets: original + 2 projections
        assert len(result) == 3
        assert result[0] is mock_bs  # Original balance sheet

        # Rule should have been called twice (for both increments)
        assert len(rule.apply_calls) == 2

        # First increment is zero-day (start_date to start_date)
        applied_bs1, applied_increment1 = rule.apply_calls[0]
        assert applied_increment1.from_date == start_date
        assert applied_increment1.to_date == start_date

        # Second increment is actual time increment
        applied_bs2, applied_increment2 = rule.apply_calls[1]
        assert applied_increment2.from_date == start_date
        assert applied_increment2.to_date == end_date

    def test_run_multiple_rules_single_increment(self) -> None:
        """Test run method with multiple rules and single time increment."""
        mock_bs = Mock(spec=BalanceSheet)
        mock_bs.copy.return_value = Mock(spec=BalanceSheet)

        rule1 = MockRule("rule1")
        rule2 = MockRule("rule2")

        start_date = datetime.date(2024, 1, 1)
        end_date = datetime.date(2024, 1, 31)
        horizon = TimeHorizon([start_date, end_date])

        projection = Projection(mock_bs, [rule1, rule2], horizon)

        with patch("bank_projections.projections.projection.logger"):
            result = projection.run()

        # TimeHorizon with 2 dates yields 2 increments, so 3 balance sheets total
        assert len(result) == 3

        # Both rules should have been called twice (for both increments)
        assert len(rule1.apply_calls) == 2
        assert len(rule2.apply_calls) == 2

    def test_run_single_rule_multiple_increments(self) -> None:
        """Test run method with single rule and multiple time increments."""
        mock_bs = Mock(spec=BalanceSheet)
        mock_bs.copy.return_value = Mock(spec=BalanceSheet)

        rule = MockRule("test_rule")

        # Create time horizon with multiple increments
        dates = [datetime.date(2024, 1, 1), datetime.date(2024, 1, 15), datetime.date(2024, 2, 1)]
        horizon = TimeHorizon(dates)

        projection = Projection(mock_bs, [rule], horizon)

        with patch("bank_projections.projections.projection.logger"):
            result = projection.run()

        # TimeHorizon with 3 dates yields 3 increments, so 4 balance sheets total
        assert len(result) == 4

        # Rule should have been called 3 times (for 3 increments)
        assert len(rule.apply_calls) == 3

    def test_run_multiple_rules_multiple_increments(self) -> None:
        """Test run method with multiple rules and multiple time increments."""
        mock_bs = Mock(spec=BalanceSheet)
        mock_bs.copy.return_value = Mock(spec=BalanceSheet)

        rule1 = MockRule("rule1")
        rule2 = MockRule("rule2")

        dates = [datetime.date(2024, 1, 1), datetime.date(2024, 1, 15), datetime.date(2024, 2, 1)]
        horizon = TimeHorizon(dates)

        projection = Projection(mock_bs, [rule1, rule2], horizon)

        with patch("bank_projections.projections.projection.logger"):
            result = projection.run()

        # TimeHorizon with 3 dates yields 3 increments, so 4 balance sheets total
        assert len(result) == 4

        # Each rule should have been called 3 times (for 3 increments)
        assert len(rule1.apply_calls) == 3
        assert len(rule2.apply_calls) == 3

    def test_run_no_rules(self) -> None:
        """Test run method with no rules."""
        mock_bs = Mock(spec=BalanceSheet)
        mock_bs.copy.return_value = Mock(spec=BalanceSheet)

        start_date = datetime.date(2024, 1, 1)
        end_date = datetime.date(2024, 1, 31)
        horizon = TimeHorizon([start_date, end_date])

        projection = Projection(mock_bs, [], horizon)

        with patch("bank_projections.projections.projection.logger"):
            result = projection.run()

        # Should still return balance sheets even with no rules (3 total)
        assert len(result) == 3
        assert result[0] is mock_bs

    def test_run_single_date_horizon(self) -> None:
        """Test run method with single date horizon."""
        mock_bs = Mock(spec=BalanceSheet)
        mock_bs.copy.return_value = Mock(spec=BalanceSheet)

        rule = MockRule("test_rule")

        # Single date horizon yields one zero-day increment (start->start)
        horizon = TimeHorizon([datetime.date(2024, 1, 1)])

        projection = Projection(mock_bs, [rule], horizon)

        with patch("bank_projections.projections.projection.logger"):
            result = projection.run()

        # Should return start balance sheet plus one projection
        assert len(result) == 2
        assert result[0] is mock_bs

        # Rule should have been called once for the zero-day increment
        assert len(rule.apply_calls) == 1

    @patch("bank_projections.projections.projection.logger")
    def test_run_logs_progress(self, mock_logger) -> None:
        """Test that run method logs progress correctly."""
        mock_bs = Mock(spec=BalanceSheet)
        mock_bs.copy.return_value = Mock(spec=BalanceSheet)

        rule = MockRule("test_rule")

        dates = [datetime.date(2024, 1, 1), datetime.date(2024, 1, 15), datetime.date(2024, 2, 1)]
        horizon = TimeHorizon(dates)

        projection = Projection(mock_bs, [rule], horizon)
        projection.run()

        # Should have logged progress for each time increment
        assert mock_logger.info.call_count == 3  # 3 increments

        # Check log message format
        calls = mock_logger.info.call_args_list
        assert "Time increment 1/3" in calls[0][0][0]
        assert "Time increment 2/3" in calls[1][0][0]
        assert "Time increment 3/3" in calls[2][0][0]

    def test_run_balance_sheet_chaining(self) -> None:
        """Test that balance sheets are properly chained between time increments."""
        # Create a real balance sheet for this test
        from examples.synthetic_data import create_synthetic_balance_sheet

        start_bs = create_synthetic_balance_sheet(current_date=datetime.date(2024, 12, 31))

        # Create a simple rule that tracks the number of times it's applied
        class CountingRule(Rule):
            def __init__(self):
                self.apply_count = 0

            def apply(self, bs: BalanceSheet, increment) -> BalanceSheet:
                self.apply_count += 1
                # Just return a copy, no mutations needed for this test
                return bs.copy()

        rule = CountingRule()

        dates = [datetime.date(2024, 1, 1), datetime.date(2024, 1, 15), datetime.date(2024, 2, 1)]
        horizon = TimeHorizon(dates)

        projection = Projection(start_bs, [rule], horizon)

        with patch("bank_projections.projections.projection.logger"):
            result = projection.run()

        # Should have 4 balance sheets (3 dates = 3 increments + original)
        assert len(result) == 4

        # Rule should have been applied 3 times (for 3 increments)
        assert rule.apply_count == 3

        # All results should be valid BalanceSheet objects
        for bs in result:
            assert bs is not None
            assert isinstance(bs, BalanceSheet)

        # First balance sheet should be the original
        assert result[0] is start_bs
