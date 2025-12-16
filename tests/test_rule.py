"""Unit tests for rule module."""

import datetime
from unittest.mock import Mock

import pytest

from bank_projections.financials.balance_sheet import BalanceSheet
from bank_projections.projections.projectionrule import ProjectionRule
from bank_projections.scenarios.scenario import ScenarioSnapShot
from bank_projections.utils.time import TimeIncrement


class MockProjectionRule(ProjectionRule):
    """Mock implementation of Rule for testing."""

    def apply(self, bs: BalanceSheet, increment: TimeIncrement, scenario: ScenarioSnapShot) -> BalanceSheet:
        return bs


class TestTimeIncrement:
    """Test TimeIncrement class functionality."""

    def test_init(self) -> None:
        """Test TimeIncrement initialization."""
        from_date = datetime.date(2025, 1, 1)
        to_date = datetime.date(2025, 1, 31)
        increment = TimeIncrement(from_date, to_date)

        assert increment.from_date == from_date
        assert increment.to_date == to_date

    def test_days_property(self) -> None:
        """Test days property calculation."""
        from_date = datetime.date(2025, 1, 1)
        to_date = datetime.date(2025, 1, 31)
        increment = TimeIncrement(from_date, to_date)

        assert increment.days == 30

    def test_days_property_same_date(self) -> None:
        """Test days property when from and to dates are the same."""
        date = datetime.date(2025, 1, 1)
        increment = TimeIncrement(date, date)

        assert increment.days == 0

    def test_days_property_leap_year(self) -> None:
        """Test days calculation across leap year."""
        from_date = datetime.date(2024, 2, 28)
        to_date = datetime.date(2024, 3, 1)
        increment = TimeIncrement(from_date, to_date)

        assert increment.days == 2  # Feb 29 exists in 2024

    def test_days_property_cross_year(self) -> None:
        """Test days calculation across year boundary."""
        from_date = datetime.date(2024, 12, 25)
        to_date = datetime.date(2025, 1, 5)
        increment = TimeIncrement(from_date, to_date)

        assert increment.days == 11


class TestRule:
    """Test Rule abstract base class."""

    def test_rule_is_abstract(self) -> None:
        """Test that Rule cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ProjectionRule()

    def test_mock_rule_implementation(self) -> None:
        """Test that concrete implementations work."""
        rule = MockProjectionRule()
        increment = TimeIncrement(datetime.date(2025, 1, 1), datetime.date(2025, 1, 31))
        mock_scenario = Mock(spec=ScenarioSnapShot)
        mock_bs = Mock(spec=BalanceSheet)

        result = rule.apply(mock_bs, increment, mock_scenario)
        assert result == mock_bs
