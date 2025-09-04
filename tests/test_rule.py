"""Unit tests for rule module."""

import datetime

import pytest

from bank_projections.projections.rule import Rule, TimeIncrement


class MockRule(Rule):
    """Mock implementation of Rule for testing."""

    def apply(self, bs, increment):
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
            Rule()

    def test_mock_rule_implementation(self) -> None:
        """Test that concrete implementations work."""
        rule = MockRule()
        increment = TimeIncrement(datetime.date(2025, 1, 1), datetime.date(2025, 1, 31))

        # Should not raise error
        result = rule.apply(None, increment)
        assert result is None  # Mock implementation returns None
