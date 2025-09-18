"""Unit tests for time module."""

import datetime

from bank_projections.projections.time import TimeHorizon, TimeIncrement, to_end_of_month


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

    def test_portion_year_property(self) -> None:
        """Test portion_year property calculation."""
        from_date = datetime.date(2024, 1, 1)
        to_date = datetime.date(2024, 12, 31)
        increment = TimeIncrement(from_date, to_date)

        expected = 365 / 365.25  # 2024 is a leap year
        assert abs(increment.portion_year - expected) < 0.001

    def test_portion_year_property_one_month(self) -> None:
        """Test portion_year property for approximately one month."""
        from_date = datetime.date(2024, 1, 1)
        to_date = datetime.date(2024, 1, 31)
        increment = TimeIncrement(from_date, to_date)

        expected = 30 / 365.25
        assert abs(increment.portion_year - expected) < 0.001


class TestToEndOfMonth:
    """Test to_end_of_month function."""

    def test_to_end_of_month_regular_month(self) -> None:
        """Test conversion to end of month for regular month."""
        date = datetime.date(2024, 1, 15)
        result = to_end_of_month(date)
        assert result == datetime.date(2024, 1, 31)

    def test_to_end_of_month_february_leap_year(self) -> None:
        """Test conversion to end of February in leap year."""
        date = datetime.date(2024, 2, 10)
        result = to_end_of_month(date)
        assert result == datetime.date(2024, 2, 29)

    def test_to_end_of_month_february_non_leap_year(self) -> None:
        """Test conversion to end of February in non-leap year."""
        date = datetime.date(2023, 2, 10)
        result = to_end_of_month(date)
        assert result == datetime.date(2023, 2, 28)

    def test_to_end_of_month_already_end_of_month(self) -> None:
        """Test when date is already at end of month."""
        date = datetime.date(2024, 1, 31)
        result = to_end_of_month(date)
        assert result == datetime.date(2024, 1, 31)

    def test_to_end_of_month_short_month(self) -> None:
        """Test conversion for month with 30 days."""
        date = datetime.date(2024, 4, 15)
        result = to_end_of_month(date)
        assert result == datetime.date(2024, 4, 30)


class TestTimeHorizon:
    """Test TimeHorizon class functionality."""

    def test_init_single_date(self) -> None:
        """Test initialization with a single date."""
        dates = [datetime.date(2024, 1, 1)]
        horizon = TimeHorizon(dates)

        assert len(horizon) == 1
        assert horizon.start_date == datetime.date(2024, 1, 1)
        assert horizon.end_date == datetime.date(2024, 1, 1)

    def test_init_multiple_dates(self) -> None:
        """Test initialization with multiple dates."""
        dates = [datetime.date(2024, 3, 1), datetime.date(2024, 1, 1), datetime.date(2024, 2, 1)]
        horizon = TimeHorizon(dates)

        assert len(horizon) == 3
        assert horizon.start_date == datetime.date(2024, 1, 1)
        assert horizon.end_date == datetime.date(2024, 3, 1)

    def test_init_removes_duplicates(self) -> None:
        """Test that duplicate dates are removed."""
        dates = [datetime.date(2024, 1, 1), datetime.date(2024, 1, 1), datetime.date(2024, 2, 1)]
        horizon = TimeHorizon(dates)

        assert len(horizon) == 2

    def test_init_sorts_dates(self) -> None:
        """Test that dates are sorted."""
        dates = [datetime.date(2024, 3, 1), datetime.date(2024, 1, 1), datetime.date(2024, 2, 1)]
        horizon = TimeHorizon(dates)

        assert horizon.dates[0] == datetime.date(2024, 1, 1)
        assert horizon.dates[1] == datetime.date(2024, 2, 1)
        assert horizon.dates[2] == datetime.date(2024, 3, 1)

    def test_from_numbers_days_only(self) -> None:
        """Test from_numbers with days only."""
        start_date = datetime.date(2024, 1, 1)
        horizon = TimeHorizon.from_numbers(start_date, number_of_days=3)

        expected_dates = [
            datetime.date(2024, 1, 1),
            datetime.date(2024, 1, 2),
            datetime.date(2024, 1, 3),
            datetime.date(2024, 1, 4),
        ]
        assert horizon.dates == expected_dates

    def test_from_numbers_weeks_only(self) -> None:
        """Test from_numbers with weeks only."""
        start_date = datetime.date(2024, 1, 1)
        horizon = TimeHorizon.from_numbers(start_date, number_of_weeks=2)

        expected_dates = [datetime.date(2024, 1, 1), datetime.date(2024, 1, 8), datetime.date(2024, 1, 15)]
        assert horizon.dates == expected_dates

    def test_from_numbers_months_only(self) -> None:
        """Test from_numbers with months only."""
        start_date = datetime.date(2024, 1, 1)
        horizon = TimeHorizon.from_numbers(start_date, number_of_months=2)

        expected_dates = [datetime.date(2024, 1, 1), datetime.date(2024, 2, 1), datetime.date(2024, 3, 1)]
        assert horizon.dates == expected_dates

    def test_from_numbers_months_end_of_month_auto_detect(self) -> None:
        """Test from_numbers with months and auto-detect end_of_month."""
        start_date = datetime.date(2024, 1, 31)  # End of month
        horizon = TimeHorizon.from_numbers(start_date, number_of_months=2)

        expected_dates = [
            datetime.date(2024, 1, 31),
            datetime.date(2024, 2, 29),  # Leap year
            datetime.date(2024, 3, 31),
        ]
        assert horizon.dates == expected_dates

    def test_from_numbers_months_end_of_month_explicit_true(self) -> None:
        """Test from_numbers with months and explicit end_of_month=True."""
        start_date = datetime.date(2024, 1, 15)
        horizon = TimeHorizon.from_numbers(start_date, number_of_months=2, end_of_month=True)

        expected_dates = [
            datetime.date(2024, 1, 15),
            datetime.date(2024, 2, 29),  # Leap year
            datetime.date(2024, 3, 31),
        ]
        assert horizon.dates == expected_dates

    def test_from_numbers_months_end_of_month_explicit_false(self) -> None:
        """Test from_numbers with months and explicit end_of_month=False."""
        start_date = datetime.date(2024, 1, 31)  # End of month
        horizon = TimeHorizon.from_numbers(start_date, number_of_months=2, end_of_month=False)

        expected_dates = [
            datetime.date(2024, 1, 31),
            datetime.date(2024, 2, 29),  # Feb 31 -> Feb 29 (leap year)
            datetime.date(2024, 3, 31),  # March has 31 days
        ]
        assert horizon.dates == expected_dates

    def test_from_numbers_quarters_only(self) -> None:
        """Test from_numbers with quarters only."""
        start_date = datetime.date(2024, 1, 1)
        horizon = TimeHorizon.from_numbers(start_date, number_of_quarters=2)

        expected_dates = [
            datetime.date(2024, 1, 1),
            datetime.date(2024, 2, 1),  # 3 months = 1 quarter
            datetime.date(2024, 5, 1),  # 6 months = 2 quarters
        ]
        assert horizon.dates == expected_dates

    def test_from_numbers_years_only(self) -> None:
        """Test from_numbers with years only."""
        start_date = datetime.date(2024, 1, 1)
        horizon = TimeHorizon.from_numbers(start_date, number_of_years=2)

        expected_dates = [datetime.date(2024, 1, 1), datetime.date(2025, 1, 1), datetime.date(2026, 1, 1)]
        assert horizon.dates == expected_dates

    def test_from_numbers_mixed_periods(self) -> None:
        """Test from_numbers with mixed time periods."""
        start_date = datetime.date(2024, 1, 1)
        horizon = TimeHorizon.from_numbers(
            start_date, number_of_days=1, number_of_weeks=1, number_of_months=1, number_of_years=1
        )

        expected_dates = [
            datetime.date(2024, 1, 1),  # start
            datetime.date(2024, 1, 2),  # +1 day
            datetime.date(2024, 1, 8),  # +1 week
            datetime.date(2024, 2, 1),  # +1 month
            datetime.date(2025, 1, 1),  # +1 year
        ]
        assert horizon.dates == expected_dates

    def test_from_numbers_no_periods(self) -> None:
        """Test from_numbers with no periods specified."""
        start_date = datetime.date(2024, 1, 1)
        horizon = TimeHorizon.from_numbers(start_date)

        assert horizon.dates == [datetime.date(2024, 1, 1)]

    def test_iter_single_date(self) -> None:
        """Test iteration with single date."""
        dates = [datetime.date(2024, 1, 1)]
        horizon = TimeHorizon(dates)

        increments = list(horizon)
        assert len(increments) == 1
        assert increments[0].from_date == datetime.date(2024, 1, 1)
        assert increments[0].to_date == datetime.date(2024, 1, 1)

    def test_iter_multiple_dates(self) -> None:
        """Test iteration with multiple dates."""
        dates = [datetime.date(2024, 1, 1), datetime.date(2024, 1, 15), datetime.date(2024, 2, 1)]
        horizon = TimeHorizon(dates)

        increments = list(horizon)
        assert len(increments) == 3

        # First increment: start_date to start_date (zero-day increment)
        assert increments[0].from_date == datetime.date(2024, 1, 1)
        assert increments[0].to_date == datetime.date(2024, 1, 1)

        # Second increment: first date to second date
        assert increments[1].from_date == datetime.date(2024, 1, 1)
        assert increments[1].to_date == datetime.date(2024, 1, 15)

        # Third increment: second date to third date
        assert increments[2].from_date == datetime.date(2024, 1, 15)
        assert increments[2].to_date == datetime.date(2024, 2, 1)

    def test_iter_is_iterator(self) -> None:
        """Test that TimeHorizon is properly iterable."""
        dates = [datetime.date(2024, 1, 1), datetime.date(2024, 1, 15)]
        horizon = TimeHorizon(dates)

        # Should be able to iterate multiple times
        increments1 = list(horizon)
        increments2 = list(horizon)

        assert len(increments1) == len(increments2) == 2
        assert increments1[0].from_date == increments2[0].from_date
        assert increments1[1].to_date == increments2[1].to_date

    def test_len_property(self) -> None:
        """Test __len__ method."""
        dates = [datetime.date(2024, 1, 1), datetime.date(2024, 2, 1), datetime.date(2024, 3, 1)]
        horizon = TimeHorizon(dates)

        assert len(horizon) == 3

    def test_start_end_date_properties(self) -> None:
        """Test start_date and end_date properties."""
        dates = [datetime.date(2024, 3, 1), datetime.date(2024, 1, 1), datetime.date(2024, 2, 1)]
        horizon = TimeHorizon(dates)

        assert horizon.start_date == datetime.date(2024, 1, 1)
        assert horizon.end_date == datetime.date(2024, 3, 1)
