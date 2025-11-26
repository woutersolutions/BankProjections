"""Unit tests for daycount fraction module."""

import datetime

import polars as pl
import pytest

from bank_projections.projections.daycount_fraction import (
    Actual360,
    Actual365Fixed,
    Actual36525,
    ActualActualISDA,
    DaycountFractionRegistry,
    Thirty360BondBasis,
    Thirty360European,
)


class TestDaycountFractionRegistry:
    """Test DaycountFractionRegistry functionality."""

    def setup_method(self) -> None:
        """Save registry state and clear for isolated testing."""
        self._original_items = DaycountFractionRegistry.items.copy()
        self._original_stripped = DaycountFractionRegistry.stripped_items.copy()
        DaycountFractionRegistry.items.clear()
        DaycountFractionRegistry.stripped_items.clear()

    def teardown_method(self) -> None:
        """Restore original registry state."""
        DaycountFractionRegistry.items.clear()
        DaycountFractionRegistry.stripped_items.clear()
        DaycountFractionRegistry.items.update(self._original_items)
        DaycountFractionRegistry.stripped_items.update(self._original_stripped)

    def test_register_valid_daycount(self) -> None:
        """Test registering a valid daycount fraction class."""
        item = Actual360()
        DaycountFractionRegistry.register("actual360", item)
        assert "actual360" in DaycountFractionRegistry.items
        assert item in DaycountFractionRegistry.items.values()

    def test_year_fraction_with_registered_daycount(self) -> None:
        """Test year_fraction works with registered daycount fractions."""
        DaycountFractionRegistry.register("actual360", Actual360())

        df = pl.DataFrame(
            {
                "DaycountBasis": ["actual360"],
                "StartDate": [datetime.date(2025, 1, 1)],
                "EndDate": [datetime.date(2025, 7, 1)],
            }
        )

        result = df.with_columns(
            year_fraction=DaycountFractionRegistry.year_fraction(pl.col("StartDate"), pl.col("EndDate"))
        )

        # 181 days / 360 = 0.502778
        assert result["year_fraction"][0] == pytest.approx(181 / 360, rel=1e-6)


class TestActual360:
    """Test Actual/360 daycount convention."""

    def test_year_fraction_basic(self) -> None:
        """Test basic year fraction calculation."""
        df = pl.DataFrame(
            {
                "start_date": [datetime.date(2025, 1, 1)],
                "end_date": [datetime.date(2025, 4, 1)],
            }
        )

        result = df.with_columns(
            fraction=Actual360.year_fraction(pl.col("start_date"), pl.col("end_date"))
        )

        # 90 days / 360 = 0.25
        assert result["fraction"][0] == pytest.approx(90 / 360, rel=1e-6)

    def test_year_fraction_full_year(self) -> None:
        """Test full year fraction calculation."""
        df = pl.DataFrame(
            {
                "start_date": [datetime.date(2025, 1, 1)],
                "end_date": [datetime.date(2026, 1, 1)],
            }
        )

        result = df.with_columns(
            fraction=Actual360.year_fraction(pl.col("start_date"), pl.col("end_date"))
        )

        # 365 days / 360 > 1.0
        assert result["fraction"][0] == pytest.approx(365 / 360, rel=1e-6)


class TestActual365Fixed:
    """Test Actual/365 Fixed daycount convention."""

    def test_year_fraction_basic(self) -> None:
        """Test basic year fraction calculation."""
        df = pl.DataFrame(
            {
                "start_date": [datetime.date(2025, 1, 1)],
                "end_date": [datetime.date(2025, 4, 1)],
            }
        )

        result = df.with_columns(
            fraction=Actual365Fixed.year_fraction(pl.col("start_date"), pl.col("end_date"))
        )

        # 90 days / 365
        assert result["fraction"][0] == pytest.approx(90 / 365, rel=1e-6)

    def test_year_fraction_full_year(self) -> None:
        """Test full year equals 1.0."""
        df = pl.DataFrame(
            {
                "start_date": [datetime.date(2025, 1, 1)],
                "end_date": [datetime.date(2026, 1, 1)],
            }
        )

        result = df.with_columns(
            fraction=Actual365Fixed.year_fraction(pl.col("start_date"), pl.col("end_date"))
        )

        # 365 days / 365 = 1.0
        assert result["fraction"][0] == pytest.approx(1.0, rel=1e-6)


class TestActual36525:
    """Test Actual/365.25 daycount convention."""

    def test_year_fraction_basic(self) -> None:
        """Test basic year fraction calculation."""
        df = pl.DataFrame(
            {
                "start_date": [datetime.date(2025, 1, 1)],
                "end_date": [datetime.date(2025, 4, 1)],
            }
        )

        result = df.with_columns(
            fraction=Actual36525.year_fraction(pl.col("start_date"), pl.col("end_date"))
        )

        # 90 days / 365.25
        assert result["fraction"][0] == pytest.approx(90 / 365.25, rel=1e-6)

    def test_year_fraction_four_years(self) -> None:
        """Test four years accounting for leap year."""
        df = pl.DataFrame(
            {
                "start_date": [datetime.date(2024, 1, 1)],
                "end_date": [datetime.date(2028, 1, 1)],
            }
        )

        result = df.with_columns(
            fraction=Actual36525.year_fraction(pl.col("start_date"), pl.col("end_date"))
        )

        # 1461 days (including leap day) / 365.25 ≈ 4.0
        assert result["fraction"][0] == pytest.approx(4.0, rel=1e-6)


class TestActualActualISDA:
    """Test Actual/Actual ISDA daycount convention."""

    def test_year_fraction_non_leap_year(self) -> None:
        """Test year fraction in a non-leap year."""
        df = pl.DataFrame(
            {
                "start_date": [datetime.date(2025, 1, 1)],
                "end_date": [datetime.date(2025, 4, 1)],
            }
        )

        result = df.with_columns(
            fraction=ActualActualISDA.year_fraction(pl.col("start_date"), pl.col("end_date"))
        )

        # 90 days / 365 (2025 is not a leap year)
        assert result["fraction"][0] == pytest.approx(90 / 365, rel=1e-6)

    def test_year_fraction_leap_year(self) -> None:
        """Test year fraction in a leap year."""
        df = pl.DataFrame(
            {
                "start_date": [datetime.date(2024, 1, 1)],
                "end_date": [datetime.date(2024, 4, 1)],
            }
        )

        result = df.with_columns(
            fraction=ActualActualISDA.year_fraction(pl.col("start_date"), pl.col("end_date"))
        )

        # 91 days / 366 (2024 is a leap year)
        assert result["fraction"][0] == pytest.approx(91 / 366, rel=1e-6)


class TestThirty360BondBasis:
    """Test 30/360 Bond Basis daycount convention."""

    def test_year_fraction_basic(self) -> None:
        """Test basic 30/360 calculation."""
        df = pl.DataFrame(
            {
                "start_date": [datetime.date(2025, 1, 15)],
                "end_date": [datetime.date(2025, 4, 15)],
            }
        )

        result = df.with_columns(
            fraction=Thirty360BondBasis.year_fraction(pl.col("start_date"), pl.col("end_date"))
        )

        # 3 months * 30 days / 360 = 90 / 360 = 0.25
        assert result["fraction"][0] == pytest.approx(0.25, rel=1e-6)

    def test_year_fraction_end_of_month_31(self) -> None:
        """Test 30/360 with day 31 adjustment."""
        df = pl.DataFrame(
            {
                "start_date": [datetime.date(2025, 1, 31)],
                "end_date": [datetime.date(2025, 3, 31)],
            }
        )

        result = df.with_columns(
            fraction=Thirty360BondBasis.year_fraction(pl.col("start_date"), pl.col("end_date"))
        )

        # Start day 31 -> 30, end day 31 -> 30 (since start >= 30)
        # 2 months * 30 = 60 days / 360 = 1/6
        assert result["fraction"][0] == pytest.approx(60 / 360, rel=1e-6)

    def test_year_fraction_full_year(self) -> None:
        """Test full year calculation."""
        df = pl.DataFrame(
            {
                "start_date": [datetime.date(2025, 1, 1)],
                "end_date": [datetime.date(2026, 1, 1)],
            }
        )

        result = df.with_columns(
            fraction=Thirty360BondBasis.year_fraction(pl.col("start_date"), pl.col("end_date"))
        )

        # 12 months * 30 days / 360 = 360 / 360 = 1.0
        assert result["fraction"][0] == pytest.approx(1.0, rel=1e-6)


class TestThirty360European:
    """Test 30E/360 (Eurobond Basis) daycount convention."""

    def test_year_fraction_basic(self) -> None:
        """Test basic 30E/360 calculation."""
        df = pl.DataFrame(
            {
                "start_date": [datetime.date(2025, 1, 15)],
                "end_date": [datetime.date(2025, 4, 15)],
            }
        )

        result = df.with_columns(
            fraction=Thirty360European.year_fraction(pl.col("start_date"), pl.col("end_date"))
        )

        # 3 months * 30 days / 360 = 90 / 360 = 0.25
        assert result["fraction"][0] == pytest.approx(0.25, rel=1e-6)

    def test_year_fraction_end_of_month_31(self) -> None:
        """Test 30E/360 with day 31 adjustment."""
        df = pl.DataFrame(
            {
                "start_date": [datetime.date(2025, 1, 31)],
                "end_date": [datetime.date(2025, 3, 31)],
            }
        )

        result = df.with_columns(
            fraction=Thirty360European.year_fraction(pl.col("start_date"), pl.col("end_date"))
        )

        # Both days 31 -> 30
        # 2 months * 30 = 60 days / 360 = 1/6
        assert result["fraction"][0] == pytest.approx(60 / 360, rel=1e-6)


class TestAllDaycountFractionsReturnValidValues:
    """Test that all registered daycount fractions return valid numeric values."""

    def test_all_fractions_return_positive_for_positive_period(self) -> None:
        """Verify all registered daycount fractions return positive values for a positive period."""
        # Create test data with all registered daycount conventions
        conventions = DaycountFractionRegistry.stripped_names()

        df = pl.DataFrame(
            {
                "DaycountBasis": conventions,
                "StartDate": [datetime.date(2025, 1, 1)] * len(conventions),
                "EndDate": [datetime.date(2025, 7, 1)] * len(conventions),
            }
        )

        result = df.with_columns(
            year_fraction=DaycountFractionRegistry.year_fraction(pl.col("StartDate"), pl.col("EndDate"))
        )

        # All fractions should be positive and finite
        for i, convention in enumerate(conventions):
            fraction = result["year_fraction"][i]
            assert fraction is not None, f"{convention} returned None"
            assert fraction > 0, f"{convention} returned non-positive value: {fraction}"
            assert fraction < 10, f"{convention} returned suspiciously large value: {fraction}"

    def test_all_fractions_return_zero_for_same_date(self) -> None:
        """Verify all registered daycount fractions return zero when dates are the same."""
        conventions = DaycountFractionRegistry.stripped_names()

        df = pl.DataFrame(
            {
                "DaycountBasis": conventions,
                "StartDate": [datetime.date(2025, 6, 15)] * len(conventions),
                "EndDate": [datetime.date(2025, 6, 15)] * len(conventions),
            }
        )

        result = df.with_columns(
            year_fraction=DaycountFractionRegistry.year_fraction(pl.col("StartDate"), pl.col("EndDate"))
        )

        for i, convention in enumerate(conventions):
            fraction = result["year_fraction"][i]
            assert fraction == pytest.approx(0.0, abs=1e-10), f"{convention} did not return 0 for same date: {fraction}"

    def test_all_fractions_are_approximately_one_for_full_year(self) -> None:
        """Verify all registered daycount fractions return approximately 1.0 for a full year."""
        conventions = DaycountFractionRegistry.stripped_names()

        df = pl.DataFrame(
            {
                "DaycountBasis": conventions,
                "StartDate": [datetime.date(2025, 1, 1)] * len(conventions),
                "EndDate": [datetime.date(2026, 1, 1)] * len(conventions),
            }
        )

        result = df.with_columns(
            year_fraction=DaycountFractionRegistry.year_fraction(pl.col("StartDate"), pl.col("EndDate"))
        )

        for i, convention in enumerate(conventions):
            fraction = result["year_fraction"][i]
            assert fraction is not None, f"{convention} returned None"
            # All conventions should return something close to 1.0 for a full year
            # Actual/360 returns 365/360 ≈ 1.014, which is the largest deviation
            assert 0.9 < fraction < 1.1, f"{convention} returned unexpected value for full year: {fraction}"

    def test_all_individual_implementations_return_valid_fractions(self) -> None:
        """Test each individual daycount fraction class directly returns valid fractions."""
        implementations = [
            (Actual360, "Actual360"),
            (Actual365Fixed, "Actual365Fixed"),
            (Actual36525, "Actual36525"),
            (ActualActualISDA, "ActualActualISDA"),
            (Thirty360BondBasis, "Thirty360BondBasis"),
            (Thirty360European, "Thirty360European"),
        ]

        df = pl.DataFrame(
            {
                "start_date": [datetime.date(2025, 3, 15)],
                "end_date": [datetime.date(2025, 9, 15)],
            }
        )

        for impl_class, name in implementations:
            result = df.with_columns(
                fraction=impl_class.year_fraction(pl.col("start_date"), pl.col("end_date"))
            )

            fraction = result["fraction"][0]
            assert fraction is not None, f"{name} returned None"
            assert isinstance(fraction, float), f"{name} did not return float: {type(fraction)}"
            assert 0 < fraction < 1, f"{name} returned invalid 6-month fraction: {fraction}"
