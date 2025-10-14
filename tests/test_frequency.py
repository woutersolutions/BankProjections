"""Unit tests for frequency module."""

import datetime

import polars as pl

from bank_projections.projections.frequency import (
    Annual,
    Daily,
    FrequencyRegistry,
    Monthly,
    Quarterly,
    SemiAnnual,
)


class TestFrequencyRegistry:
    """Test FrequencyRegistry functionality."""

    def setup_method(self) -> None:
        """Save registry state and clear for isolated testing."""
        self._original_registry = FrequencyRegistry.items.copy()
        FrequencyRegistry.items.clear()

    def teardown_method(self) -> None:
        """Restore original registry state."""
        FrequencyRegistry.items.clear()
        FrequencyRegistry.items.update(self._original_registry)

    def test_register_valid_frequency(self) -> None:
        """Test registering a valid frequency class."""
        item = Monthly()
        FrequencyRegistry.register("Monthly", item)
        assert "Monthly" in FrequencyRegistry.items
        assert item in FrequencyRegistry.items.values()

    def test_number_due_with_registered_frequency(self) -> None:
        """Test number_due works with registered frequencies."""
        FrequencyRegistry.register("Monthly", Monthly())

        df = pl.DataFrame(
            {
                "CouponFrequency": ["monthly"],
                "CouponDate": [datetime.date(2025, 1, 15)],
                "ProjectionDate": [datetime.date(2025, 4, 15)],
            }
        )

        result = df.with_columns(
            payments_due=FrequencyRegistry.number_due(pl.col("CouponDate"), pl.col("ProjectionDate"))
        )

        assert result["payments_due"][0] == 4  # Jan, Feb, Mar, Apr


class TestMonthly:
    """Test Monthly frequency class."""

    def test_number_due(self) -> None:
        """Test calculating number of payments due."""
        df = pl.DataFrame(
            {
                "coupon_date": [datetime.date(2025, 1, 15)],
                "projection_date": [datetime.date(2025, 4, 20)],
            }
        )

        result = df.with_columns(payments=Monthly.number_due(pl.col("coupon_date"), pl.col("projection_date")))

        assert result["payments"][0] == 4  # Jan, Feb, Mar, Apr (day >= 15)


class TestQuarterly:
    """Test Quarterly frequency class."""

    def test_number_due(self) -> None:
        """Test calculating quarterly payments due."""
        df = pl.DataFrame(
            {
                "coupon_date": [datetime.date(2025, 1, 15)],
                "projection_date": [datetime.date(2025, 7, 20)],
            }
        )

        result = df.with_columns(payments=Quarterly.number_due(pl.col("coupon_date"), pl.col("projection_date")))

        assert result["payments"][0] == 3  # Q1, Q2, Q3


class TestSemiAnnual:
    """Test SemiAnnual frequency class."""

    def test_number_due(self) -> None:
        """Test calculating semi-annual payments due."""
        df = pl.DataFrame(
            {
                "coupon_date": [datetime.date(2025, 1, 15)],
                "projection_date": [datetime.date(2025, 12, 20)],
            }
        )

        result = df.with_columns(payments=SemiAnnual.number_due(pl.col("coupon_date"), pl.col("projection_date")))

        assert result["payments"][0] == 2  # Two semi-annual payments


class TestAnnual:
    """Test Annual frequency class."""

    def test_number_due(self) -> None:
        """Test calculating annual payments due."""
        df = pl.DataFrame(
            {
                "coupon_date": [datetime.date(2024, 1, 15)],
                "projection_date": [datetime.date(2026, 2, 20)],
            }
        )

        result = df.with_columns(payments=Annual.number_due(pl.col("coupon_date"), pl.col("projection_date")))

        assert result["payments"][0] == 3  # 2024, 2025, 2026


class TestDaily:
    """Test Daily frequency class."""

    def test_number_due(self) -> None:
        """Test calculating daily payments due."""
        df = pl.DataFrame(
            {
                "coupon_date": [datetime.date(2025, 1, 15)],
                "projection_date": [datetime.date(2025, 1, 20)],
            }
        )

        result = df.with_columns(payments=Daily.number_due(pl.col("coupon_date"), pl.col("projection_date")))

        assert result["payments"][0] == 6  # 6 days (inclusive: 15, 16, 17, 18, 19, 20)
