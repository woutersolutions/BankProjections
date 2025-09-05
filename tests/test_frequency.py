"""Unit tests for frequency module."""

import datetime

import polars as pl
import pytest

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
        """Clear registry before each test."""
        FrequencyRegistry._registry.clear()

    def test_register_valid_frequency(self) -> None:
        """Test registering a valid frequency class."""
        FrequencyRegistry.register("Monthly", Monthly)
        assert "Monthly" in FrequencyRegistry._registry
        assert FrequencyRegistry._registry["Monthly"] is Monthly

    def test_register_invalid_frequency(self) -> None:
        """Test registering an invalid class raises ValueError."""

        class NotAFrequency:
            pass

        with pytest.raises(ValueError, match="must be a subclass of Frequency"):
            FrequencyRegistry.register("Invalid", NotAFrequency)

    def test_advance_next_with_registered_frequency(self) -> None:
        """Test advance_next works with registered frequencies."""
        FrequencyRegistry.register("Monthly", Monthly)

        df = pl.DataFrame(
            {
                "CouponFrequency": ["Monthly"],
                "Date": [datetime.date(2025, 1, 15)],
            }
        )

        result = df.with_columns(next_date=FrequencyRegistry.advance_next(pl.col("Date"), pl.lit(2)))

        expected_date = datetime.date(2025, 3, 15)
        assert result["next_date"][0] == expected_date

    def test_number_due_with_registered_frequency(self) -> None:
        """Test number_due works with registered frequencies."""
        FrequencyRegistry.register("Monthly", Monthly)

        df = pl.DataFrame(
            {
                "CouponFrequency": ["Monthly"],
                "CouponDate": [datetime.date(2025, 1, 15)],
                "ProjectionDate": [datetime.date(2025, 4, 15)],
            }
        )

        result = df.with_columns(
            payments_due=FrequencyRegistry.number_due(pl.col("CouponDate"), pl.col("ProjectionDate"))
        )

        assert result["payments_due"][0] == 4  # Jan, Feb, Mar, Apr

    def test_portion_passed_with_registered_frequency(self) -> None:
        """Test portion_passed works with registered frequencies."""
        FrequencyRegistry.register("Monthly", Monthly)

        df = pl.DataFrame(
            {
                "CouponFrequency": ["Monthly"],
                "NextCouponDate": [datetime.date(2025, 2, 15)],
            }
        )

        projection_date = datetime.date(2025, 2, 1)
        result = df.with_columns(portion=FrequencyRegistry.portion_passed(pl.col("NextCouponDate"), projection_date))

        expected_portion = 1 - (15 - 1) / 30  # 14 days passed out of 30
        assert abs(result["portion"][0] - expected_portion) < 0.01


class TestMonthly:
    """Test Monthly frequency class."""

    def test_advance_next(self) -> None:
        """Test advancing date by number of months."""
        df = pl.DataFrame({"date": [datetime.date(2025, 1, 15)]})

        result = df.with_columns(next_date=Monthly.advance_next(pl.col("date"), pl.lit(3)))

        expected_date = datetime.date(2025, 4, 15)
        assert result["next_date"][0] == expected_date

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

    def test_portion_passed(self) -> None:
        """Test calculating portion of period passed."""
        df = pl.DataFrame({"next_coupon": [datetime.date(2025, 2, 15)]})

        projection_date = datetime.date(2025, 2, 1)
        result = df.with_columns(portion=Monthly.portion_passed(pl.col("next_coupon"), projection_date))

        expected_portion = 1 - (15 - 1) / 30
        assert abs(result["portion"][0] - expected_portion) < 0.01


class TestQuarterly:
    """Test Quarterly frequency class."""

    def test_advance_next(self) -> None:
        """Test advancing date by quarters."""
        df = pl.DataFrame({"date": [datetime.date(2025, 1, 15)]})

        result = df.with_columns(next_date=Quarterly.advance_next(pl.col("date"), pl.lit(2)))

        expected_date = datetime.date(2025, 7, 15)  # 6 months later
        assert result["next_date"][0] == expected_date

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

    def test_portion_passed(self) -> None:
        """Test calculating portion passed for quarterly."""
        df = pl.DataFrame({"next_coupon": [datetime.date(2025, 4, 15)]})

        projection_date = datetime.date(2025, 3, 15)
        result = df.with_columns(portion=Quarterly.portion_passed(pl.col("next_coupon"), projection_date))

        expected_portion = 1 - (31) / 90  # ~1 month left of 3-month quarter
        assert abs(result["portion"][0] - expected_portion) < 0.05


class TestSemiAnnual:
    """Test SemiAnnual frequency class."""

    def test_advance_next(self) -> None:
        """Test advancing date by semi-annual periods."""
        df = pl.DataFrame({"date": [datetime.date(2025, 1, 15)]})

        result = df.with_columns(next_date=SemiAnnual.advance_next(pl.col("date"), pl.lit(1)))

        expected_date = datetime.date(2025, 7, 15)  # 6 months later
        assert result["next_date"][0] == expected_date

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

    def test_advance_next(self) -> None:
        """Test advancing date by annual periods."""
        df = pl.DataFrame({"date": [datetime.date(2025, 1, 15)]})

        result = df.with_columns(next_date=Annual.advance_next(pl.col("date"), pl.lit(2)))

        expected_date = datetime.date(2027, 1, 15)  # 2 years later
        assert result["next_date"][0] == expected_date

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

    def test_advance_next(self) -> None:
        """Test advancing date by days."""
        df = pl.DataFrame({"date": [datetime.date(2025, 1, 15)]})

        result = df.with_columns(next_date=Daily.advance_next(pl.col("date"), pl.lit(30)))

        expected_date = datetime.date(2025, 2, 14)
        assert result["next_date"][0] == expected_date

    def test_number_due(self) -> None:
        """Test calculating daily payments due."""
        df = pl.DataFrame(
            {
                "coupon_date": [datetime.date(2025, 1, 15)],
                "projection_date": [datetime.date(2025, 1, 20)],
            }
        )

        result = df.with_columns(payments=Daily.number_due(pl.col("coupon_date"), pl.col("projection_date")))

        assert result["payments"][0] == 5  # 5 days

    def test_portion_passed(self) -> None:
        """Test portion passed for daily (always 0)."""
        df = pl.DataFrame({"next_coupon": [datetime.date(2025, 2, 15)]})

        projection_date = datetime.date(2025, 2, 10)
        result = df.with_columns(portion=Daily.portion_passed(pl.col("next_coupon"), projection_date))

        assert result["portion"][0] == -4
