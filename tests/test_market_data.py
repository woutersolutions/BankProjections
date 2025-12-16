"""Tests for market data module."""

import pandas as pd
import pytest

from bank_projections.financials.market_data import Curves, parse_tenor


class TestCurves:
    """Test Curves functionality."""

    def test_curves_init_empty(self):
        """Test Curves initialization with no data."""
        curves = Curves()
        assert len(curves.data) == 0

    def test_curves_init_with_data(self):
        """Test Curves initialization with data."""
        df = pd.DataFrame(
            {"Name": ["euribor", "euribor"], "Type": ["spot", "spot"], "Tenor": ["3m", "6m"], "Rate": [0.03, 0.035]}
        )
        curves = Curves(df)
        assert len(curves.data) == 2

    def test_get_spot_rates(self):
        """Test getting spot rates from curves."""
        df = pd.DataFrame(
            {"Name": ["euribor", "euribor"], "Type": ["spot", "spot"], "Tenor": ["3m", "6m"], "Rate": [0.03, 0.035]}
        )
        curves = Curves(df)
        spot_rates = curves.get_spot_rates()

        assert isinstance(spot_rates, dict)
        assert "euribor3m" in spot_rates
        assert "euribor6m" in spot_rates
        assert spot_rates["euribor3m"] == 0.03
        assert spot_rates["euribor6m"] == 0.035

    def test_get_zero_rates(self):
        """Test getting zero rates from curves."""
        df = pd.DataFrame(
            {
                "Name": ["euribor", "euribor"],
                "Type": ["zero", "zero"],
                "MaturityYears": [1.0, 5.0],
                "Rate": [0.03, 0.035],
            }
        )
        curves = Curves(df)
        zero_rates = curves.get_zero_rates()

        assert isinstance(zero_rates, pd.DataFrame)
        assert "Name" in zero_rates.columns
        assert "MaturityYears" in zero_rates.columns
        assert "Rate" in zero_rates.columns

    def test_floating_rate_expr_with_data(self):
        """Test floating rate expression with valid data."""
        df = pd.DataFrame(
            {"Name": ["euribor", "libor"], "Type": ["spot", "spot"], "Tenor": ["3m", "6m"], "Rate": [0.03, 0.035]}
        )
        curves = Curves(df)
        expr = curves.floating_rate_expr()

        # This should return a polars expression
        assert expr is not None


class TestParseTenor:
    """Test parse_tenor function."""

    def test_parse_tenor_days(self):
        """Test parsing tenor with days."""
        assert parse_tenor("1d") == pytest.approx(1 / 365.25, rel=1e-6)
        assert parse_tenor("7d") == pytest.approx(7 / 365.25, rel=1e-6)

    def test_parse_tenor_weeks(self):
        """Test parsing tenor with weeks."""
        assert parse_tenor("1w") == pytest.approx(7 / 365.25, rel=1e-6)
        assert parse_tenor("2w") == pytest.approx(14 / 365.25, rel=1e-6)

    def test_parse_tenor_months(self):
        """Test parsing tenor with months."""
        assert parse_tenor("1m") == pytest.approx(1 / 12, rel=1e-6)
        assert parse_tenor("3m") == pytest.approx(3 / 12, rel=1e-6)

    def test_parse_tenor_years(self):
        """Test parsing tenor with years."""
        assert parse_tenor("1y") == pytest.approx(1.0, rel=1e-6)
        assert parse_tenor("5y") == pytest.approx(5.0, rel=1e-6)

    def test_parse_tenor_none(self):
        """Test parsing None tenor."""
        assert parse_tenor(None) is None

    def test_parse_tenor_invalid_unit(self):
        """Test parsing tenor with invalid unit."""
        with pytest.raises(ValueError, match="Unknown tenor unit"):
            parse_tenor("1x")

    def test_parse_tenor_case_insensitive(self):
        """Test that parsing is case insensitive."""
        assert parse_tenor("1M") == pytest.approx(1 / 12, rel=1e-6)
        assert parse_tenor("1Y") == pytest.approx(1.0, rel=1e-6)
