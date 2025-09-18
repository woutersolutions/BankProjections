"""Tests for market data module."""

import datetime

import pandas as pd
import pytest

from bank_projections.projections.market_data import CurveData, Curves, MarketData, MarketRates, parse_tenor


class TestCurveData:
    """Test CurveData functionality."""

    def test_curve_data_init_empty(self):
        """Test CurveData initialization with no data."""
        curve_data = CurveData()
        assert len(curve_data.data) == 0
        expected_columns = ["Date", "Name", "Type", "Tenor", "Maturity", "Rate", "MaturityYears"]
        assert list(curve_data.data.columns) == expected_columns

    def test_curve_data_init_with_data(self):
        """Test CurveData initialization with data."""
        df = pd.DataFrame(
            {
                "Date": [pd.Timestamp(2023, 1, 1)],
                "Name": ["euribor"],
                "Type": ["spot"],
                "Tenor": ["3m"],
                "Maturity": ["3m"],
                "Rate": [0.03],
            }
        )
        curve_data = CurveData(df)
        assert len(curve_data.data) == 1
        assert curve_data.data["Name"].iloc[0] == "euribor"

    def test_curve_data_combine(self):
        """Test combining two CurveData objects."""
        df1 = pd.DataFrame(
            {
                "Date": [pd.Timestamp(2023, 1, 1)],
                "Name": ["euribor"],
                "Type": ["spot"],
                "Tenor": ["3m"],
                "Maturity": ["3m"],
                "Rate": [0.03],
            }
        )
        df2 = pd.DataFrame(
            {
                "Date": [pd.Timestamp(2023, 1, 2)],
                "Name": ["libor"],
                "Type": ["spot"],
                "Tenor": ["6m"],
                "Maturity": ["6m"],
                "Rate": [0.035],
            }
        )

        curve1 = CurveData(df1)
        curve2 = CurveData(df2)
        combined = curve1.combine(curve2)

        assert len(combined.data) == 2
        assert "euribor" in combined.data["Name"].values
        assert "libor" in combined.data["Name"].values

    def test_get_curves_valid_date(self):
        """Test getting curves for a valid date."""
        df = pd.DataFrame(
            {
                "Date": [pd.Timestamp(2023, 1, 1), pd.Timestamp(2023, 1, 2)],
                "Name": ["euribor", "euribor"],
                "Type": ["spot", "spot"],
                "Tenor": ["3m", "6m"],
                "Maturity": ["3m", "6m"],
                "Rate": [0.03, 0.035],
            }
        )
        curve_data = CurveData(df)
        curves = curve_data.get_curves(datetime.date(2023, 1, 2))

        assert isinstance(curves, Curves)
        assert len(curves.data) == 1  # Only the latest date

    def test_get_curves_no_data_available(self):
        """Test getting curves when no data is available for the date."""
        df = pd.DataFrame(
            {
                "Date": [pd.Timestamp(2023, 1, 1)],
                "Name": ["euribor"],
                "Type": ["spot"],
                "Tenor": ["3m"],
                "Maturity": ["3m"],
                "Rate": [0.03],
            }
        )
        curve_data = CurveData(df)

        with pytest.raises(ValueError, match="No curve data available for date"):
            curve_data.get_curves(datetime.date(2022, 12, 31))


class TestCurves:
    """Test Curves functionality."""

    def test_curves_init_empty(self):
        """Test Curves initialization with no data."""
        curves = Curves()
        assert len(curves.data) == 0

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

    def test_floating_rate_expr_with_data(self):
        """Test floating rate expression with valid data."""
        df = pd.DataFrame(
            {"Name": ["euribor", "libor"], "Type": ["spot", "spot"], "Tenor": ["3m", "6m"], "Rate": [0.03, 0.035]}
        )
        curves = Curves(df)
        expr = curves.floating_rate_expr()

        # This should return a polars expression
        assert expr is not None


class TestMarketData:
    """Test MarketData functionality."""

    def test_market_data_init_empty(self):
        """Test MarketData initialization with no curves."""
        market_data = MarketData()
        assert isinstance(market_data.curves, CurveData)
        assert len(market_data.curves.data) == 0

    def test_market_data_init_with_curves(self):
        """Test MarketData initialization with curves."""
        curve_data = CurveData()
        market_data = MarketData(curve_data)
        assert market_data.curves is curve_data

    def test_market_data_combine(self):
        """Test combining two MarketData objects."""
        df1 = pd.DataFrame(
            {
                "Date": [pd.Timestamp(2023, 1, 1)],
                "Name": ["euribor"],
                "Type": ["spot"],
                "Tenor": ["3m"],
                "Maturity": ["3m"],
                "Rate": [0.03],
            }
        )
        df2 = pd.DataFrame(
            {
                "Date": [pd.Timestamp(2023, 1, 2)],
                "Name": ["libor"],
                "Type": ["spot"],
                "Tenor": ["6m"],
                "Maturity": ["6m"],
                "Rate": [0.035],
            }
        )

        market1 = MarketData(CurveData(df1))
        market2 = MarketData(CurveData(df2))
        combined = market1.combine(market2)

        assert isinstance(combined, MarketData)
        assert len(combined.curves.data) == 2

    def test_get_market_rates(self):
        """Test getting market rates for a specific date."""
        df = pd.DataFrame(
            {
                "Date": [pd.Timestamp(2023, 1, 1)],
                "Name": ["euribor"],
                "Type": ["spot"],
                "Tenor": ["3m"],
                "Maturity": ["3m"],
                "Rate": [0.03],
            }
        )
        market_data = MarketData(CurveData(df))
        market_rates = market_data.get_market_rates(datetime.date(2023, 1, 1))

        assert isinstance(market_rates, MarketRates)


class TestMarketRates:
    """Test MarketRates functionality."""

    def test_market_rates_init_empty(self):
        """Test MarketRates initialization with no curves."""
        market_rates = MarketRates()
        assert isinstance(market_rates.curves, Curves)

    def test_market_rates_init_with_curves(self):
        """Test MarketRates initialization with curves."""
        curves = Curves()
        market_rates = MarketRates(curves)
        assert market_rates.curves is curves


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
