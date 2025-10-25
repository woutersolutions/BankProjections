import datetime
import os
import tempfile
from unittest.mock import MagicMock, Mock, patch

import polars as pl
import pytest

from bank_projections.financials.balance_sheet import BalanceSheet
from bank_projections.financials.market_data import MarketData, MarketRates
from bank_projections.projections.projection import Projection, ProjectionResult
from bank_projections.projections.rule import Rule
from bank_projections.scenarios.scenario import Scenario
from bank_projections.utils.time import TimeHorizon, TimeIncrement


class TestProjectionResult:
    def test_projection_result_initialization(self):
        balance_sheets = [pl.DataFrame({"col1": [1, 2], "col2": [3, 4]})]
        pnls = [pl.DataFrame({"pnl_col": [10, 20]})]
        cashflows = [pl.DataFrame({"cf_col": [100, 200]})]
        ocis = [pl.DataFrame({"oci_col": [5, 10]})]
        metric_list = [pl.DataFrame({"metric": [5]})]
        profitability_list = [pl.DataFrame({"profitability": [0.05]})]
        run_info: dict[str, int | datetime.date | datetime.datetime | float] = {
            "StartDate": datetime.date(2023, 1, 1),
            "EndDate": datetime.date(2023, 1, 31),
            "NumberOfIncrements": 1,
            "Scenarios": 1,
        }

        result = ProjectionResult(balance_sheets, pnls, cashflows, ocis, metric_list, profitability_list, run_info)

        assert result.balance_sheets == balance_sheets
        assert result.pnls == pnls
        assert result.cashflows == cashflows
        assert result.ocis == ocis
        assert result.metric_list == metric_list
        assert result.profitability_list == profitability_list
        assert result.run_info == run_info

    def test_projection_result_to_dict(self):
        """Test converting ProjectionResult to dictionary."""
        balance_sheets = [
            pl.DataFrame(
                {
                    "Scenario": ["base"],
                    "ProjectionDate": [datetime.date(2023, 1, 31)],
                    "asset": [1000],
                    "liability": [-800],
                }
            ),
            pl.DataFrame(
                {
                    "Scenario": ["base"],
                    "ProjectionDate": [datetime.date(2023, 2, 28)],
                    "asset": [1100],
                    "liability": [-900],
                }
            ),
        ]
        pnls = [
            pl.DataFrame({"Scenario": ["base"], "ProjectionDate": [datetime.date(2023, 1, 31)], "income": [50]}),
            pl.DataFrame({"Scenario": ["base"], "ProjectionDate": [datetime.date(2023, 2, 28)], "income": [60]}),
        ]
        cashflows = [
            pl.DataFrame({"Scenario": ["base"], "ProjectionDate": [datetime.date(2023, 1, 31)], "cash_in": [100]}),
            pl.DataFrame({"Scenario": ["base"], "ProjectionDate": [datetime.date(2023, 2, 28)], "cash_in": [110]}),
        ]
        ocis = [
            pl.DataFrame({"Scenario": ["base"], "ProjectionDate": [datetime.date(2023, 1, 31)], "oci": [5]}),
            pl.DataFrame({"Scenario": ["base"], "ProjectionDate": [datetime.date(2023, 2, 28)], "oci": [6]}),
        ]
        metric_list = [
            pl.DataFrame({"Scenario": ["base"], "ProjectionDate": [datetime.date(2023, 1, 31)], "metric": [5]}),
            pl.DataFrame({"Scenario": ["base"], "ProjectionDate": [datetime.date(2023, 2, 28)], "metric": [6]}),
        ]
        profitability_list = [
            pl.DataFrame({"Scenario": ["base"], "ProjectionDate": [datetime.date(2023, 1, 31)], "profitability": [0.05]}),
            pl.DataFrame({"Scenario": ["base"], "ProjectionDate": [datetime.date(2023, 2, 28)], "profitability": [0.06]}),
        ]
        run_info: dict[str, int | datetime.date | datetime.datetime | float] = {
            "StartDate": datetime.date(2023, 1, 1),
            "EndDate": datetime.date(2023, 2, 28),
            "NumberOfIncrements": 2,
            "Scenarios": 1,
        }

        result = ProjectionResult(balance_sheets, pnls, cashflows, ocis, metric_list, profitability_list, run_info)
        result_dict = result.to_dict()

        assert "BalanceSheets" in result_dict
        assert "P&Ls" in result_dict
        assert "Cashflows" in result_dict
        assert "OCIs" in result_dict
        assert "RunInfo" in result_dict

        # Check that Scenario and ProjectionDate columns exist
        assert "Scenario" in result_dict["BalanceSheets"].columns
        assert "ProjectionDate" in result_dict["BalanceSheets"].columns
        assert "Scenario" in result_dict["P&Ls"].columns
        assert "ProjectionDate" in result_dict["P&Ls"].columns
        assert "Scenario" in result_dict["Cashflows"].columns
        assert "ProjectionDate" in result_dict["Cashflows"].columns
        assert "Scenario" in result_dict["OCIs"].columns
        assert "ProjectionDate" in result_dict["OCIs"].columns

    def test_projection_result_to_excel(self):
        """Test exporting ProjectionResult to Excel."""
        balance_sheets = [
            pl.DataFrame({"Scenario": ["base"], "ProjectionDate": [datetime.date(2023, 1, 31)], "asset": [1000]})
        ]
        pnls = [pl.DataFrame({"Scenario": ["base"], "ProjectionDate": [datetime.date(2023, 1, 31)], "income": [50]})]
        cashflows = [
            pl.DataFrame({"Scenario": ["base"], "ProjectionDate": [datetime.date(2023, 1, 31)], "cash_in": [100]})
        ]
        ocis = [pl.DataFrame({"Scenario": ["base"], "ProjectionDate": [datetime.date(2023, 1, 31)], "oci": [5]})]
        metric_list = [
            pl.DataFrame({"Scenario": ["base"], "ProjectionDate": [datetime.date(2023, 1, 31)], "metric": [5]})
        ]
        profitability_list = [
            pl.DataFrame({"Scenario": ["base"], "ProjectionDate": [datetime.date(2023, 1, 31)], "profitability": [0.05]})
        ]
        run_info: dict[str, int | datetime.date | datetime.datetime | float] = {
            "StartDate": datetime.date(2023, 1, 1),
            "EndDate": datetime.date(2023, 1, 31),
            "NumberOfIncrements": 1,
            "Scenarios": 1,
        }

        result = ProjectionResult(balance_sheets, pnls, cashflows, ocis, metric_list, profitability_list, run_info)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test_output.xlsx")
            result.to_excel(file_path)

            # Check that a file was created (with timestamp suffix)
            files = os.listdir(temp_dir)
            excel_files = [f for f in files if f.startswith("test_output_") and f.endswith(".xlsx")]
            assert len(excel_files) == 1


class TestProjection:
    def test_projection_initialization(self):
        mock_scenario = Mock(spec=Scenario)
        mock_horizon = Mock(spec=TimeHorizon)

        projection = Projection({"base": mock_scenario}, mock_horizon)

        assert "base" in projection.scenarios
        assert projection.scenarios["base"] == mock_scenario
        assert projection.horizon == mock_horizon

    @patch("bank_projections.projections.projection.calculate_metrics")
    def test_projection_run_single_increment(self, mock_calculate_metrics):
        # Create mock dependencies
        mock_rule = Mock(spec=Rule)
        mock_bs = MagicMock(spec=BalanceSheet)
        mock_increment = Mock(spec=TimeIncrement)
        mock_increment.from_date = datetime.date(2023, 1, 1)
        mock_increment.to_date = datetime.date(2023, 1, 31)

        # Mock the balance sheet methods
        mock_bs.__len__.return_value = 1000
        mock_bs.validate.return_value = None
        mock_bs_copy = Mock(spec=BalanceSheet)
        mock_bs.copy.return_value = mock_bs_copy
        mock_bs_initialized = Mock(spec=BalanceSheet)
        mock_bs_copy.initialize_new_date.return_value = mock_bs_initialized

        # Mock rule apply method - note this receives the initialized balance sheet
        mock_rule.apply.return_value = mock_bs_initialized

        # Mock aggregate method
        mock_agg_bs = pl.DataFrame({"asset": [1000]})
        mock_pnl = pl.DataFrame({"income": [50]})
        mock_cashflow = pl.DataFrame({"cash": [100]})
        mock_oci = pl.DataFrame({"oci": [10]})
        mock_bs_initialized.aggregate.return_value = (mock_agg_bs, mock_pnl, mock_cashflow, mock_oci)
        mock_bs_initialized.validate.return_value = None

        # Mock calculate_metrics to return a DataFrame
        mock_metrics = pl.DataFrame({"CET1 Ratio": [0.15], "Leverage Ratio": [0.05]})
        mock_calculate_metrics.return_value = mock_metrics

        # Create scenario with market data
        mock_market_data = Mock(spec=MarketData)
        mock_market_rates = Mock(spec=MarketRates)
        mock_market_data.get_market_rates.return_value = mock_market_rates
        scenario = Scenario(rules={"rule1": mock_rule}, market_data=mock_market_data)

        # Create horizon mock
        mock_horizon = MagicMock()
        mock_horizon.__len__.return_value = 1
        mock_horizon.__iter__.return_value = iter([mock_increment])

        projection = Projection({"base": scenario}, mock_horizon)
        result = projection.run(mock_bs)

        # Verify calls
        mock_rule.apply.assert_called_once_with(mock_bs_initialized, mock_increment, mock_market_rates)
        mock_bs_initialized.aggregate.assert_called_once()
        mock_bs_initialized.validate.assert_called_once()

        # Verify result
        assert isinstance(result, ProjectionResult)
        assert len(result.balance_sheets) == 1
        assert len(result.pnls) == 1
        assert len(result.cashflows) == 1
        assert "run_info" in dir(result)

    @patch("bank_projections.projections.projection.calculate_metrics")
    def test_projection_run_multiple_increments(self, mock_calculate_metrics):
        # Create mock dependencies
        mock_rule1 = Mock(spec=Rule)
        mock_rule2 = Mock(spec=Rule)
        mock_bs = MagicMock(spec=BalanceSheet)
        mock_bs.__len__.return_value = 1000

        mock_increment1 = Mock(spec=TimeIncrement)
        mock_increment1.from_date = datetime.date(2023, 1, 1)
        mock_increment1.to_date = datetime.date(2023, 1, 31)

        mock_increment2 = Mock(spec=TimeIncrement)
        mock_increment2.from_date = datetime.date(2023, 2, 1)
        mock_increment2.to_date = datetime.date(2023, 2, 28)

        # Mock the balance sheet methods
        mock_bs.validate.return_value = None
        mock_bs_copy = Mock(spec=BalanceSheet)
        mock_bs.copy.return_value = mock_bs_copy
        mock_bs_initialized1 = Mock(spec=BalanceSheet)
        mock_bs_initialized2 = Mock(spec=BalanceSheet)

        # Setup multiple initialize_new_date calls for multiple increments
        # But we also need to set up chained calls as each iteration reassigns bs
        mock_bs_copy.initialize_new_date.return_value = mock_bs_initialized1
        mock_bs_initialized1.initialize_new_date.return_value = mock_bs_initialized2

        # Mock aggregate method - return fresh dataframes for each call
        def make_aggregate_return():
            return (
                pl.DataFrame({"asset": [1000]}),
                pl.DataFrame({"income": [50]}),
                pl.DataFrame({"cash": [100]}),
                pl.DataFrame({"oci": [10]}),
            )

        mock_bs_initialized1.aggregate.side_effect = [make_aggregate_return()]
        mock_bs_initialized1.validate.return_value = None
        mock_bs_initialized2.aggregate.side_effect = [make_aggregate_return()]
        mock_bs_initialized2.validate.return_value = None

        # Mock calculate_metrics to return a DataFrame
        mock_metrics = pl.DataFrame({"CET1 Ratio": [0.15], "Leverage Ratio": [0.05]})
        mock_calculate_metrics.return_value = mock_metrics

        # Create scenario with market data
        mock_market_data = Mock(spec=MarketData)
        mock_market_rates = Mock(spec=MarketRates)
        mock_market_data.get_market_rates.return_value = mock_market_rates
        composite_rule = Scenario(rules={"1": mock_rule1, "2": mock_rule2}, market_data=mock_market_data)

        # Mock the composite rule's apply method to simulate applying both rules
        # The apply method should return the balance sheet that was passed in
        composite_rule.apply = Mock(side_effect=lambda bs, increment, market_rates: bs)

        # Create horizon mock
        mock_horizon = MagicMock()
        mock_horizon.__len__.return_value = 2
        mock_horizon.__iter__.return_value = iter([mock_increment1, mock_increment2])

        projection = Projection({"base": composite_rule}, mock_horizon)
        result = projection.run(mock_bs)

        # Verify the composite rule is applied for each increment
        assert composite_rule.apply.call_count == 2

        # Verify aggregate is called for each increment
        mock_bs_initialized1.aggregate.assert_called_once()
        mock_bs_initialized2.aggregate.assert_called_once()

        # Verify validate is called for each increment
        mock_bs_initialized1.validate.assert_called_once()
        mock_bs_initialized2.validate.assert_called_once()

        # Verify result structure - now one result per increment (not per rule)
        assert isinstance(result, ProjectionResult)
        assert len(result.balance_sheets) == 2  # 2 increments
        assert len(result.pnls) == 2
        assert len(result.cashflows) == 2

    @patch("bank_projections.projections.projection.calculate_metrics")
    def test_projection_run_no_rules(self, mock_calculate_metrics):
        mock_bs = MagicMock(spec=BalanceSheet)
        mock_bs.__len__.return_value = 1000
        mock_increment = Mock(spec=TimeIncrement)
        mock_increment.from_date = datetime.date(2023, 1, 1)
        mock_increment.to_date = datetime.date(2023, 1, 31)

        # Mock the balance sheet methods
        mock_bs.validate.return_value = None
        mock_bs_copy = Mock(spec=BalanceSheet)
        mock_bs.copy.return_value = mock_bs_copy
        mock_bs_initialized = Mock(spec=BalanceSheet)
        mock_bs_copy.initialize_new_date.return_value = mock_bs_initialized

        # Mock aggregate method
        mock_agg_bs = pl.DataFrame({"asset": [1000]})
        mock_pnl = pl.DataFrame({"income": [50]})
        mock_cashflow = pl.DataFrame({"cash": [100]})
        mock_oci = pl.DataFrame({"oci": [10]})
        mock_bs_initialized.aggregate.return_value = (mock_agg_bs, mock_pnl, mock_cashflow, mock_oci)
        mock_bs_initialized.validate.return_value = None

        # Mock calculate_metrics to return a DataFrame
        mock_metrics = pl.DataFrame({"CET1 Ratio": [0.15], "Leverage Ratio": [0.05]})
        mock_calculate_metrics.return_value = mock_metrics

        # Create horizon mock with single increment
        mock_horizon = MagicMock()
        mock_horizon.__len__.return_value = 1
        mock_horizon.__iter__.return_value = iter([mock_increment])

        # Create scenario with market data but no rules
        mock_market_data = Mock(spec=MarketData)
        mock_market_rates = Mock(spec=MarketRates)
        mock_market_data.get_market_rates.return_value = mock_market_rates
        empty_rule = Scenario(rules={}, market_data=mock_market_data)
        projection = Projection({"base": empty_rule}, mock_horizon)
        result = projection.run(mock_bs)

        # Verify calls
        mock_bs_initialized.validate.assert_called_once()
        mock_bs_initialized.aggregate.assert_called_once()

        # Verify result - even with empty rules, we still get one result per increment
        assert isinstance(result, ProjectionResult)
        assert len(result.balance_sheets) == 1
        assert len(result.pnls) == 1
        assert len(result.cashflows) == 1

    def test_projection_run_empty_horizon(self):
        mock_bs = MagicMock(spec=BalanceSheet)
        mock_bs.__len__.return_value = 1000

        # Create empty horizon mock
        mock_horizon = MagicMock()
        mock_horizon.__len__.return_value = 0
        mock_horizon.__iter__.return_value = iter([])
        mock_horizon.start_date = datetime.date(2023, 1, 1)
        mock_horizon.end_date = datetime.date(2023, 1, 1)

        mock_scenario = Mock(spec=Scenario)
        projection = Projection({"base": mock_scenario}, mock_horizon)
        result = projection.run(mock_bs)

        # Verify no calls were made (no rule.apply to check since we use empty horizon)
        mock_bs.validate.assert_not_called()

        # Verify empty result
        assert isinstance(result, ProjectionResult)
        assert len(result.balance_sheets) == 0
        assert len(result.pnls) == 0
        assert len(result.cashflows) == 0

    def test_projection_run_rule_exception_handling(self):
        """Test that exceptions in rule application are propagated."""
        mock_rule = Mock(spec=Rule)
        mock_bs = MagicMock(spec=BalanceSheet)
        mock_bs.__len__.return_value = 1000
        mock_increment = Mock(spec=TimeIncrement)
        mock_increment.from_date = datetime.date(2023, 1, 1)
        mock_increment.to_date = datetime.date(2023, 1, 31)

        # Mock copy method to return a copy of the balance sheet
        mock_bs.copy.return_value = mock_bs

        # Make rule apply raise an exception
        mock_rule.apply.side_effect = ValueError("Rule application failed")

        # Create scenario with market data
        mock_market_data = Mock(spec=MarketData)
        mock_market_rates = Mock(spec=MarketRates)
        mock_market_data.get_market_rates.return_value = mock_market_rates
        scenario = Scenario(rules={"rule1": mock_rule}, market_data=mock_market_data)

        # Create horizon mock
        mock_horizon = MagicMock()
        mock_horizon.__len__.return_value = 1
        mock_horizon.__iter__.return_value = iter([mock_increment])

        projection = Projection({"base": scenario}, mock_horizon)

        with pytest.raises(ValueError, match="Rule application failed"):
            projection.run(mock_bs)
