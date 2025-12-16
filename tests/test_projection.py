"""Tests for projection module."""

import datetime
import os
import tempfile
from unittest.mock import MagicMock, Mock

import polars as pl

from bank_projections.financials.balance_sheet import BalanceSheet
from bank_projections.projections.projection import Projection, ProjectionResult
from bank_projections.projections.projectionrule import ProjectionRule
from bank_projections.scenarios.scenario import Scenario
from bank_projections.utils.time import TimeHorizon


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
            pl.DataFrame(
                {"Scenario": ["base"], "ProjectionDate": [datetime.date(2023, 1, 31)], "profitability": [0.05]}
            ),
            pl.DataFrame(
                {"Scenario": ["base"], "ProjectionDate": [datetime.date(2023, 2, 28)], "profitability": [0.06]}
            ),
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
            pl.DataFrame(
                {"Scenario": ["base"], "ProjectionDate": [datetime.date(2023, 1, 31)], "profitability": [0.05]}
            )
        ]
        run_info: dict[str, int | datetime.date | datetime.datetime | float] = {
            "StartDate": datetime.date(2023, 1, 1),
            "EndDate": datetime.date(2023, 1, 31),
            "NumberOfIncrements": 1,
            "Scenarios": 1,
        }

        result = ProjectionResult(balance_sheets, pnls, cashflows, ocis, metric_list, profitability_list, run_info)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Use strftime format in filename as the new implementation expects
            file_path = os.path.join(temp_dir, "test_output_%Y%m%d_%H%M%S.xlsx")
            result.to_excel(file_path)

            # Check that a file was created (with strftime-formatted timestamp)
            files = os.listdir(temp_dir)
            excel_files = [f for f in files if f.startswith("test_output_") and f.endswith(".xlsx")]
            assert len(excel_files) == 1


class TestProjection:
    def test_projection_initialization(self):
        mock_scenario = Mock(spec=Scenario)
        mock_horizon = Mock(spec=TimeHorizon)
        mock_rules = {"rule1": Mock(spec=ProjectionRule)}

        projection = Projection({"base": mock_scenario}, mock_horizon, mock_rules)

        assert "base" in projection.scenarios
        assert projection.scenarios["base"] == mock_scenario
        assert projection.horizon == mock_horizon
        assert projection.rules == mock_rules

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
        mock_rules: dict[str, ProjectionRule] = {}
        projection = Projection({"base": mock_scenario}, mock_horizon, mock_rules)
        result = projection.run(mock_bs)

        # Verify no calls were made (no rule.apply to check since we use empty horizon)
        mock_bs.validate.assert_not_called()

        # Verify empty result
        assert isinstance(result, ProjectionResult)
        assert len(result.balance_sheets) == 0
        assert len(result.pnls) == 0
        assert len(result.cashflows) == 0
