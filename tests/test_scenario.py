"""Tests for scenario module."""

import datetime

import pandas as pd

from bank_projections.scenarios.excel_sheet_format import KeyValueInput, TableInput
from bank_projections.scenarios.scenario import Scenario, ScenarioSnapShot
from bank_projections.utils.time import TimeIncrement


class TestScenario:
    """Test Scenario functionality."""

    def test_scenario_init_with_curve_input(self):
        """Test Scenario initialization with curve input."""
        curve_data_df = pd.DataFrame(
            {
                "Date": [datetime.date(2024, 1, 1)] * 4,
                "Name": ["euribor", "euribor", "euribor", "euribor"],
                "Tenor": ["3m", "6m", "1y", "5y"],
                "Type": ["spot", "spot", "zero", "zero"],
                "Rate": [0.0285, 0.0305, 0.030, 0.032],
                "Maturity": ["3m", "6m", "1y", "5y"],
            }
        )
        curve_input = TableInput(
            table=curve_data_df,
            general_tags={},
            template_name="interestrates",
        )
        tax_input = KeyValueInput(
            general_tags={"Tax Rate": 0.25},
            template_name="tax",
        )
        audit_input = KeyValueInput(
            general_tags={"ClosingMonth": 12, "AuditMonth": 3},
            template_name="audit",
        )

        scenario = Scenario(excel_inputs=[curve_input, tax_input, audit_input])
        assert scenario is not None

    def test_scenario_snapshot_at(self, minimal_scenario):
        """Test getting snapshot at a specific time increment."""
        increment = TimeIncrement(datetime.date(2024, 1, 1), datetime.date(2024, 1, 31))
        snapshot = minimal_scenario.snapshot_at(increment)

        assert isinstance(snapshot, ScenarioSnapShot)
        assert snapshot.curves is not None

    def test_scenario_snapshot_has_tax_info(self, minimal_scenario):
        """Test that snapshot contains tax information."""
        increment = TimeIncrement(datetime.date(2024, 1, 1), datetime.date(2024, 1, 31))
        snapshot = minimal_scenario.snapshot_at(increment)

        assert snapshot.tax is not None

    def test_scenario_snapshot_has_audit_info(self, minimal_scenario):
        """Test that snapshot contains audit information."""
        increment = TimeIncrement(datetime.date(2024, 1, 1), datetime.date(2024, 1, 31))
        snapshot = minimal_scenario.snapshot_at(increment)

        assert snapshot.audit is not None


class TestScenarioSnapShot:
    """Test ScenarioSnapShot functionality."""

    def test_snapshot_curves_available(self, minimal_scenario_snapshot):
        """Test that curves are available in snapshot."""
        assert minimal_scenario_snapshot.curves is not None

    def test_snapshot_production_available(self, minimal_scenario_snapshot):
        """Test that production is available in snapshot (may be empty list)."""
        assert minimal_scenario_snapshot.production is not None
        assert isinstance(minimal_scenario_snapshot.production, list)
