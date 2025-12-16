import datetime

import pandas as pd
import pytest

from bank_projections.financials.market_data import Curves
from bank_projections.scenarios.excel_sheet_format import KeyValueInput, TableInput
from bank_projections.scenarios.scenario import Scenario
from bank_projections.utils.time import TimeIncrement
from examples.synthetic_data import create_synthetic_balance_sheet


@pytest.fixture(scope="session")
def minimal_scenario():
    """Create a minimal scenario for testing that includes market data.

    Session-scoped to avoid recreating for every test.
    """
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
    # Provide manual repayment rate for Intangible assets (which use manual redemption type)
    mutations_df = pd.DataFrame(
        {
            "SubItemType": ["Intangible assets"],
            "metric": ["repaymentrate"],
            "Amount": [0.0],  # No repayment for intangible assets
        }
    )
    mutations_input = TableInput(
        table=mutations_df,
        general_tags={},
        template_name="balancesheetmutations",
    )
    return Scenario(excel_inputs=[curve_input, tax_input, audit_input, mutations_input])


@pytest.fixture(scope="session")
def minimal_scenario_snapshot(minimal_scenario):
    """Create a ScenarioSnapShot for testing.

    Session-scoped to avoid recreating for every test.
    """
    increment = TimeIncrement(datetime.date(2024, 1, 1), datetime.date(2024, 12, 31))
    return minimal_scenario.snapshot_at(increment)


@pytest.fixture
def sample_curves():
    """Provide sample curves for testing."""
    df = pd.DataFrame(
        {
            "Name": ["euribor", "euribor", "euribor", "euribor"],
            "Type": ["spot", "spot", "zero", "zero"],
            "Tenor": ["3m", "6m", "1y", "5y"],
            "Rate": [0.0285, 0.0305, 0.030, 0.032],
        }
    )
    return Curves(df)


@pytest.fixture(scope="session")
def _test_balance_sheet_cached(minimal_scenario):
    """Create a test balance sheet once per session.

    Private fixture - use test_balance_sheet instead to get a fresh copy.
    """
    bs = create_synthetic_balance_sheet(current_date=datetime.date(2024, 12, 31), scenario=minimal_scenario)
    bs.validate()
    return bs


@pytest.fixture
def test_balance_sheet(_test_balance_sheet_cached):
    """Provide a fresh copy of the test balance sheet for each test.

    Returns a copy to ensure test isolation while avoiding expensive recreation.
    """
    return _test_balance_sheet_cached.copy()
