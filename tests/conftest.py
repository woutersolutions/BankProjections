import datetime

import pandas as pd
import pytest

from bank_projections.financials.market_data import CurveData, MarketData
from bank_projections.scenarios.scenario import Scenario
from examples.synthetic_data import create_synthetic_balance_sheet


@pytest.fixture
def minimal_scenario():
    """Create a minimal scenario for testing that includes market data."""
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
    curve_data = CurveData(curve_data_df)
    market_data = MarketData(curves=curve_data)
    return Scenario(market_data=market_data)


@pytest.fixture
def test_balance_sheet(minimal_scenario):
    """Create a test balance sheet with the minimal scenario."""
    bs = create_synthetic_balance_sheet(current_date=datetime.date(2024, 12, 31), scenario=minimal_scenario)
    bs.validate()
    return bs
