"""Unit tests for CostIncomeRule module."""

import datetime

from bank_projections.scenarios.cost_income import CostIncomeRule
from bank_projections.utils.time import TimeIncrement
from examples.synthetic_data import create_synthetic_balance_sheet


class TestCostIncomeRule:
    """Test CostIncomeRule class."""

    def test_cost_income_rule_instantiation(self):
        """Test that CostIncomeRule can be instantiated."""
        rule = CostIncomeRule()
        assert rule is not None

    def test_apply_no_cost_income_in_snapshot(self, minimal_scenario, minimal_scenario_snapshot):
        """Test applying cost_income rule when there's no cost_income in snapshot."""
        rule = CostIncomeRule()

        bs = create_synthetic_balance_sheet(datetime.date(2024, 1, 1), scenario=minimal_scenario)
        increment = TimeIncrement(datetime.date(2024, 1, 1), datetime.date(2024, 1, 31))

        # minimal_scenario_snapshot has no cost_income items
        result = rule.apply(bs, increment, minimal_scenario_snapshot)

        assert result is not None
        assert result == bs  # No changes since no cost_income items
