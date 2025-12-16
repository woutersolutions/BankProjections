"""Tests for production rule."""

import datetime

from bank_projections.scenarios.production import ProductionRule
from bank_projections.utils.time import TimeIncrement
from examples.synthetic_data import create_synthetic_balance_sheet


class TestProductionRule:
    """Test ProductionRule class."""

    def test_production_rule_instantiation(self):
        """Test that ProductionRule can be instantiated."""
        rule = ProductionRule()
        assert rule is not None

    def test_apply_no_production_in_snapshot(self, minimal_scenario, minimal_scenario_snapshot):
        """Test applying production rule when there's no production in snapshot."""
        rule = ProductionRule()

        bs = create_synthetic_balance_sheet(datetime.date(2024, 1, 1), scenario=minimal_scenario)
        increment = TimeIncrement(datetime.date(2024, 1, 1), datetime.date(2024, 1, 31))

        # minimal_scenario_snapshot has no production items
        result = rule.apply(bs, increment, minimal_scenario_snapshot)

        assert result is not None
        assert result == bs  # No changes since no production items
