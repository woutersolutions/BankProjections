"""Tests for scenario module."""

import datetime
from unittest.mock import Mock

from bank_projections.financials.balance_sheet import BalanceSheet
from bank_projections.projections.market_data import MarketData, MarketRates
from bank_projections.projections.rule import Rule
from bank_projections.projections.time import TimeIncrement
from bank_projections.scenarios.scenario import Scenario


class MockRule(Rule):
    """Mock rule for testing."""

    def apply(self, bs: BalanceSheet, increment: TimeIncrement, market_rates: MarketRates) -> BalanceSheet:
        return bs


class TestScenario:
    """Test Scenario functionality."""

    def test_scenario_init_minimal(self):
        """Test Scenario initialization with minimal parameters."""
        scenario = Scenario()

        assert isinstance(scenario.rules, dict)
        assert len(scenario.rules) == 0
        assert isinstance(scenario.market_data, MarketData)

    def test_scenario_init_with_rules(self):
        """Test Scenario initialization with rules."""
        rule1 = MockRule()
        rule2 = MockRule()
        rules = {"rule1": rule1, "rule2": rule2}

        scenario = Scenario(rules=rules)

        assert scenario.rules == rules
        assert len(scenario.rules) == 2

    def test_scenario_init_with_market_data(self):
        """Test Scenario initialization with market data."""
        market_data = MarketData()
        scenario = Scenario(market_data=market_data)

        assert scenario.market_data is market_data

    def test_scenario_apply_no_rules(self):
        """Test applying scenario with no rules."""
        scenario = Scenario()
        mock_bs = Mock(spec=BalanceSheet)
        increment = TimeIncrement(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
        market_rates = Mock(spec=MarketRates)

        result = scenario.apply(mock_bs, increment, market_rates)

        # With no rules, the balance sheet should be returned unchanged
        assert result is mock_bs

    def test_scenario_apply_with_rules(self):
        """Test applying scenario with rules."""
        rule1 = Mock(spec=Rule)
        rule2 = Mock(spec=Rule)
        rules = {"rule1": rule1, "rule2": rule2}

        scenario = Scenario(rules=rules)
        mock_bs = Mock(spec=BalanceSheet)
        increment = TimeIncrement(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
        market_rates = Mock(spec=MarketRates)

        # Mock the rule applications to return the same balance sheet
        rule1.apply.return_value = mock_bs
        rule2.apply.return_value = mock_bs

        result = scenario.apply(mock_bs, increment, market_rates)

        # Both rules should be applied
        rule1.apply.assert_called_once_with(mock_bs, increment, market_rates)
        rule2.apply.assert_called_once_with(mock_bs, increment, market_rates)
        assert result is mock_bs

    def test_scenario_apply_rule_chaining(self):
        """Test that rules are applied in sequence (chained)."""
        rule1 = Mock(spec=Rule)
        rule2 = Mock(spec=Rule)
        rules = {"rule1": rule1, "rule2": rule2}

        scenario = Scenario(rules=rules)

        mock_bs_initial = Mock(spec=BalanceSheet)
        mock_bs_after_rule1 = Mock(spec=BalanceSheet)
        mock_bs_after_rule2 = Mock(spec=BalanceSheet)

        increment = TimeIncrement(datetime.date(2023, 1, 1), datetime.date(2023, 1, 31))
        market_rates = Mock(spec=MarketRates)

        # Set up the chain: rule1 transforms initial BS, rule2 transforms result of rule1
        rule1.apply.return_value = mock_bs_after_rule1
        rule2.apply.return_value = mock_bs_after_rule2

        result = scenario.apply(mock_bs_initial, increment, market_rates)

        # Verify the chaining
        rule1.apply.assert_called_once_with(mock_bs_initial, increment, market_rates)
        rule2.apply.assert_called_once_with(mock_bs_after_rule1, increment, market_rates)
        assert result is mock_bs_after_rule2
