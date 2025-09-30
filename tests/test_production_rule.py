import datetime

import pandas as pd
import pytest

from bank_projections.projections.market_data import MarketRates
from bank_projections.projections.time import TimeIncrement
from bank_projections.scenarios.production import ProductionRule
from examples.synthetic_data import create_synthetic_balance_sheet


class TestProductionRule:
    def test_init_minimal_with_reference_item_name(self):
        rule_input = {
            "ReferenceItem": "cash account",
            "Date": "2024-01-15",
            "ItemType": "Cash",
            "interest_rate": 0.03,
        }
        rule = ProductionRule(rule_input)

        assert rule.date == datetime.date(2024, 1, 15)
        assert rule.reference_item is not None
        assert rule.labels["ItemType"] == "Cash"
        assert rule.metrics["interestrate"] == 0.03
        assert rule.maturity is None
        assert rule.multiplicative is False

    def test_init_with_empty_reference_item(self):
        rule_input = {
            "ReferenceItem": "",
            "Date": "2024-01-15",
        }
        rule = ProductionRule(rule_input)

        assert rule.reference_item is None

    def test_init_with_na_reference_item(self):
        rule_input = {
            "ReferenceItem": pd.NA,
            "Date": "2024-01-15",
        }
        rule = ProductionRule(rule_input)

        assert rule.reference_item is None

    def test_init_with_maturity(self):
        rule_input = {
            "ReferenceItem": "cash account",
            "Date": "2024-01-15",
            "Maturity": 10,
            "interest_rate": 0.03,
        }
        rule = ProductionRule(rule_input)

        assert rule.maturity == 10

    def test_init_with_multiplicative(self):
        rule_input = {
            "ReferenceItem": "cash account",
            "Date": "2024-01-15",
            "Multiplicative": "true",
        }
        rule = ProductionRule(rule_input)

        assert rule.multiplicative is True

    def test_init_with_reference_labels(self):
        rule_input = {
            "Date": "2024-01-15",
            "ReferenceItemType": "Cash",
            "ReferenceCurrency": "EUR",
            "ItemType": "Cash",
        }
        rule = ProductionRule(rule_input)

        assert rule.reference_item is not None
        assert rule.reference_item.identifiers["ItemType"] == "Cash"
        assert rule.reference_item.identifiers["Currency"] == "EUR"
        assert rule.labels["ItemType"] == "Cash"

    def test_init_with_empty_values_skipped(self):
        rule_input = {
            "ReferenceItem": "cash account",
            "Date": "2024-01-15",
            "EmptyKey": "",
            "NaKey": pd.NA,
        }
        rule = ProductionRule(rule_input)

        assert rule.date == datetime.date(2024, 1, 15)

    def test_init_with_unrecognized_key(self):
        rule_input = {
            "ReferenceItem": "cash account",
            "Date": "2024-01-15",
            "UnknownKey": "value",
        }

        with pytest.raises(KeyError, match="UnknownKey not recognized in BalanceSheetMutationRule"):
            ProductionRule(rule_input)

    def test_apply_without_reference_item(self):
        rule_input = {
            "Date": "2024-01-15",
        }
        rule = ProductionRule(rule_input)

        bs = create_synthetic_balance_sheet(datetime.date(2024, 1, 1))
        increment = TimeIncrement(datetime.date(2024, 1, 1), datetime.date(2024, 1, 31))
        market_rates = MarketRates()

        with pytest.raises(NotImplementedError, match="Production without reference item not yet implement"):
            rule.apply(bs, increment, market_rates)

    def test_apply_date_not_in_increment(self):
        rule_input = {
            "ReferenceItem": "cash account",
            "Date": "2024-02-15",
            "ItemType": "Cash",
        }
        rule = ProductionRule(rule_input)

        bs = create_synthetic_balance_sheet(datetime.date(2024, 1, 1))
        increment = TimeIncrement(datetime.date(2024, 1, 1), datetime.date(2024, 1, 31))
        market_rates = MarketRates()

        result = rule.apply(bs, increment, market_rates)

        assert result == bs

    def test_apply_date_is_none(self):
        rule_input = {
            "ReferenceItem": "cash account",
        }
        rule = ProductionRule(rule_input)

        bs = create_synthetic_balance_sheet(datetime.date(2024, 1, 1))
        increment = TimeIncrement(datetime.date(2024, 1, 1), datetime.date(2024, 1, 31))
        market_rates = MarketRates()

        with pytest.raises(ValueError, match="Date must be specified for production"):
            rule.apply(bs, increment, market_rates)

