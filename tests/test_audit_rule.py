import datetime

import polars as pl
import pytest

from bank_projections.financials.balance_sheet_item import BalanceSheetItem, BalanceSheetItemRegistry
from bank_projections.projections.market_data import MarketRates
from bank_projections.projections.time import TimeIncrement
from bank_projections.scenarios.audit import AuditRule
from examples.synthetic_data import create_synthetic_balance_sheet


class TestAuditRule:
    def test_init_minimal(self):
        rule_input = {
            "ClosingMonth": 12,
            "AuditMonth": 3,
            "TargetItemType": "Mortgages",
        }
        rule = AuditRule(rule_input)

        assert rule.closing_month == 12
        assert rule.audit_month == 3
        assert rule.target.identifiers["ItemType"] == "Mortgages"

    def test_init_with_multiple_target_labels(self):
        rule_input = {
            "ClosingMonth": 12,
            "AuditMonth": 3,
            "TargetItemType": "Mortgages",
            "TargetCurrency": "EUR",
        }
        rule = AuditRule(rule_input)

        assert rule.target.identifiers["ItemType"] == "Mortgages"
        assert rule.target.identifiers["Currency"] == "EUR"

    def test_init_with_unrecognized_key(self):
        rule_input = {
            "ClosingMonth": 12,
            "AuditMonth": 3,
            "UnknownKey": "value",
        }

        with pytest.raises(KeyError, match="UnknownKey not recognized in AuditRule"):
            AuditRule(rule_input)

    def test_apply_no_audit_in_increment(self):
        rule_input = {
            "ClosingMonth": 12,
            "AuditMonth": 3,
            "TargetItemType": "Mortgages",
        }
        rule = AuditRule(rule_input)

        bs = create_synthetic_balance_sheet(datetime.date(2024, 1, 1))
        increment = TimeIncrement(datetime.date(2024, 1, 1), datetime.date(2024, 2, 28))
        market_rates = MarketRates()

        result = rule.apply(bs, increment, market_rates)

        assert result == bs

    def test_apply_with_audit_in_increment(self):
        rule_input = {
            "ClosingMonth": 12,
            "AuditMonth": 3,
            "TargetItemType": "Mortgages",
        }
        rule = AuditRule(rule_input)

        bs = create_synthetic_balance_sheet(datetime.date(2024, 1, 1))
        increment = TimeIncrement(datetime.date(2024, 1, 1), datetime.date(2024, 3, 31))
        market_rates = MarketRates()

        result = rule.apply(bs, increment, market_rates)

        assert result is not None

    def test_apply_audit_date_calculation(self):
        rule_input = {
            "ClosingMonth": 12,
            "AuditMonth": 3,
            "TargetItemType": "Mortgages",
        }
        rule = AuditRule(rule_input)

        bs = create_synthetic_balance_sheet(datetime.date(2024, 1, 1))
        increment = TimeIncrement(datetime.date(2024, 3, 1), datetime.date(2024, 3, 31))
        market_rates = MarketRates()

        result = rule.apply(bs, increment, market_rates)

        assert result is not None

    def test_apply_no_audit_before_audit_month(self):
        rule_input = {
            "ClosingMonth": 12,
            "AuditMonth": 6,
            "TargetItemType": "Mortgages",
        }
        rule = AuditRule(rule_input)

        bs = create_synthetic_balance_sheet(datetime.date(2024, 1, 1))
        increment = TimeIncrement(datetime.date(2024, 1, 1), datetime.date(2024, 3, 31))
        market_rates = MarketRates()

        result = rule.apply(bs, increment, market_rates)

        assert result == bs
