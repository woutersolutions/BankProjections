"""Unit tests for CostIncomeRule module."""

import datetime

import pytest

from bank_projections.financials.market_data import MarketRates
from bank_projections.scenarios.cost_income import CostIncomeRule
from bank_projections.utils.time import TimeIncrement
from examples.synthetic_data import generate_synthetic_curves


@pytest.fixture
def market_rates():
    """Provide market rates for testing."""
    return MarketRates(generate_synthetic_curves())


class TestCostIncomeRuleInit:
    """Test CostIncomeRule initialization and validation."""

    def test_init_minimal(self):
        """Test minimal initialization with required fields."""
        rule_input = {"date": "2025-01-15", "amount": 1000.0, "rule": "test"}
        rule = CostIncomeRule(rule_input)

        assert rule.cashflow_date == datetime.date(2025, 1, 15)
        assert rule.amount == 1000.0
        assert rule.rule == "test"
        assert rule.pnl_start is None
        assert rule.pnl_end is None

    def test_init_with_pnl_period(self):
        """Test initialization with P&L start and end dates."""
        rule_input = {
            "date": "2025-01-15",
            "amount": 1000.0,
            "rule": "test",
            "pnlstart": "2025-01-01",
            "pnlend": "2025-12-31",
        }
        rule = CostIncomeRule(rule_input)

        assert rule.pnl_start == datetime.date(2025, 1, 1)
        assert rule.pnl_end == datetime.date(2025, 12, 31)

    def test_init_missing_date_raises_error(self):
        """Test that missing date raises ValueError."""
        rule_input = {"amount": 1000.0, "rule": "test"}

        with pytest.raises(ValueError, match="Date must be specified"):
            CostIncomeRule(rule_input)

    def test_init_missing_amount_raises_error(self):
        """Test that missing amount raises ValueError."""
        rule_input = {"date": "2025-01-15", "rule": "test"}

        with pytest.raises(ValueError, match="Amount must be specified"):
            CostIncomeRule(rule_input)

    def test_init_pnl_start_without_end_raises_error(self):
        """Test that pnlstart without pnlend raises ValueError."""
        rule_input = {
            "date": "2025-01-15",
            "amount": 1000.0,
            "rule": "test",
            "pnlstart": "2025-01-01",
        }

        with pytest.raises(ValueError, match="Both pnlstart and pnlend must be specified"):
            CostIncomeRule(rule_input)

    def test_init_pnl_start_after_end_raises_error(self):
        """Test that pnlstart after pnlend raises ValueError."""
        rule_input = {
            "date": "2025-01-15",
            "amount": 1000.0,
            "rule": "test",
            "pnlstart": "2025-12-31",
            "pnlend": "2025-01-01",
        }

        with pytest.raises(ValueError, match="pnlstart must be before or equal to pnlend"):
            CostIncomeRule(rule_input)

    def test_init_unrecognized_key_raises_error(self):
        """Test that unrecognized key raises KeyError."""
        rule_input = {
            "date": "2025-01-15",
            "amount": 1000.0,
            "rule": "test",
            "unknown_key": "value",
        }

        with pytest.raises(KeyError, match="unknown_key not recognized"):
            CostIncomeRule(rule_input)


class TestCostIncomeRuleApply:
    """Test CostIncomeRule application to balance sheets."""

    def test_apply_immediate_recognition(self, test_balance_sheet, market_rates):
        """Test cost/income with immediate P&L recognition (no pnl period)."""
        rule_input = {"date": "2025-01-15", "amount": 1000.0, "rule": "immediate_income"}
        rule = CostIncomeRule(rule_input)

        increment = TimeIncrement(from_date=datetime.date(2025, 1, 1), to_date=datetime.date(2025, 1, 31))

        result_bs = rule.apply(test_balance_sheet, increment, market_rates)

        # Validate balance sheet remains balanced
        result_bs.validate()

        # Check that cashflows were generated
        assert len(result_bs.cashflows) > 0

    def test_apply_outside_increment(self, test_balance_sheet, market_rates):
        """Test that rule doesn't apply when date is outside increment."""
        rule_input = {"date": "2025-06-15", "amount": 1000.0, "rule": "future_income"}
        rule = CostIncomeRule(rule_input)

        increment = TimeIncrement(from_date=datetime.date(2025, 1, 1), to_date=datetime.date(2025, 1, 31))

        initial_cashflows = len(test_balance_sheet.cashflows)
        result_bs = rule.apply(test_balance_sheet, increment, market_rates)

        # Should not generate new cashflows since date is outside increment
        new_cashflows = len(result_bs.cashflows)
        assert new_cashflows == initial_cashflows

    def test_apply_prepaid_revenue(self, test_balance_sheet, market_rates):
        """Test prepaid revenue scenario (positive amount, cashflow before P&L)."""
        rule_input = {
            "date": "2024-12-15",
            "amount": 12000.0,
            "rule": "prepaid_revenue",
            "pnlstart": "2025-01-01",
            "pnlend": "2025-12-31",
        }
        rule = CostIncomeRule(rule_input)

        # Apply during the P&L period
        increment = TimeIncrement(from_date=datetime.date(2025, 1, 1), to_date=datetime.date(2025, 1, 31))

        result_bs = rule.apply(test_balance_sheet, increment, market_rates)

        # Should recognize portion of revenue during increment
        # Balance sheet should remain valid
        result_bs.validate()

    def test_apply_prepaid_expenses(self, test_balance_sheet, market_rates):
        """Test prepaid expenses scenario (negative amount, cashflow before P&L)."""
        rule_input = {
            "date": "2024-12-15",
            "amount": -12000.0,
            "rule": "prepaid_expenses",
            "pnlstart": "2025-01-01",
            "pnlend": "2025-12-31",
        }
        rule = CostIncomeRule(rule_input)

        # Apply during the P&L period
        increment = TimeIncrement(from_date=datetime.date(2025, 1, 1), to_date=datetime.date(2025, 1, 31))

        result_bs = rule.apply(test_balance_sheet, increment, market_rates)

        # Balance sheet should remain valid
        result_bs.validate()

    def test_apply_unpaid_revenue(self, test_balance_sheet, market_rates):
        """Test unpaid revenue scenario (positive amount, cashflow after P&L)."""
        rule_input = {
            "date": "2025-12-31",
            "amount": 12000.0,
            "rule": "unpaid_revenue",
            "pnlstart": "2025-01-01",
            "pnlend": "2025-12-31",
        }
        rule = CostIncomeRule(rule_input)

        # Apply during the P&L period
        increment = TimeIncrement(from_date=datetime.date(2025, 1, 1), to_date=datetime.date(2025, 1, 31))

        result_bs = rule.apply(test_balance_sheet, increment, market_rates)

        # Balance sheet should remain valid
        result_bs.validate()

    def test_apply_unpaid_expenses(self, test_balance_sheet, market_rates):
        """Test unpaid expenses scenario (negative amount, cashflow after P&L)."""
        rule_input = {
            "date": "2025-12-31",
            "amount": -12000.0,
            "rule": "unpaid_expenses",
            "pnlstart": "2025-01-01",
            "pnlend": "2025-12-31",
        }
        rule = CostIncomeRule(rule_input)

        # Apply during the P&L period
        increment = TimeIncrement(from_date=datetime.date(2025, 1, 1), to_date=datetime.date(2025, 1, 31))

        result_bs = rule.apply(test_balance_sheet, increment, market_rates)

        # Balance sheet should remain valid
        result_bs.validate()

    def test_apply_mid_period_revenue(self, test_balance_sheet, market_rates):
        """Test revenue with cashflow in the middle of the P&L period."""
        rule_input = {
            "date": "2025-06-15",
            "amount": 12000.0,
            "rule": "mid_period_revenue",
            "pnlstart": "2025-01-01",
            "pnlend": "2025-12-31",
        }
        rule = CostIncomeRule(rule_input)

        # Apply before cashflow date (should accrue unpaid revenue)
        increment = TimeIncrement(from_date=datetime.date(2025, 1, 1), to_date=datetime.date(2025, 1, 31))
        result_bs = rule.apply(test_balance_sheet, increment, market_rates)
        result_bs.validate()

    def test_apply_mid_period_expense(self, test_balance_sheet, market_rates):
        """Test expense with cashflow in the middle of the P&L period."""
        rule_input = {
            "date": "2025-06-15",
            "amount": -12000.0,
            "rule": "mid_period_expense",
            "pnlstart": "2025-01-01",
            "pnlend": "2025-12-31",
        }
        rule = CostIncomeRule(rule_input)

        # Apply before cashflow date (should accrue unpaid expense)
        increment = TimeIncrement(from_date=datetime.date(2025, 1, 1), to_date=datetime.date(2025, 1, 31))
        result_bs = rule.apply(test_balance_sheet, increment, market_rates)
        result_bs.validate()

    def test_apply_mid_period_on_cashflow_date(self, test_balance_sheet, market_rates):
        """Test cashflow within P&L period on the cashflow date itself."""
        rule_input = {
            "date": "2025-06-15",
            "amount": 12000.0,
            "rule": "mid_period_cashflow",
            "pnlstart": "2025-01-01",
            "pnlend": "2025-12-31",
        }
        rule = CostIncomeRule(rule_input)

        # Apply on the cashflow date
        increment = TimeIncrement(from_date=datetime.date(2025, 6, 1), to_date=datetime.date(2025, 6, 30))
        result_bs = rule.apply(test_balance_sheet, increment, market_rates)
        result_bs.validate()

    def test_apply_mid_period_after_cashflow(self, test_balance_sheet, market_rates):
        """Test cashflow within P&L period after the cashflow date."""
        rule_input = {
            "date": "2025-06-15",
            "amount": 12000.0,
            "rule": "mid_period_after",
            "pnlstart": "2025-01-01",
            "pnlend": "2025-12-31",
        }
        rule = CostIncomeRule(rule_input)

        # First apply on cashflow date to create the prepaid item
        increment1 = TimeIncrement(from_date=datetime.date(2025, 6, 1), to_date=datetime.date(2025, 6, 30))
        result_bs = rule.apply(test_balance_sheet, increment1, market_rates)

        # Then apply after cashflow date (should amortize prepaid item)
        increment2 = TimeIncrement(from_date=datetime.date(2025, 7, 1), to_date=datetime.date(2025, 7, 31))
        result_bs = rule.apply(result_bs, increment2, market_rates)
        result_bs.validate()
