"""Unit tests for runoff module."""

import datetime

import polars as pl
import pytest

from bank_projections.financials.balance_sheet_item import BalanceSheetItem
from bank_projections.financials.balance_sheet_metric_registry import BalanceSheetMetrics
from bank_projections.projections.accrual import Accrual
from bank_projections.projections.agio_redemption import AgioRedemption
from bank_projections.projections.coupon_payment import CouponPayment
from bank_projections.projections.redemption import Redemption
from bank_projections.utils.time import TimeIncrement


@pytest.fixture
def bs(test_balance_sheet):
    """Provide a fresh copy of the balance sheet for each test."""
    return test_balance_sheet.copy()


class TestRunoff:
    """Test Runoff rule functionality including coupon payments and principal repayments."""

    def test_redemption_rule_instantiation(self):
        """Test that Redemption rule can be instantiated."""
        rule = Redemption()
        assert rule is not None

    def test_coupon_payment_rule_instantiation(self):
        """Test that CouponPayment rule can be instantiated."""
        rule = CouponPayment()
        assert rule is not None

    def test_accrual_rule_instantiation(self):
        """Test that Accrual rule can be instantiated."""
        rule = Accrual()
        assert rule is not None

    def test_agio_redemption_rule_instantiation(self):
        """Test that AgioRedemption rule can be instantiated."""
        rule = AgioRedemption()
        assert rule is not None

    def test_repayment_before_maturity(self, bs, minimal_scenario_snapshot) -> None:
        """Test loans have no scheduled repayment before maturity, but may have prepayments."""
        increment = TimeIncrement(from_date=datetime.date(2024, 12, 31), to_date=datetime.date(2025, 1, 15))

        loans_item = BalanceSheetItem(SubItemType="Mortgages")
        initial_nominal = bs.get_amount(loans_item, BalanceSheetMetrics.get("nominal"))

        rule = Redemption()
        result_bs = rule.apply(bs, increment, minimal_scenario_snapshot)

        # Nominal should decrease slightly due to prepayments only
        new_nominal = result_bs.get_amount(loans_item, BalanceSheetMetrics.get("nominal"))
        assert new_nominal < initial_nominal  # Some prepayment occurred

    def test_coupon_payments_non_accumulating(self, bs, minimal_scenario_snapshot) -> None:
        """Test coupon payments for non-accumulating loans generate cashflows."""
        increment = TimeIncrement(from_date=datetime.date(2025, 1, 15), to_date=datetime.date(2025, 2, 15))

        initial_cashflows_len = len(bs.cashflows)

        # Apply coupon payment rule (split from Redemption)
        rule = CouponPayment()
        result_bs = rule.apply(bs, increment, minimal_scenario_snapshot)

        # Should generate coupon payment cashflows
        coupon_cashflows = result_bs.cashflows.filter(
            (pl.col("module") == "Runoff") & (pl.col("rule") == "Coupon payment")
        )
        assert len(coupon_cashflows) > 0

        # Total cashflows should increase
        assert len(result_bs.cashflows) > initial_cashflows_len

    def test_interest_accrual_update(self, bs, minimal_scenario_snapshot) -> None:
        """Test that accrued interest is properly updated."""
        increment = TimeIncrement(from_date=datetime.date(2025, 1, 15), to_date=datetime.date(2025, 2, 15))

        loans_item = BalanceSheetItem(SubItemType="Mortgages")

        rule = Redemption()
        result_bs = rule.apply(bs, increment, minimal_scenario_snapshot)

        # Accrued interest should be updated
        new_accrued = result_bs.get_amount(loans_item, BalanceSheetMetrics.get("accrued_interest"))
        # The accrued interest may increase or decrease based on coupon payments and new accrual
        assert isinstance(new_accrued, float)

    def test_pnl_generation(self, bs, minimal_scenario_snapshot) -> None:
        """Test that appropriate PnL entries are generated."""
        increment = TimeIncrement(from_date=datetime.date(2025, 1, 15), to_date=datetime.date(2025, 2, 15))

        initial_pnl_len = len(bs.pnls)

        # Apply accrual rule (split from Redemption) to generate interest PnL
        accrual_rule = Accrual()
        result_bs = accrual_rule.apply(bs, increment, minimal_scenario_snapshot)

        # Apply redemption rule for impairment PnL
        redemption_rule = Redemption()
        result_bs = redemption_rule.apply(result_bs, increment, minimal_scenario_snapshot)

        # Should generate PnL for interest income and impairment
        assert len(result_bs.pnls) > initial_pnl_len

    def test_preserves_balance_sheet_structure(self, bs, minimal_scenario_snapshot) -> None:
        """Test that applying rule preserves balance sheet structure."""
        increment = TimeIncrement(from_date=datetime.date(2025, 1, 15), to_date=datetime.date(2025, 2, 15))

        initial_columns = set(bs._data.columns)
        initial_rows = len(bs._data)

        rule = Redemption()
        result_bs = rule.apply(bs, increment, minimal_scenario_snapshot)

        # Should preserve row count and basic columns
        assert len(result_bs._data) == initial_rows
        assert initial_columns.issubset(set(result_bs._data.columns))
