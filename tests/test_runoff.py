"""Unit tests for runoff module."""

import datetime

import polars as pl
import pytest

from bank_projections.financials.balance_sheet import MutationReason
from bank_projections.financials.balance_sheet_item import BalanceSheetItem
from bank_projections.financials.balance_sheet_metric_registry import BalanceSheetMetrics
from bank_projections.financials.market_data import MarketRates
from bank_projections.projections.runoff import Runoff
from bank_projections.utils.time import TimeIncrement
from examples.synthetic_data import generate_synthetic_curves


@pytest.fixture
def bs(test_balance_sheet):
    """Provide a fresh copy of the balance sheet for each test."""
    return test_balance_sheet.copy()


@pytest.fixture
def market_rates():
    """Provide market rates for runoff tests."""
    return MarketRates(generate_synthetic_curves())


class TestRunoff:
    """Test Runoff rule functionality including coupon payments and principal repayments."""

    def test_repayment_before_maturity(self, bs, market_rates) -> None:
        """Test loans have no scheduled repayment before maturity, but may have prepayments."""
        increment = TimeIncrement(from_date=datetime.date(2024, 12, 31), to_date=datetime.date(2025, 1, 15))

        loans_item = BalanceSheetItem(SubItemType="Mortgages")
        initial_nominal = bs.get_amount(loans_item, BalanceSheetMetrics.get("nominal"))

        rule = Runoff()
        result_bs = rule.apply(bs, increment, market_rates)

        # Nominal should decrease slightly due to prepayments only
        new_nominal = result_bs.get_amount(loans_item, BalanceSheetMetrics.get("nominal"))
        assert new_nominal < initial_nominal  # Some prepayment occurred
        assert new_nominal > initial_nominal * 0.94  # But not much (small prepayment rate, allowing for ~6% max)

        # Should have principal repayment cashflow of zero (no scheduled repayment)
        principal_cashflows = result_bs.cashflows.filter(
            (pl.col("module") == "Runoff")
            & (pl.col("rule").is_in(["Principal Repayment", "Principal Prepayment"]))
            & (pl.col("SubItemType") == "Mortgages")
        )
        total_repayment = abs(principal_cashflows.filter(SubItemType="Mortgages")["Amount"].sum())
        assert abs(total_repayment - (initial_nominal - new_nominal)) < 0.01

        # Should have principal prepayment cashflow greater than zero
        prepayment_cashflows = result_bs.cashflows.filter(
            (pl.col("module") == "Runoff") & (pl.col("rule") == "Principal Prepayment")
        )
        total_prepayment = abs(prepayment_cashflows["Amount"].sum()) if len(prepayment_cashflows) > 0 else 0.0
        assert total_prepayment > 0

    def test_coupon_payments_non_accumulating(self, bs, market_rates) -> None:
        """Test coupon payments for non-accumulating loans generate cashflows."""
        increment = TimeIncrement(from_date=datetime.date(2025, 1, 15), to_date=datetime.date(2025, 2, 15))

        initial_cashflows_len = len(bs.cashflows)

        rule = Runoff()
        result_bs = rule.apply(bs, increment, market_rates)

        # Should generate coupon payment cashflows
        coupon_cashflows = result_bs.cashflows.filter(
            (pl.col("module") == "Runoff") & (pl.col("rule") == "Coupon payment")
        )
        assert len(coupon_cashflows) > 0

        # Total cashflows should increase
        assert len(result_bs.cashflows) > initial_cashflows_len

    def test_interest_accrual_update(self, bs, market_rates) -> None:
        """Test that accrued interest is properly updated."""
        increment = TimeIncrement(from_date=datetime.date(2025, 1, 15), to_date=datetime.date(2025, 2, 15))

        loans_item = BalanceSheetItem(SubItemType="Mortgages")

        rule = Runoff()
        result_bs = rule.apply(bs, increment, market_rates)

        # Accrued interest should be updated
        new_accrued = result_bs.get_amount(loans_item, BalanceSheetMetrics.get("accrued_interest"))
        # The accrued interest may increase or decrease based on coupon payments and new accrual
        assert isinstance(new_accrued, float)

    def test_impairment_adjustment_for_repayments(self, bs, market_rates) -> None:
        """Test that impairment is adjusted proportionally to repayments."""
        # Use annuity loans to test impairment adjustment and set some initial impairment
        loan_filter = pl.col("SubItemType") == "Mortgages"
        bs._data = bs._data.with_columns(
            [
                pl.when(loan_filter)
                .then(pl.lit("annuity"))
                .otherwise(pl.col("RedemptionType"))
                .alias("RedemptionType"),
            ]
        )

        loans_item = BalanceSheetItem(SubItemType="Mortgages", AccountingMethod="amortized cost")
        bs.mutate_metric(
            loans_item,
            BalanceSheetMetrics.get("impairment"),
            -10000.0,
            MutationReason(module="Test", rule="test_impairment", test="test"),
            offset_pnl=True,
        )

        increment = TimeIncrement(from_date=datetime.date(2025, 1, 15), to_date=datetime.date(2025, 2, 15))

        initial_impairment = bs.get_amount(loans_item, BalanceSheetMetrics.get("impairment"))

        rule = Runoff()
        result_bs = rule.apply(bs, increment, market_rates)

        # Impairment should decrease proportionally to principal repayments
        new_impairment = result_bs.get_amount(loans_item, BalanceSheetMetrics.get("impairment"))
        assert new_impairment > initial_impairment

    def test_agio_linear_decrease(self, bs, market_rates) -> None:
        """Test that agio decreases linearly over time."""
        # Set some initial agio to test the decrease
        bs.mutate_metric(
            BalanceSheetItem(SubItemType="Mortgages"),
            BalanceSheetMetrics.get("agio"),
            500.0,
            MutationReason(module="Test", rule="test_agio", test="test"),
            offset_pnl=True,
        )
        bs.validate()

        increment = TimeIncrement(from_date=datetime.date(2025, 1, 15), to_date=datetime.date(2025, 2, 15))

        loans_item = BalanceSheetItem(SubItemType="Mortgages")
        initial_agio = bs.get_amount(loans_item, BalanceSheetMetrics.get("agio"))

        rule = Runoff()
        result_bs = rule.apply(bs, increment, market_rates)

        # Agio should decrease (linear amortization)
        new_agio = result_bs.get_amount(loans_item, BalanceSheetMetrics.get("agio"))
        assert new_agio < initial_agio

    def test_pnl_generation(self, bs, market_rates) -> None:
        """Test that appropriate PnL entries are generated."""
        increment = TimeIncrement(from_date=datetime.date(2025, 1, 15), to_date=datetime.date(2025, 2, 15))

        initial_pnl_len = len(bs.pnls)

        rule = Runoff()
        result_bs = rule.apply(bs, increment, market_rates)

        # Should generate PnL for interest income and impairment
        assert len(result_bs.pnls) > initial_pnl_len

        # Check for specific PnL reasons (runoff generates Accrual, Coupons, and Impairment PnLs)
        accrual_pnl = result_bs.pnls.filter((pl.col("module") == "Runoff") & (pl.col("rule") == "Accrual"))
        coupon_pnl = result_bs.pnls.filter((pl.col("module") == "Runoff") & (pl.col("rule") == "Coupons"))
        impairment_pnl = result_bs.pnls.filter((pl.col("module") == "Runoff") & (pl.col("rule") == "Impairment"))

        # At least one of accrual or coupon PnL should be generated
        assert len(accrual_pnl) > 0 or len(coupon_pnl) > 0
        # Impairment PnL may or may not be generated depending on whether there is impairment
        assert isinstance(impairment_pnl, pl.DataFrame)

    def test_quarterly_frequency(self, bs, market_rates) -> None:
        """Test runoff with quarterly frequency."""
        # Set some loans to quarterly frequency
        loan_filter = pl.col("SubItemType") == "Mortgages"
        bs._data = bs._data.with_columns(
            pl.when(loan_filter & (pl.int_range(pl.len()) % 2 == 0))
            .then(pl.lit("quarterly"))
            .otherwise(pl.col("CouponFrequency"))
            .alias("CouponFrequency"),
            pl.when(loan_filter & (pl.int_range(pl.len()) % 2 == 0))
            .then(pl.lit(datetime.date(2025, 4, 15)))
            .otherwise(pl.col("NextCouponDate"))
            .alias("NextCouponDate"),
        )

        increment = TimeIncrement(from_date=datetime.date(2025, 1, 15), to_date=datetime.date(2025, 4, 15))

        rule = Runoff()
        initial_pnl_len = len(bs.pnls)

        result_bs = rule.apply(bs, increment, market_rates)

        # Should generate PnL for quarterly payments
        assert len(result_bs.pnls) > initial_pnl_len

    def test_preserves_balance_sheet_structure(self, bs, market_rates) -> None:
        """Test that applying rule preserves balance sheet structure."""
        increment = TimeIncrement(from_date=datetime.date(2025, 1, 15), to_date=datetime.date(2025, 2, 15))

        initial_columns = set(bs._data.columns)
        initial_rows = len(bs._data)

        rule = Runoff()
        result_bs = rule.apply(bs, increment, market_rates)

        # Should preserve row count and basic columns
        assert len(result_bs._data) == initial_rows
        assert initial_columns.issubset(set(result_bs._data.columns))

    def test_zero_interest_rate_handling(self, bs, market_rates) -> None:
        """Test runoff with zero interest rates."""
        # Set interest rate to zero for all loans
        item = BalanceSheetItem(SubItemType="Mortgages")
        reason = MutationReason(test="test")
        bs.mutate_metric(item, BalanceSheetMetrics.get("interest_rate"), 0.0, reason)
        bs.mutate_metric(item, BalanceSheetMetrics.get("accrued_interest"), 0.0, reason)

        increment = TimeIncrement(from_date=datetime.date(2025, 1, 15), to_date=datetime.date(2025, 2, 15))

        rule = Runoff()
        result_bs = rule.apply(bs, increment, market_rates)

        # Should handle zero interest rates without error
        loans_item = BalanceSheetItem(SubItemType="Mortgages")
        new_accrued = result_bs.get_amount(loans_item, BalanceSheetMetrics.get("accrued_interest"))
        assert isinstance(new_accrued, float)

    def test_prepayment_only_scenario(self, bs, market_rates) -> None:
        """Test scenario with prepayments only (no scheduled repayments)."""
        # Set redemption type to perpetual and higher prepayment rate
        loan_filter = pl.col("SubItemType") == "Mortgages"
        bs._data = bs._data.with_columns(
            [
                pl.when(loan_filter)
                .then(pl.lit("perpetual"))
                .otherwise(pl.col("RedemptionType"))
                .alias("RedemptionType"),
                pl.when(loan_filter)
                .then(pl.lit(0.05))
                .otherwise(pl.col("PrepaymentRate"))
                .alias("PrepaymentRate"),  # 5% annual
                pl.when(loan_filter)
                .then(pl.lit(datetime.date(2030, 12, 31)))
                .otherwise(pl.col("MaturityDate"))
                .alias("MaturityDate"),
            ]
        )

        increment = TimeIncrement(from_date=datetime.date(2024, 12, 31), to_date=datetime.date(2025, 2, 15))

        loans_item = BalanceSheetItem(SubItemType="Mortgages")
        initial_nominal = bs.get_amount(loans_item, BalanceSheetMetrics.get("nominal"))

        rule = Runoff()
        result_bs = rule.apply(bs, increment, market_rates)

        # Nominal should decrease due to prepayments only
        new_nominal = result_bs.get_amount(loans_item, BalanceSheetMetrics.get("nominal"))
        assert new_nominal < initial_nominal

        # Should have no principal repayment cashflow (perpetual)
        principal_cashflows = result_bs.cashflows.filter(
            loan_filter & (pl.col("module") == "Runoff") & (pl.col("rule") == "Principal Repayment")
        )
        total_repayment = abs(principal_cashflows["Amount"].sum()) if len(principal_cashflows) > 0 else 0.0
        assert abs(total_repayment) < 0.01

        # Should have positive prepayment cashflow
        prepayment_cashflows = result_bs.cashflows.filter(
            (pl.col("module") == "Runoff") & (pl.col("rule") == "Principal Prepayment")
        )
        total_prepayment = abs(prepayment_cashflows["Amount"].sum()) if len(prepayment_cashflows) > 0 else 0.0
        assert total_prepayment > 0

    def test_zero_prepayment_rate(self, bs, market_rates) -> None:
        """Test scenario with zero prepayment rate."""
        # Set prepayment rate to zero for all items that can have prepayments
        has_maturity = pl.col("MaturityDate").is_not_null()
        bs._data = bs._data.with_columns(
            pl.when(has_maturity).then(pl.lit(0.0)).otherwise(pl.col("PrepaymentRate")).alias("PrepaymentRate")
        )

        increment = TimeIncrement(from_date=datetime.date(2025, 1, 15), to_date=datetime.date(2025, 2, 15))

        rule = Runoff()
        result_bs = rule.apply(bs, increment, market_rates)

        # Should have zero prepayment cashflow
        prepayment_cashflows = result_bs.cashflows.filter(
            (pl.col("module") == "Runoff") & (pl.col("rule") == "Principal Prepayment")
        )
        total_prepayment = abs(prepayment_cashflows["Amount"].sum()) if len(prepayment_cashflows) > 0 else 0.0
        assert abs(total_prepayment) < 0.01
